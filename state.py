# state.py
from dataclasses import dataclass
from typing import Tuple, List, Optional, Literal, Dict
import numpy as np

Dir = Literal["UP", "DOWN", "LEFT", "RIGHT"]

# Row/Col movement on a numpy board (r is y, c is x)
DIR_VEC = {
    "UP":    (-1,  0),
    "DOWN":  ( 1,  0),
    "LEFT":  ( 0, -1),
    "RIGHT": ( 0,  1),
}

DIRS: List[Dir] = ["UP", "LEFT", "RIGHT", "DOWN"]  # deterministic order


@dataclass(frozen=True)
class State:
    board: np.ndarray
    H: int
    W: int
    me_head: Tuple[int, int]
    opp_head: Tuple[int, int]
    me_trail: Tuple[Tuple[int, int], ...]
    opp_trail: Tuple[Tuple[int, int], ...]
    me_len: int
    opp_len: int
    me_alive: bool
    opp_alive: bool
    me_boosts: int
    opp_boosts: int
    turn: int
    me_last_dir: Optional[Dir]
    opp_last_dir: Optional[Dir]
    role: Literal["agent1", "agent2"]

    # ---------- geometry helpers ----------
    def wrap(self, r: int, c: int) -> Tuple[int, int]:
        return (r % self.H, c % self.W)

    def next_pos(self, pos: Tuple[int, int], d: Dir) -> Tuple[int, int]:
        dr, dc = DIR_VEC[d]
        return self.wrap(pos[0] + dr, pos[1] + dc)

    def is_open(self, r: int, c: int) -> bool:
        rr, cc = self.wrap(r, c)
        return self.board[rr, cc] == 0

    def is_safe_dir_from(self, pos: Tuple[int, int], d: Dir) -> bool:
        nr, nc = self.next_pos(pos, d)
        return self.board[nr, nc] == 0

    def legal_dirs_from(self, pos: Tuple[int, int]) -> List[Dir]:
        return [d for d in DIRS if self.is_safe_dir_from(pos, d)]

    def legal_dirs_me(self) -> List[Dir]:
        return self.legal_dirs_from(self.me_head)

    def legal_dirs_opp(self) -> List[Dir]:
        return self.legal_dirs_from(self.opp_head)

    # ---------- simulation helpers used by heuristics/search ----------
    def after_me(self, move: Dir) -> "State":
        return self._after_generic(move, who="me", boost=False)

    def after_opp(self, move: Dir) -> "State":
        return self._after_generic(move, who="opp", boost=False)

    def after_me_boost(self, move: Dir) -> "State":
        return self._after_generic(move, who="me", boost=True)

    def _after_generic(self, move: Dir, who: Literal["me","opp"], boost: bool) -> "State":
        # validate move
        if move not in ("UP", "DOWN", "LEFT", "RIGHT"):
            return self  # return unchanged if nonsense

        board = self.board.copy()
        me_head, opp_head = self.me_head, self.opp_head
        me_trail = list(self.me_trail)
        opp_trail = list(self.opp_trail)

        steps = 2 if boost else 1
        head = me_head if who == "me" else opp_head
        trail = me_trail if who == "me" else opp_trail

        alive_me  = self.me_alive
        alive_opp = self.opp_alive

        for _ in range(steps):
            head = self.next_pos(head, move)
            r, c = head

            # collision with any wall/trail
            if board[r, c] != 0:
                if who == "me":
                    alive_me = False
                    me_head = head
                    me_trail = tuple(me_trail + [head])
                else:
                    alive_opp = False
                    opp_head = head
                    opp_trail = tuple(opp_trail + [head])
                return State(
                    board=board, H=self.H, W=self.W,
                    me_head=me_head if who=="opp" else head,
                    opp_head=opp_head if who=="me" else head,
                    me_trail=tuple(me_trail) if who=="opp" else tuple(trail + [head]),
                    opp_trail=tuple(opp_trail) if who=="me" else tuple(trail + [head]),
                    me_len=self.me_len + (1 if who=="me" else 0),
                    opp_len=self.opp_len + (1 if who=="opp" else 0),
                    me_alive=alive_me, opp_alive=alive_opp,
                    me_boosts=self.me_boosts - (1 if who=="me" and boost else 0),
                    opp_boosts=self.opp_boosts - (1 if who=="opp" and boost else 0),
                    turn=self.turn + 1,
                    me_last_dir=move if who=="me" else self.me_last_dir,
                    opp_last_dir=move if who=="opp" else self.opp_last_dir,
                    role=self.role
                )

            # normal step: stamp and grow
            board[r, c] = 1
            trail.append(head)

        # commit head/trail after all steps
        if who == "me":
            me_head, me_trail = head, trail
        else:
            opp_head, opp_trail = head, trail

        return State(
            board=board, H=self.H, W=self.W,
            me_head=me_head, opp_head=opp_head,
            me_trail=tuple(me_trail), opp_trail=tuple(opp_trail),
            me_len=self.me_len + (2 if (who=="me" and boost) else (1 if who=="me" else 0)),
            opp_len=self.opp_len + (2 if (who=="opp" and boost) else (1 if who=="opp" else 0)),
            me_alive=alive_me, opp_alive=alive_opp,
            me_boosts=self.me_boosts - (1 if who=="me" and boost else 0),
            opp_boosts=self.opp_boosts - (1 if who=="opp" and boost else 0),
            turn=self.turn + 1,
            me_last_dir=move if who=="me" else self.me_last_dir,
            opp_last_dir=move if who=="opp" else self.opp_last_dir,
            role=self.role
        )


# ---------- internal utils ----------

def _swap_xy_to_rc(trail_like) -> Tuple[Tuple[int, int], ...]:
    """
    Engine sends positions as (x, y) = (col, row).
    Convert to (r, c) = (y, x) for numpy row/col access.
    """
    try:
        return tuple((int(p[1]), int(p[0])) for p in trail_like)
    except Exception:
        return tuple()

def _derive_head(trail: Tuple[Tuple[int, int], ...]) -> Optional[Tuple[int, int]]:
    return trail[-1] if trail else None

def _derive_last_dir(trail: Tuple[Tuple[int, int], ...], H: int, W: int) -> Optional[Dir]:
    if len(trail) < 2:
        return None
    (r1, c1), (r2, c2) = trail[-2], trail[-1]
    dr = r2 - r1
    dc = c2 - c1
    # handle torus wrap to normalize to {-1,0,1}
    if dr == 0:
        pass
    elif dr > 0 and dr != 1 and (r1 == H-1 and r2 == 0):
        dr = 1
    elif dr < 0 and dr != -1 and (r1 == 0 and r2 == H-1):
        dr = -1
    else:
        dr = max(-1, min(1, dr))

    if dc == 0:
        pass
    elif dc > 0 and dc != 1 and (c1 == W-1 and c2 == 0):
        dc = 1
    elif dc < 0 and dc != -1 and (c1 == 0 and c2 == W-1):
        dc = -1
    else:
        dc = max(-1, min(1, dc))

    if (dr, dc) == (-1, 0): return "UP"
    if (dr, dc) == ( 1, 0): return "DOWN"
    if (dr, dc) == ( 0,-1): return "LEFT"
    if (dr, dc) == ( 0, 1): return "RIGHT"
    return None

def _ensure_board(board_like, H: int, W: int) -> np.ndarray:
    if board_like is None:
        arr = np.zeros((H, W), dtype=np.int8)
    else:
        arr = np.asarray(board_like, dtype=np.int8)
        if arr.shape != (H, W):
            try:
                arr = arr.reshape(H, W)
            except Exception:
                arr = np.zeros((H, W), dtype=np.int8)
        arr = (arr != 0).astype(np.int8)
    return arr

def _stamp_trail_as_walls(board: np.ndarray, trail: Tuple[Tuple[int,int], ...]) -> None:
    for (r, c) in trail:
        rr, cc = r % board.shape[0], c % board.shape[1]
        board[rr, cc] = 1

def parse_state(data: Dict, role: Literal["agent1","agent2"]="agent1",
                H: int=18, W: int=20) -> State:
    raw_board = data.get("board", None)

    # Convert incoming (x,y) → (r,c)
    a1_trail = _swap_xy_to_rc(data.get("agent1_trail", []))
    a2_trail = _swap_xy_to_rc(data.get("agent2_trail", []))

    a1_len = int(data.get("agent1_length", len(a1_trail) or 1))
    a2_len = int(data.get("agent2_length", len(a2_trail) or 1))
    a1_alive = bool(data.get("agent1_alive", True))
    a2_alive = bool(data.get("agent2_alive", True))
    a1_boosts = int(data.get("agent1_boosts", 0))
    a2_boosts = int(data.get("agent2_boosts", 0))
    turn = int(data.get("turn_count", 0))

    board = _ensure_board(raw_board, H, W)

    # Ensure trails are walls on the board
    _stamp_trail_as_walls(board, a1_trail)
    _stamp_trail_as_walls(board, a2_trail)

    a1_head = _derive_head(a1_trail) or (0, 0)
    a2_head = _derive_head(a2_trail) or (0, 1)
    a1_last = _derive_last_dir(a1_trail, H, W)
    a2_last = _derive_last_dir(a2_trail, H, W)

    if role == "agent1":
        me_head, opp_head = a1_head, a2_head
        me_trail, opp_trail = a1_trail, a2_trail
        me_len, opp_len = a1_len, a2_len
        me_alive, opp_alive = a1_alive, a2_alive
        me_boosts, opp_boosts = a1_boosts, a2_boosts
        me_last, opp_last = a1_last, a2_last
    else:
        me_head, opp_head = a2_head, a1_head
        me_trail, opp_trail = a2_trail, a1_trail
        me_len, opp_len = a2_len, a1_len
        me_alive, opp_alive = a2_alive, a1_alive
        me_boosts, opp_boosts = a2_boosts, a1_boosts
        me_last, opp_last = a2_last, a1_last

    return State(
        board=board, H=H, W=W,
        me_head=me_head, opp_head=opp_head,
        me_trail=me_trail, opp_trail=opp_trail,
        me_len=me_len, opp_len=opp_len,
        me_alive=me_alive, opp_alive=opp_alive,
        me_boosts=me_boosts, opp_boosts=opp_boosts,
        turn=turn,
        me_last_dir=me_last, opp_last_dir=opp_last,
        role=role
    )
