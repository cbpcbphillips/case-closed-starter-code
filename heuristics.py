# heuristics.py
from collections import deque
from typing import List, Tuple, Set
import numpy as np
from state import State, DIRS

# =========================
# Weights / knobs to tune
# =========================
W_SPACE        = 1.20    # my_reachable - opp_reachable_est
W_BRANCH       = 0.20    # branching factor after my move
W_TUNNEL       = -0.18   # penalty for long 1-wide corridors
W_STRAIGHT     = 0.06    # slight bias to keep direction (mainly early)
W_CHOKE        = -0.40   # my branching<=1 while opp branching>=2
W_HEADON_WIN   = 0.30    # reward if a (legal) head-on would be winning by length
W_VORONOI      = 0.35    # bonus for cutting more territory than opponent
W_STEP_IN_LOSS = -1e9    # hard block for illegal/losing options
W_THREAT2      = -0.25   # soft penalty: opp can reach within 2 ticks

EPS_NEAR_TIE   = 3.0     # points within which moves are considered near-ties
BOOST_DELTA_T  = 8       # min space gain to allow a boost
NOISE_SCALE    = 0.05    # tiny deterministic noise to break symmetry

# =========================
# Basic helpers
# =========================
def flood_fill(board: np.ndarray, start: Tuple[int, int]) -> int:
    """Count reachable open cells from `start` on a torus grid."""
    H, W = board.shape
    if board[start] != 0:
        return 0
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            rr = (r+dr) % H; cc = (c+dc) % W
            if board[rr, cc] == 0:
                nxt = (rr, cc)
                if nxt not in seen:
                    seen.add(nxt); q.append(nxt)
    return len(seen)

def branching(board: np.ndarray, head: Tuple[int,int]) -> int:
    """Number of open neighbors around head (0..4), wrap-aware."""
    H, W = board.shape
    r, c = head
    b = 0
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
        rr = (r+dr) % H; cc = (c+dc) % W
        if board[rr, cc] == 0:
            b += 1
    return b

def tunnel_depth(board: np.ndarray, head: Tuple[int,int], max_depth: int = 6) -> int:
    """Simple estimate of 1-wide corridor depth reachable from head."""
    H, W = board.shape
    depth = 0
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
        r, c = head; d = 0
        while d < max_depth:
            r = (r+dr) % H; c = (c+dc) % W
            if board[r, c] != 0:
                break
            # branching at this new cell
            b = 0
            for xr, xc in ((-1,0),(1,0),(0,-1),(0,1)):
                rr = (r+xr) % H; cc = (c+xc) % W
                if board[rr, cc] == 0:
                    b += 1
            if b > 2:  # corridor likely opens here
                break
            d += 1
        depth = max(depth, d)
    return depth

# =========================
# Opponent threat modeling
# =========================
def candidate_opp_dirs(s: State) -> List[str]:
    """All plausible opponent moves this tick (non-crashing), prefer straight."""
    dirs = []
    if s.opp_last_dir and s.is_safe_dir_from(s.opp_head, s.opp_last_dir):
        dirs.append(s.opp_last_dir)
    for d in DIRS:
        if d != s.opp_last_dir and s.is_safe_dir_from(s.opp_head, d):
            dirs.append(d)
    return dirs or DIRS[:]  # if terminal, return something benign

def opp_threat_cells_1(s: State) -> Set[Tuple[int,int]]:
    """Cells opponent can occupy after this tick (simultaneous move)."""
    return {s.next_pos(s.opp_head, d) for d in candidate_opp_dirs(s)}

def opp_threat_cells_2(s: State) -> Set[Tuple[int,int]]:
    """Cells opponent can reach within 2 ticks (rough, for soft avoidance)."""
    H, W = s.board.shape
    cells = set()
    step1 = []
    for d1 in candidate_opp_dirs(s):
        n1 = s.next_pos(s.opp_head, d1)
        if s.board[n1] == 0:
            step1.append(n1)
            cells.add(n1)
    for n1 in step1:
        r, c = n1
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            rr = (r+dr) % H; cc = (c+dc) % W
            if s.board[rr, cc] == 0:
                cells.add((rr, cc))
    return cells

def head_swap_threat(s: State) -> bool:
    """True if opponent can move onto our current head (potential head-swap)."""
    for d in candidate_opp_dirs(s):
        if s.next_pos(s.opp_head, d) == s.me_head:
            return True
    return False

def predict_opp_dir(s: State) -> str:
    """Cheap single-dir predictor (used for opp_reachable estimate)."""
    dirs = candidate_opp_dirs(s)
    return dirs[0] if dirs else DIRS[0]

# =========================
# Symmetry breaking helpers
# =========================
def _hash_noise(s: State, move: str) -> float:
    """Deterministic tiny noise to break symmetry in near-ties."""
    mcode = {"UP":1, "DOWN":2, "LEFT":3, "RIGHT":4}[move]
    h = (s.turn * 2654435761) ^ (s.me_head[0] << 16) ^ (s.me_head[1] << 8) ^ mcode
    h ^= (h >> 13); h = (h * 0x5bd1e995) & 0xFFFFFFFF; h ^= (h >> 15)
    # map to [-1,1]
    return ((h & 0xFFFF) / 32767.5 - 1.0) * NOISE_SCALE

def _voronoi_split(board: np.ndarray, a: Tuple[int,int], b: Tuple[int,int]) -> Tuple[int,int]:
    """
    Label open cells by which head is closer (BFS on torus). Ties ignored.
    Returns (mine, theirs).
    """
    H, W = board.shape
    mine = theirs = 0
    dist_a = {a: 0}; dist_b = {b: 0}
    qa, qb = deque([a]), deque([b])

    def expand(q, dist):
        r, c = q.popleft()
        d = dist[(r,c)]
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            rr = (r+dr) % H; cc = (c+dc) % W
            if board[rr,cc] == 0 and (rr,cc) not in dist:
                dist[(rr,cc)] = d+1
                q.append((rr,cc))

    while qa: expand(qa, dist_a)
    while qb: expand(qb, dist_b)

    for (r,c), da in dist_a.items():
        if board[r,c] != 0:
            continue
        db = dist_b.get((r,c))
        if db is None or da < db:
            mine += 1
    for (r,c), db in dist_b.items():
        if board[r,c] != 0:
            continue
        da = dist_a.get((r,c))
        if da is None or db < da:
            theirs += 1
    return mine, theirs

# =========================
# Distance helpers (for anti-convergence)
# =========================
def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def chebyshev(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

# =========================
# Move legality with threats (stricter)
# =========================
def safe_moves(s: State) -> List[str]:
    """
    Legal moves that avoid:
    - walls/trails,
    - contested next cells (opp can step there this tick),
    - head-swap,
    - AND (if alternatives exist) opponent's 2-tick threat & distance reductions when close.
    """
    t1 = opp_threat_cells_1(s)
    t2 = opp_threat_cells_2(s)  # soft, but we’ll filter if alternatives exist
    swap_possible = head_swap_threat(s)

    # 1) base legality (walls) + hard contested & swap rules
    base = []
    for d in DIRS:
        nr, nc = s.next_pos(s.me_head, d)

        # walls/trails
        if s.board[nr, nc] != 0:
            continue

        # contested same-cell this tick (only allow if strictly longer)
        if (nr, nc) in t1 and not (s.me_len > s.opp_len):
            continue

        # head-swap (only allow if strictly longer)
        if swap_possible and (nr, nc) == s.opp_head and not (s.me_len > s.opp_len):
            continue

        base.append(d)

    if not base:
        return []

    # 2) if any moves avoid opp 2-tick threat, prefer only those
    avoid_t2 = [d for d in base if s.next_pos(s.me_head, d) not in t2]
    if avoid_t2:
        base = avoid_t2

    # 3) if we’re close (<=3 by Chebyshev), avoid moves that REDUCE distance
    #    to the opponent when there exists at least one that doesn’t.
    dist0 = chebyshev(s.me_head, s.opp_head)
    if dist0 <= 3 and not (s.me_len > s.opp_len):
        non_reduce = []
        for d in base:
            nxt = s.next_pos(s.me_head, d)
            if chebyshev(nxt, s.opp_head) >= dist0:
                non_reduce.append(d)
        if non_reduce:
            base = non_reduce

    return base

# =========================
# Scoring & selection
# =========================
def score_move(s: State, move: str) -> float:
    # Guard: if move is currently unsafe by our filter, block it
    if move not in safe_moves(s):
        return W_STEP_IN_LOSS

    # Precompute soft threat map (2 ticks)
    t2 = opp_threat_cells_2(s)

    # simulate my step
    board1 = s.board.copy()
    me_next = s.next_pos(s.me_head, move)
    board1[me_next] = 1

    # opponent predicted reply on a copy that includes my step
    opp_move = predict_opp_dir(s)
    opp_next = s.next_pos(s.opp_head, opp_move)
    board2 = board1.copy()
    if board2[opp_next] == 0:
        board2[opp_next] = 1

    # reachability (space control)
    my_space  = flood_fill(board1, me_next)
    opp_space = flood_fill(board2, opp_next)

    # local shape features
    br  = branching(board1, me_next)
    tun = tunnel_depth(board1, me_next)

    # choke indicator
    opp_br = branching(board2, opp_next)
    choke_flag = 1 if (br <= 1 and opp_br >= 2) else 0

    # head-on indicator (predicted path; immediate ties already filtered)
    headon_flag = 0
    if me_next == opp_next:
        if s.me_len > s.opp_len:
            headon_flag = 1
        elif s.me_len < s.opp_len:
            headon_flag = -1

    # Voronoi split after my step (before opp writes theirs)
    v_mine, v_theirs = _voronoi_split(board1, me_next, s.opp_head)
    voronoi_gain = (v_mine - v_theirs)

    # soft threat penalty (2-tick reach) + tiny deterministic noise
    threat_pen = W_THREAT2 if (me_next in t2) else 0.0
    noise = _hash_noise(s, move)

    score = (
        W_SPACE        * (my_space - opp_space)
        + W_BRANCH     * (br - 2)
        + W_TUNNEL     * (tun)
        + W_STRAIGHT   * (1 if (s.me_last_dir and move == s.me_last_dir) else 0)
        + W_CHOKE      * (choke_flag)
        + W_HEADON_WIN * (headon_flag)
        + W_VORONOI    * (voronoi_gain)
        + threat_pen
        + noise
    )
    return score

def choose_by_heuristic(s: State) -> str:
    legal = safe_moves(s)
    if not legal:
        return "UP"  # no legal moves; return something valid

    best = None; best_score = float("-inf"); near = []
    for m in legal:
        sc = score_move(s, m)
        if sc > best_score + 1e-9:
            best, best_score, near = m, sc, [(m, sc)]
        elif abs(sc - best_score) <= EPS_NEAR_TIE:
            near.append((m, sc))

    # tie breaks: keep straight if possible; else more branching
    if len(near) > 1 and s.me_last_dir:
        straight = [m for m,_ in near if m == s.me_last_dir]
        if straight:
            return straight[0]
        def br_for(m):
            b = s.board.copy()
            nxt = s.next_pos(s.me_head, m); b[nxt] = 1
            return branching(b, nxt)
        return max((m for m,_ in near), key=br_for)

    return best or legal[0]

# =========================
# Boost policy
# =========================
def maybe_apply_boost(s: State, move: str, threshold: int = BOOST_DELTA_T) -> str:
    """
    Minimal, safe boost policy:
    - Only consider boosting if baseline landing is safe and uncontested.
    - Boost landing must also be safe and uncontested.
    - Require space gain AND favor boosts that improve Voronoi split.
    """
    if s.me_boosts <= 0:
        return move

    t1 = opp_threat_cells_1(s)  # opponent’s next-tick cells

    # baseline landing
    b0 = s.board.copy()
    n0 = s.next_pos(s.me_head, move)
    if b0[n0] != 0 or (n0 in t1 and not (s.me_len > s.opp_len)):
        return move
    b0[n0] = 1
    base_space = flood_fill(b0, n0)
    v0_m, v0_o = _voronoi_split(b0, n0, s.opp_head)
    vor0 = v0_m - v0_o

    # boosted landing
    b1 = b0.copy()
    n1 = s.next_pos(n0, move)
    if b1[n1] != 0 or (n1 in t1 and not (s.me_len > s.opp_len)):
        return move
    b1[n1] = 1
    boost_space = flood_fill(b1, n1)
    v1_m, v1_o = _voronoi_split(b1, n1, s.opp_head)
    vor1 = v1_m - v1_o

    # require both space gain and a Voronoi improvement
    if (boost_space - base_space) >= threshold and (vor1 - vor0) >= 4:
        return f"{move}:BOOST"
    return move
