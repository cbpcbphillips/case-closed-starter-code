# heuristics.py — full module with robust weights loader + train-mode randomness

from __future__ import annotations

import os
import json
import time
import random
from pathlib import Path
from collections import deque
from typing import List, Tuple, Set

import numpy as np

from state import State, DIRS

# =========================
# Default Weights / Knobs
# =========================
W_SPACE        = 1.30    # space control
W_SPACE_REPLY  = 0.35    # still-ours space after opp reply
W_BRANCH       = 0.22
W_TUNNEL       = -0.20
W_STRAIGHT     = 0.05
W_CHOKE        = -0.45
W_HEADON_WIN   = 0.30
W_VORONOI      = 0.40
W_VORONOI_RPLY = 0.20
W_ARTICULATE   = -0.50   # articulation-ish penalty
W_STEP_IN_LOSS = -1e9
W_THREAT2      = -0.25

EPS_NEAR_TIE   = 3.0
BOOST_DELTA_T  = 8

# Opposite-direction map (avoid 180° reversals)
OPP = {"UP":"DOWN","DOWN":"UP","LEFT":"RIGHT","RIGHT":"LEFT"}

# Small search/time knobs
BEAM_K_ME       = 3
BEAM_K_OPP      = 3
MOVE_BUDGET_SEC = 0.030  # ~30ms per move

# ===== Train-mode flags / randomness =====
TRAIN_MODE        = os.getenv("HEURISTICS_TRAIN", "0") == "1"
TEMP_SAMPLE       = 0.60   # softmax temperature for near-ties (lower = greedier)
RAND_OPENING_P    = 0.25   # chance to inject a non-straight opening choice at turns 0–2
STRAIGHT_STICK_P  = 0.65   # when near-tie, chance to keep going straight
NOISE_SCALE       = 0.12 if TRAIN_MODE else 0.05  # more noise only in train mode

# -- allowed keys for weights.json --
_ALLOWED_KEYS = {
    "W_SPACE","W_SPACE_REPLY","W_BRANCH","W_TUNNEL","W_STRAIGHT","W_CHOKE",
    "W_HEADON_WIN","W_VORONOI","W_VORONOI_RPLY","W_ARTICULATE","W_STEP_IN_LOSS",
    "W_THREAT2","EPS_NEAR_TIE","BOOST_DELTA_T","NOISE_SCALE",
    "BEAM_K_ME","BEAM_K_OPP","MOVE_BUDGET_SEC",
}

def _apply_weights(cfg: dict) -> None:
    g = globals()
    for k, v in cfg.items():
        if k in _ALLOWED_KEYS:
            g[k] = v

def _weights_path() -> Path:
    p = os.environ.get("WEIGHTS_PATH")
    if p:
        return Path(p).expanduser().resolve()
    return Path(__file__).resolve().parent / "weights.json"

def reload_weights(verbose: bool = False) -> bool:
    path = _weights_path()
    if not path.exists():
        if verbose:
            print(f"[heuristics] weights file not found: {path}")
        return False
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        _apply_weights(cfg)
        if verbose:
            print(f"[heuristics] using weights from: {path}")
        return True
    except Exception as e:
        if verbose:
            print(f"[heuristics] failed to load weights from {path}: {e}")
        return False

# load once at import time
reload_weights(verbose=False)

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

def chebyshev(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

def _softmax_sample(moves: List[str], scores: List[float], temp: float) -> str:
    mx = max(scores)
    exps = [np.exp((s - mx) / max(1e-6, temp)) for s in scores]
    Z = sum(exps) or 1.0
    probs = [e / Z for e in exps]
    return np.random.choice(moves, p=probs)

# =========================
# Safety / legality
# =========================
def safe_moves(s: State) -> List[str]:
    """
    Legal moves that avoid:
    - walls/trails,
    - contested next cells (opp can step there this tick) unless strictly longer,
    - head-swap unless strictly longer,
    - (if alternatives exist) opponent's 2-tick threat,
    - (when close) reducing distance to opponent,
    - (if alternatives exist) immediate 180° reversals.
    """
    t1 = opp_threat_cells_1(s)
    t2 = opp_threat_cells_2(s)
    swap_possible = head_swap_threat(s)

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

    # Forbid immediate 180° reversals unless forced
    if s.me_last_dir and s.me_last_dir in OPP:
        opp_dir = OPP[s.me_last_dir]
        base_wo = [d for d in base if d != opp_dir]
        if base_wo:
            base = base_wo

    # If any moves avoid opp 2-tick threat, prefer only those
    avoid_t2 = [d for d in base if s.next_pos(s.me_head, d) not in t2]
    if avoid_t2:
        base = avoid_t2

    # If we're close (<=3 by Chebyshev), avoid moves that REDUCE distance
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
# Articulation-ish test (local trap detector)
# =========================
def _articulationish_penalty(board: np.ndarray, at: Tuple[int,int]) -> float:
    """
    If occupying 'at' likely splits our future space into small pockets,
    penalize. Approximate by closing forward-ish neighbors and comparing fills.
    """
    if board[at] != 0:
        return 1.0
    H, W = board.shape
    S0 = flood_fill(board, at)
    r, c = at
    deltas = [(-1,0),(1,0),(0,-1),(0,1)]
    bad = 0
    for dr, dc in deltas:
        rr = (r+dr) % H; cc = (c+dc) % W
        if board[rr, cc] != 0:
            continue
        tmp = board.copy()
        tmp[rr, cc] = 1  # close chokepoint
        S1 = flood_fill(tmp, at)
        if S1 + 4 < S0:
            bad += 1
            if bad >= 2:
                break
    return float(bad)  # 0,1,2

# =========================
# Scoring & selection
# =========================
def _score_after_reply(s: State, my_move: str) -> Tuple[float, float]:
    """Return (space_after_me, space_after_best_reply)."""
    b_me = s.board.copy()
    me_next = s.next_pos(s.me_head, my_move)
    b_me[me_next] = 1
    my_space = flood_fill(b_me, me_next)

    opp_best_space = -1
    for od in candidate_opp_dirs(s):
        opp_next = s.next_pos(s.opp_head, od)
        if b_me[opp_next] != 0:
            continue
        b2 = b_me.copy()
        b2[opp_next] = 1
        sp = flood_fill(b2, opp_next)
        if sp > opp_best_space:
            opp_best_space = sp
    if opp_best_space < 0:
        opp_best_space = 0
    return float(my_space), float(opp_best_space)

def score_move(s: State, move: str) -> float:
    # block unsafe options
    if move not in safe_moves(s):
        return W_STEP_IN_LOSS

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

    # reply-aware spaces
    me_space0, opp_space0 = _score_after_reply(s, move)

    # local shape features
    br  = branching(board1, me_next)
    tun = tunnel_depth(board1, me_next)

    # choke indicator
    opp_br = branching(board2, opp_next)
    choke_flag = 1 if (br <= 1 and opp_br >= 2) else 0

    # head-on indicator
    headon_flag = 0
    if me_next == opp_next:
        if s.me_len > s.opp_len:
            headon_flag = 1
        elif s.me_len < s.opp_len:
            headon_flag = -1

    # Voronoi splits
    v_mine, v_theirs = _voronoi_split(board1, me_next, s.opp_head)
    voronoi_gain = (v_mine - v_theirs)

    # reply-aware Voronoi (opponent steps on board1)
    v2_mine, v2_theirs = _voronoi_split(board2, me_next, opp_next)
    vor_after_reply = (v2_mine - v2_theirs)

    # articulation-ish penalty
    art_hits = _articulationish_penalty(board1, me_next)

    # soft threat + tiny deterministic noise
    t2 = opp_threat_cells_2(s)
    threat_pen = W_THREAT2 if (me_next in t2) else 0.0
    noise = _hash_noise(s, move)

    score = (
        W_SPACE        * (my_space - opp_space)
        + W_SPACE_REPLY* (me_space0 - opp_space0)
        + W_BRANCH     * (br - 2)
        + W_TUNNEL     * (tun)
        + W_STRAIGHT   * (1 if (s.me_last_dir and move == s.me_last_dir) else 0)
        + W_CHOKE      * (choke_flag)
        + W_HEADON_WIN * (headon_flag)
        + W_VORONOI    * (voronoi_gain)
        + W_VORONOI_RPLY * (vor_after_reply)
        + W_ARTICULATE * (art_hits)
        + threat_pen
        + noise
    )
    return score

def _rank_moves(s: State, moves: List[str]) -> List[Tuple[str, float]]:
    scored = [(m, score_move(s, m)) for m in moves]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

def _beam_lookahead(s: State, budget_sec: float = MOVE_BUDGET_SEC) -> str:
    start = time.perf_counter()
    legal = safe_moves(s)
    if not legal:
        return "UP"

    # If time is extremely tight, just pick best 1-ply
    if budget_sec <= 0.005:
        return max(legal, key=lambda m: score_move(s, m))

    first = _rank_moves(s, legal)[:BEAM_K_ME]
    best_move, best_val = first[0][0], first[0][1]

    for m, _sc in first:
        if time.perf_counter() - start > budget_sec:
            break
        # write our step
        b1 = s.board.copy()
        me1 = s.next_pos(s.me_head, m)
        b1[me1] = 1

        # enumerate opp replies among plausible candidates on b1
        opp_cands = candidate_opp_dirs(s)
        opp_ranked = []
        for od in opp_cands:
            onxt = s.next_pos(s.opp_head, od)
            if b1[onxt] != 0:
                continue
            b2 = b1.copy(); b2[onxt] = 1
            opp_ranked.append((od, flood_fill(b2, onxt)))
        opp_ranked.sort(key=lambda x: x[1], reverse=True)
        opp_ranked = opp_ranked[:BEAM_K_OPP] if opp_ranked else []

        worst_case = float("inf")
        # our value if opp picks the reply worst for us
        for od,_ in opp_ranked or [("UP",0)]:
            v = score_move(s, m)  # reuse scorer (already reply-aware)
            if v < worst_case:
                worst_case = v

        if worst_case > best_val:
            best_val = worst_case
            best_move = m

    # near-tie straight bias
    near = [mv for mv,val in first if abs(val - best_val) <= EPS_NEAR_TIE]
    if len(near) > 1 and s.me_last_dir and s.me_last_dir in near:
        return s.me_last_dir
    return best_move

def choose_by_heuristic(s: State) -> str:
    legal = safe_moves(s)
    if not legal:
        return "UP"

    # Small randomized opening to avoid perfect mirrors (train mode only)
    if TRAIN_MODE and s.turn <= 2 and random.random() < RAND_OPENING_P:
        if s.me_last_dir:
            opp = OPP.get(s.me_last_dir)
            cand = [d for d in legal if d != opp] or legal
            return random.choice(cand)
        return random.choice(legal)

    # Score all legal
    scored = [(m, score_move(s, m)) for m in legal]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_val = scored[0][1]

    # Collect near-ties
    near_moves  = [m for (m, sc) in scored if (best_val - sc) <= EPS_NEAR_TIE]
    near_scores = [sc for (m, sc) in scored if (best_val - sc) <= EPS_NEAR_TIE]

    if len(near_moves) > 1:
        # sometimes keep straight if it's among near ties
        if s.me_last_dir in near_moves and random.random() < STRAIGHT_STICK_P:
            return s.me_last_dir
        # train mode: softmax sample for exploration
        if TRAIN_MODE:
            return _softmax_sample(near_moves, near_scores, TEMP_SAMPLE)
        # match mode: deterministic tiebreak by branching
        def br_for(m):
            b = s.board.copy()
            nxt = s.next_pos(s.me_head, m); b[nxt] = 1
            return branching(b, nxt)
        return max(near_moves, key=br_for)

    return scored[0][0]

# =========================
# Boost policy
# =========================
def maybe_apply_boost(s: State, move: str, threshold: int = BOOST_DELTA_T) -> str:
    """
    Minimal, safe boost policy:
    - Only consider boosting if baseline landing is safe and uncontested.
    - Boost landing must also be safe and uncontested.
    - Require space gain AND favor boosts that improve Voronoi split,
      or when we likely created disjoint regions and need rapid expansion.
    """
    if s.me_boosts <= 0:
        return move

    if TRAIN_MODE:
        # jitter ±2 cells during training for exploration on borderline boosts
        threshold = int(threshold + np.random.randint(-2, 3))

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

    # detect disjoint components after our step (approx)
    opp_reaches_us = False
    tmp = b0.copy()
    if tmp[s.opp_head] == 0:
        q = deque([s.opp_head]); seen = {s.opp_head}
        H,W = tmp.shape
        while q:
            r,c = q.popleft()
            if (r,c) == n0: opp_reaches_us = True; break
            for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                rr=(r+dr)%H; cc=(c+dc)%W
                if tmp[rr,cc]==0 and (rr,cc) not in seen:
                    seen.add((rr,cc)); q.append((rr,cc))

    # boosted landing
    b1 = b0.copy()
    n1 = s.next_pos(n0, move)
    if b1[n1] != 0 or (n1 in t1 and not (s.me_len > s.opp_len)):
        return move
    b1[n1] = 1
    boost_space = flood_fill(b1, n1)
    v1_m, v1_o = _voronoi_split(b1, n1, s.opp_head)
    vor1 = v1_m - v1_o

    space_gain = (boost_space - base_space)
    vor_gain   = (vor1 - vor0)

    # Rule A: power cut + territory gain
    if space_gain >= threshold and vor_gain >= 4:
        return f"{move}:BOOST"

    # Rule B: if regions are disjoint and we’re smaller, expand fast
    if not opp_reaches_us and boost_space > base_space + 4:
        return f"{move}:BOOST"

    return move
