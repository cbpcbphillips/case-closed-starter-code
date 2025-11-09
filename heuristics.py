# heuristics.py
from __future__ import annotations

from collections import deque, defaultdict
from typing import Any, Iterable, Optional, Tuple, List, Dict, Set
import math
import random

from config import cfg

# =========================
# Tunable Weights (config-driven)
# =========================
W_VORONOI      = float(cfg("W_VORONOI",        0.50))
W_SPACE        = float(cfg("W_SPACE",          0.20))
W_BRANCH       = float(cfg("W_BRANCH",         0.05))
W_OPP_REACH    = float(cfg("W_OPP_REACH",      0.20))
W_TRAP         = float(cfg("W_TRAP",           0.15))
W_CENTER       = float(cfg("W_CENTER",         0.02))

W_TUNNEL       = float(cfg("W_TUNNEL",         0.12))   # penalize long 1-wide corridors
W_SPLIT        = float(cfg("W_SPLIT",          0.70))   # reward isolating opponent (different comps)
W_CONTEST      = float(cfg("W_CONTEST",        0.40))   # frontier pressure where we arrive first
W_JUNCTION     = float(cfg("W_JUNCTION",       0.05))   # prefer being closer to nearest junction
W_POCKET       = float(cfg("W_POCKET",         0.35))   # prefer larger safe pockets late
W_CUTPOINT     = float(cfg("W_CUTPOINT",       0.25))   # articulation-point pressure (local)
W_FRONTIER_SAFE= float(cfg("W_FRONTIER_SAFE",  0.20))
W_HEADON_WIN   = float(cfg("W_HEADON_WIN",     0.60))
W_HEADON_LOSS  = float(cfg("W_HEADON_LOSS",    0.90))
W_REP_BREAK    = float(cfg("W_REP_BREAK",      0.0025)) # tiny tie-breaker

MAX_TUNNEL_PROBE   = int(cfg("MAX_TUNNEL_PROBE",   8))
CONTEST_CAP        = int(cfg("CONTEST_CAP",        6))
JUNCTION_CAP       = int(cfg("JUNCTION_CAP",      10))
POCKET_CAP         = int(cfg("POCKET_CAP",        18))
FRONTIER_DEPTH     = int(cfg("FRONTIER_DEPTH",     2))
CUT_LOCAL_RADIUS   = int(cfg("CUT_LOCAL_RADIUS",   3))
DEADEND_CAP        = int(cfg("DEADEND_CAP",        8))

# Quiescence (light tie extension)
QUIESCENCE_ENABLE  = int(cfg("QUIESCENCE_ENABLE",  1))
QUIESCENCE_MARGIN  = float(cfg("QUIESCENCE_MARGIN", 0.06))

# Random tie-break so mirror openings don’t sync forever
TIE_NOISE = float(cfg("TIE_NOISE", 1e-3))

# Boost policy
BOOST_VORO_DELTA   = float(cfg("BOOST_VORO_DELTA", 0.05))  # as fraction of board size
OPENING_TURNS      = int(cfg("OPENING_TURNS", 3))

# Directions (strings to match API)
DIRS = ("UP", "DOWN", "LEFT", "RIGHT")
VEC  = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

# =========================
# Public API
# =========================

def choose_by_heuristic(state: Any, rng: Optional[random.Random] = None) -> str:
    """
    Composite heuristic with 1-ply opponent reply, move ordering, and a small
    transposition cache per decision. Keeps public API intact.
    """
    rng = rng or random.Random(0)
    width, height = _dims(state)
    board_size = width * height

    me_head = _me_head(state)
    opp_head = _opp_head(state)

    legal_me = _legal_dirs_me(state)
    if not legal_me:
        return "UP"  # failsafe

    # Opening nudge to break symmetry: prefer moving slightly away from center
    t = _turn_number(state) or 0
    opening_bias: Dict[str, float] = {}
    if t < OPENING_TURNS:
        cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
        for m in legal_me:
            ny, nx = _torus_add(me_head, VEC[m], width, height)
            dist = _euclid(nx, ny, cx, cy)
            bias = (dist / max(width, height)) * 0.01
            opening_bias[m] = bias

    # Per-turn transposition cache (cheap key, swap to Zobrist later if needed)
    tt: Dict[Tuple, float] = {}

    def _cheap_key(s: Any) -> Tuple:
        # Heads + dims + compact blocked representation length
        # (Using frozenset is fine for now; can replace with Zobrist)
        return (_me_head(s), _opp_head(s), width, height, len(_blocked(s)))

    # Pre-root cache for expensive calcs reuse
    cache_root = _EvalCache(width, height, _blocked(state), me_head, opp_head)

    # Pre-build candidates with move ordering
    ordered: List[Tuple[str, Tuple, Any]] = []
    for m in legal_me:
        s2 = _simulate_me(state, m)
        if s2 is None or not _is_legal_state(s2):
            continue
        key = _ordering_key(s2, cache_root)
        ordered.append((m, key, s2))
    if not ordered:
        return legal_me[0]

    ordered.sort(key=lambda t: t[1])

    best_move, best_val = None, -1e18
    second_val = -1e18

    for m, _, s2 in ordered:
        # Opponent best reply (1-ply)
        opp_best = +1e18
        best_opp_state = None
        for om in _legal_dirs_opp(s2):
            s3 = _simulate_opp(s2, om)
            if s3 is None or not _is_legal_state(s3):
                continue
            # Use a small cache
            k = tt.get(_cheap_key(s3))
            if k is None:
                k = _score_for_opp(s3, cache_root)
                tt[_cheap_key(s3)] = k
            if k < opp_best:
                opp_best = k
                best_opp_state = s3

        my_val = _score_for_me(s2, cache_root, rng, board_size)

        # Compose: my score minus opponent best reply
        val = my_val - opp_best

        # Small split-create lookahead: if our move likely isolates, bonus
        if best_opp_state is not None:
            if _split_bonus(best_opp_state, _swap_heads_cache(cache_root)) > 0.5:
                val += 0.02  # small nudge; keeps stability

        # Opening bias + (conditional) jitter
        if opening_bias:
            val += opening_bias.get(m, 0.0)
        val += rng.uniform(-TIE_NOISE, TIE_NOISE)

        if val > best_val:
            second_val = best_val
            best_val, best_move = val, m
        elif val > second_val:
            second_val = val

    # Lightweight quiescence: if top 2 are near-tied, extend one more ply for the winner
    if (
        QUIESCENCE_ENABLE
        and best_move is not None
        and second_val > -1e17
        and (best_val - second_val) < QUIESCENCE_MARGIN
    ):
        m = best_move
        s2 = _simulate_me(state, m)
        if s2:
            # Evaluate after opponent best reply + our best reply among legal
            opp_best = +1e18
            best_opp = None
            for om in _legal_dirs_opp(s2):
                s3 = _simulate_opp(s2, om)
                if s3 is None or not _is_legal_state(s3):
                    continue
                k = _score_for_opp(s3, cache_root)
                if k < opp_best:
                    opp_best, best_opp = k, s3
            if best_opp:
                # pick our next reply with simple static key
                reply_best = -1e18
                for m2 in _legal_dirs_me(best_opp):
                    s4 = _simulate_me(best_opp, m2)
                    if s4 is None or not _is_legal_state(s4):
                        continue
                    reply_best = max(reply_best, _score_for_me(s4, cache_root, rng, board_size))
                best_val += 0.25 * reply_best  # small extension weight

    if best_move is None:
        return legal_me[0]

    return best_move


def maybe_apply_boost(state: Any, base_move: str, delta_t: Optional[int] = None) -> str:
    """
    Decide whether to append ':BOOST' to base_move.
    Improved: consider phase, voronoi/area deficit, and remaining boosts.
    """
    boosts = _me_boosts(state)
    if boosts <= 0:
        return base_move

    width, height = _dims(state)
    board_size = width * height
    t = _turn_number(state) or 0
    phase = min(1.0, t / max(1, (width * height) / 2))

    s_base = _simulate_me(state, base_move)
    s_boost = _simulate_me_with_boost(state, base_move) or s_base
    if s_base is None or s_boost is None:
        return base_move

    cache_base = _EvalCache(width, height, _blocked(s_base), _me_head(s_base), _opp_head(s_base))
    cache_boost = _EvalCache(width, height, _blocked(s_boost), _me_head(s_boost), _opp_head(s_boost))

    base_voro = _voronoi_delta(s_base, cache_base) / max(1, board_size)
    boost_voro = _voronoi_delta(s_boost, cache_boost) / max(1, board_size)

    area_base = _reachable_area(s_base, "me", cache_base)
    area_opp  = _reachable_area(s_base, "opp", cache_base)
    area_def  = (area_opp - area_base) / max(1, board_size)

    # Spend boosts more aggressively when behind or late
    if (boost_voro - base_voro) > BOOST_VORO_DELTA:
        return f"{base_move}:BOOST"
    if phase > 0.65 and area_def > 0.02:
        return f"{base_move}:BOOST"
    if phase > 0.85:
        return f"{base_move}:BOOST"

    return base_move

# =========================
# Scoring Internals
# =========================

class _EvalCache:
    """
    Small helper to reuse expensive calcs across move branches.
    """
    __slots__ = (
        "width", "height", "blocked_set", "me_head", "opp_head",
        "_degree_cache",
        "_reach_cache_me", "_reach_cache_opp",
        "_voro_cache",
        "_opp_next_cache", "_me_next_cache",
        "_comp_map"
    )
    def __init__(self, width: int, height: int, blocked_set: Set[Tuple[int, int]],
                 me_head: Tuple[int, int], opp_head: Tuple[int, int]):
        self.width = width
        self.height = height
        self.blocked_set = blocked_set
        self.me_head = me_head
        self.opp_head = opp_head
        self._degree_cache: Dict[Tuple[int, int], int] = {}
        self._reach_cache_me: Optional[int] = None
        self._reach_cache_opp: Optional[int] = None
        self._voro_cache: Optional[Tuple[int, int]] = None
        self._opp_next_cache: Optional[Set[Tuple[int, int]]] = None
        self._me_next_cache: Optional[Set[Tuple[int, int]]] = None
        self._comp_map: Optional[List[List[int]]] = None


def _swap_heads_cache(c: _EvalCache) -> _EvalCache:
    return _EvalCache(c.width, c.height, c.blocked_set, c.opp_head, c.me_head)


def _score_for_me(state: Any, cache: _EvalCache, rng: random.Random, board_size: int) -> float:
    width, height = cache.width, cache.height
    me = _me_head(state)
    opp = _opp_head(state)

    voro_me, voro_opp = _voronoi_counts(state, cache)
    area_me = _reachable_area(state, who="me", cache=cache)
    area_opp = _reachable_area(state, who="opp", cache=cache)

    deg = _empty_degree(state, me, cache)
    center_term = _center_bias(me, width, height)

    # Opp “attack squares” next turn (inflate if they have boosts and are close)
    opp_next = _tiles_opp_can_enter_next(state, cache, inflate_boost=True)
    danger = 1 if me in opp_next else 0

    # New signals
    tunnel = _tunnel_len(me, cache, MAX_TUNNEL_PROBE)
    split = _split_bonus(state, cache)                         # 0/1
    contest = _contested_pressure(state, cache, CONTEST_CAP)
    my_j = _nearest_junction_dist(cache.me_head, cache, JUNCTION_CAP)
    op_j = _nearest_junction_dist(cache.opp_head, cache, JUNCTION_CAP)
    pocket_size, pocket_escapes = _pocket_info(state, cache, POCKET_CAP)
    cut_press_me, cut_press_opp = _cutpoint_pressure(state, cache, CUT_LOCAL_RADIUS)
    frontier_safe = _frontier_safety(state, cache, FRONTIER_DEPTH)
    deadend = _deadend_depth(state, cache, DEADEND_CAP)

    # Phase weights (simple schedule)
    t = _turn_number(state) or 0
    phase = min(1.0, t / max(1, (width * height) / 2))
    # Lerp helpers for a few weights
    def lerp(a, b, p): return a + (b - a) * p
    w_pocket  = lerp(0.05, W_POCKET, phase)
    w_front   = lerp(0.05, W_FRONTIER_SAFE, phase)
    w_voro    = lerp(0.20, W_VORONOI, min(phase, 0.7)/0.7)

    score  = 0.0
    score += w_voro   * (voro_me - voro_opp) / max(1, board_size)
    score += W_SPACE  *  area_me / max(1, board_size)
    score += W_BRANCH * (deg - 2)
    score += -W_OPP_REACH * danger
    score += W_TRAP   * max(0, area_me - area_opp) / max(1, board_size)
    score += W_CENTER * center_term
    score += -W_TUNNEL * (tunnel / max(1, MAX_TUNNEL_PROBE))
    score += W_SPLIT  * split
    score += W_CONTEST * contest
    score += W_JUNCTION * (op_j - my_j) / max(1, JUNCTION_CAP)

    # Pockets & frontier & dead-ends (late-game safety)
    score += w_pocket * (pocket_size / max(1, POCKET_CAP))
    score += 0.05 * (pocket_escapes / 4.0)  # up to ~ +0.05
    score += w_front * frontier_safe
    score += -0.08 * (deadend / max(1, DEADEND_CAP))

    # Cutpoint pressure: prefer creating APs near opponent; avoid near self
    score += W_CUTPOINT * (cut_press_opp - cut_press_me)

    # Head-on preference: if tie arrival to same tile next step
    me_next = _tiles_me_can_enter_next(state, cache)
    if (opp in me_next) and (me in opp_next):
        # If we are materially ahead in area, avoid suicide;
        # if behind, slight bias to contest.
        if area_me > area_opp:
            score += -W_HEADON_LOSS * 0.25
        else:
            score += W_HEADON_WIN * 0.15

    # Subtle noise only for tie-breaking; doesn’t steer big choices
    score += rng.uniform(-TIE_NOISE, TIE_NOISE)
    score += W_REP_BREAK * rng.random()
    return score


def _score_for_opp(state: Any, cache: _EvalCache) -> float:
    """
    Opponent’s mirror score (no noise).
    """
    width, height = cache.width, cache.height

    voro_me, voro_opp = _voronoi_counts_mirrored(state, cache)  # flipped roles
    area_me = _reachable_area(state, who="opp", cache=cache)    # opp area from original
    area_opp = _reachable_area(state, who="me", cache=cache)

    me_mir = _opp_head(state)      # “their” head from our POV
    deg = _empty_degree(state, me_mir, cache, mirrored=True)
    center_term = _center_bias(me_mir, width, height)

    my_next = _tiles_me_can_enter_next(state, cache)
    danger = 1 if me_mir in my_next else 0

    # Mirror signals using swapped-head cache
    tmp = _swap_heads_cache(cache)
    tunnel = _tunnel_len(tmp.me_head, tmp, MAX_TUNNEL_PROBE)
    split = _split_bonus(state, tmp)
    contest = _contested_pressure(state, tmp, CONTEST_CAP)
    my_j = _nearest_junction_dist(tmp.me_head, tmp, JUNCTION_CAP)
    op_j = _nearest_junction_dist(tmp.opp_head, tmp, JUNCTION_CAP)
    pocket_size, pocket_escapes = _pocket_info(state, tmp, POCKET_CAP)
    cut_press_me, cut_press_opp = _cutpoint_pressure(state, tmp, CUT_LOCAL_RADIUS)
    frontier_safe = _frontier_safety(state, tmp, FRONTIER_DEPTH)
    deadend = _deadend_depth(state, tmp, DEADEND_CAP)

    # Phase schedule (mirror)
    t = _turn_number(state) or 0
    phase = min(1.0, t / max(1, (width * height) / 2))
    def lerp(a, b, p): return a + (b - a) * p
    w_pocket  = lerp(0.05, W_POCKET, phase)
    w_front   = lerp(0.05, W_FRONTIER_SAFE, phase)
    w_voro    = lerp(0.20, W_VORONOI, min(phase, 0.7)/0.7)

    score  = 0.0
    score += w_voro   * (voro_me - voro_opp) / max(1, width * height)
    score += W_SPACE  *  area_me / max(1, width * height)
    score += W_BRANCH * (deg - 2)
    score += -W_OPP_REACH * danger
    score += W_TRAP   * max(0, area_me - area_opp) / max(1, width * height)
    score += W_CENTER * center_term
    score += -W_TUNNEL * (tunnel / max(1, MAX_TUNNEL_PROBE))
    score += W_SPLIT  * split
    score += W_CONTEST * contest
    score += W_JUNCTION * (op_j - my_j) / max(1, JUNCTION_CAP)

    score += w_pocket * (pocket_size / max(1, POCKET_CAP))
    score += 0.05 * (pocket_escapes / 4.0)
    score += w_front * frontier_safe
    score += -0.08 * (deadend / max(1, DEADEND_CAP))
    score += W_CUTPOINT * (cut_press_opp - cut_press_me)
    return score

# =========================
# Geometry, Flood, Voronoi
# =========================

def _neighbors(y: int, x: int, w: int, h: int) -> Iterable[Tuple[int,int]]:
    yield ((y - 1) % h, x)       # UP
    yield ((y + 1) % h, x)       # DOWN
    yield (y, (x - 1) % w)       # LEFT
    yield (y, (x + 1) % w)       # RIGHT


def _degree(y: int, x: int, cache: _EvalCache) -> int:
    cnt = 0
    for ny, nx in _neighbors(y, x, cache.width, cache.height):
        if (ny, nx) not in cache.blocked_set:
            cnt += 1
    return cnt


def _empty_degree(state: Any, pt: Tuple[int,int], cache: _EvalCache, mirrored: bool=False) -> int:
    if pt in cache._degree_cache:
        return cache._degree_cache[pt]
    cnt = 0
    for ny, nx in _neighbors(pt[0], pt[1], cache.width, cache.height):
        if (ny, nx) not in cache.blocked_set:
            cnt += 1
    cache._degree_cache[pt] = cnt
    return cnt


def _reachable_area(state: Any, who: str, cache: _EvalCache) -> int:
    if who == "me" and cache._reach_cache_me is not None:
        return cache._reach_cache_me
    if who == "opp" and cache._reach_cache_opp is not None:
        return cache._reach_cache_opp

    start = cache.me_head if who == "me" else cache.opp_head
    seen: Set[Tuple[int, int]] = set([start])
    q: deque[Tuple[int, int]] = deque([start])
    blocked = cache.blocked_set
    w, h = cache.width, cache.height
    area = 0
    while q:
        y, x = q.popleft()
        area += 1
        for ny, nx in _neighbors(y, x, w, h):
            if (ny, nx) in blocked or (ny, nx) in seen:
                continue
            seen.add((ny, nx))
            q.append((ny, nx))
    if who == "me":
        cache._reach_cache_me = area
    else:
        cache._reach_cache_opp = area
    return area


def _voronoi_counts(state: Any, cache: _EvalCache) -> Tuple[int,int]:
    if cache._voro_cache is not None:
        return cache._voro_cache
    w, h = cache.width, cache.height
    blocked = cache.blocked_set
    me_q, opp_q = deque(), deque()
    me_dist: Dict[Tuple[int,int], int] = {}
    opp_dist: Dict[Tuple[int,int], int] = {}

    me_q.append(cache.me_head); me_dist[cache.me_head] = 0
    opp_q.append(cache.opp_head); opp_dist[cache.opp_head] = 0

    # BFS from both heads
    while me_q:
        y, x = me_q.popleft()
        d = me_dist[(y, x)]
        for ny, nx in _neighbors(y, x, w, h):
            if (ny, nx) in blocked or (ny, nx) in me_dist: continue
            me_dist[(ny, nx)] = d + 1
            me_q.append((ny, nx))

    while opp_q:
        y, x = opp_q.popleft()
        d = opp_dist[(y, x)]
        for ny, nx in _neighbors(y, x, w, h):
            if (ny, nx) in blocked or (ny, nx) in opp_dist: continue
            opp_dist[(ny, nx)] = d + 1
            opp_q.append((ny, nx))

    me_cnt = opp_cnt = 0
    for y in range(h):
        for x in range(w):
            if (y, x) in blocked: continue
            dm = me_dist.get((y, x))
            do = opp_dist.get((y, x))
            if dm is None and do is None: continue
            if do is None or (dm is not None and dm < do):
                me_cnt += 1
            elif dm is None or do < dm:
                opp_cnt += 1
            # ties: count for neither

    cache._voro_cache = (me_cnt, opp_cnt)
    return cache._voro_cache


def _voronoi_counts_mirrored(state: Any, cache: _EvalCache) -> Tuple[int,int]:
    tmp = _swap_heads_cache(cache)
    return _voronoi_counts(state, tmp)


def _voronoi_delta(state: Any, cache: _EvalCache) -> int:
    me, opp = _voronoi_counts(state, cache)
    return me - opp


def _component_id_map(cache: _EvalCache) -> List[List[int]]:
    """Connected-component labels for empty tiles (blocked_set treated as walls)."""
    w, h = cache.width, cache.height
    comp: List[List[int]] = [[-1] * w for _ in range(h)]
    blocked = cache.blocked_set
    cid = 0
    for y in range(h):
        for x in range(w):
            if (y, x) in blocked or comp[y][x] != -1: continue
            q: deque[Tuple[int,int]] = deque([(y, x)])
            comp[y][x] = cid
            while q:
                yy, xx = q.popleft()
                for ny, nx in _neighbors(yy, xx, w, h):
                    if (ny, nx) in blocked or comp[ny][nx] != -1: continue
                    comp[ny][nx] = cid
                    q.append((ny, nx))
            cid += 1
    return comp


def _split_bonus(state: Any, cache: _EvalCache) -> float:
    if cache._comp_map is None:
        cache._comp_map = _component_id_map(cache)
    comp = cache._comp_map
    myc = comp[cache.me_head[0]][cache.me_head[1]]
    oppc = comp[cache.opp_head[0]][cache.opp_head[1]]
    if myc == -1 or oppc == -1:
        return 0.0
    return 1.0 if myc != oppc else 0.0


def _contested_pressure(state: Any, cache: _EvalCache, cap: int = 6) -> float:
    """Reward tiles reachable by both within <= cap, esp. where we arrive sooner."""
    w, h = cache.width, cache.height
    blocked = cache.blocked_set

    def dists(src: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
        dist: Dict[Tuple[int, int], int] = {src: 0}
        q: deque[Tuple[int, int]] = deque([src])
        while q:
            y, x = q.popleft()
            d = dist[(y, x)]
            for ny, nx in _neighbors(y, x, w, h):
                if (ny, nx) in blocked or (ny, nx) in dist: continue
                nd = d + 1
                if nd > cap: continue
                dist[(ny, nx)] = nd
                q.append((ny, nx))
        return dist

    dm = dists(cache.me_head)
    do = dists(cache.opp_head)
    wins = 0
    ties = 0
    for p, d in dm.items():
        if p in do:
            diff = do[p] - d
            if diff >= 1:
                wins += 1
            elif diff == 0:
                ties += 1
    board_size = max(1, w * h)
    return (wins + 0.1 * ties) / board_size


def _tunnel_len(start: Tuple[int,int], cache: _EvalCache, max_probe: int = 8) -> int:
    y, x = start
    w, h = cache.width, cache.height
    blocked = cache.blocked_set
    seen: Set[Tuple[int,int]] = set()
    cur = (y, x)
    steps = 0
    while steps < max_probe:
        nbrs = [(ny, nx) for (ny, nx) in _neighbors(cur[0], cur[1], w, h) if (ny, nx) not in blocked]
        if len(nbrs) != 1:
            break
        nxt = nbrs[0]
        if nxt in seen:
            break
        seen.add(nxt)
        cur = nxt
        steps += 1
    return steps


def _nearest_junction_dist(start: Tuple[int,int], cache: _EvalCache, cap: int = 10) -> int:
    w, h = cache.width, cache.height
    blocked = cache.blocked_set
    q = deque([(start, 0)])
    seen = {start}
    while q:
        (y, x), d = q.popleft()
        if d > cap:
            return cap + 1
        deg = 0
        for ny, nx in _neighbors(y, x, w, h):
            if (ny, nx) not in blocked:
                deg += 1
        if deg >= 3:
            return d
        for ny, nx in _neighbors(y, x, w, h):
            if (ny, nx) in blocked or (ny, nx) in seen:
                continue
            seen.add((ny, nx))
            q.append(((ny, nx), d + 1))
    return cap + 1

# ---------- Added signals: pocket, cutpoints, frontier, dead-ends ----------

def _pocket_info(state: Any, cache: _EvalCache, cap: int = 18) -> Tuple[int, int]:
    """
    Approx pocket size near our head (limited BFS) and count of "escapes"
    (nodes within BFS that have degree >= 3). Cheap late-game safety proxy.
    """
    start = cache.me_head
    w, h = cache.width, cache.height
    blocked = cache.blocked_set
    seen: Set[Tuple[int,int]] = {start}
    q: deque[Tuple[Tuple[int,int], int]] = deque([(start, 0)])
    size = 0
    escapes = 0
    while q:
        (y, x), d = q.popleft()
        size += 1
        deg = 0
        nbrs = []
        for ny, nx in _neighbors(y, x, w, h):
            if (ny, nx) in blocked:
                continue
            deg += 1
            nbrs.append((ny, nx))
        if deg >= 3:
            escapes += 1
        if d >= cap:
            continue
        for p in nbrs:
            if p in seen:
                continue
            seen.add(p)
            q.append((p, d+1))
    return (min(size, cap), min(escapes, 4))


def _cutpoint_pressure(state: Any, cache: _EvalCache, radius: int = 3) -> Tuple[float, float]:
    """
    Compute local articulation-point pressure around each head within a radius.
    Returns (pressure_near_me, pressure_near_opp), each normalized ~[0,1].
    """
    def local_nodes(center: Tuple[int,int]) -> Set[Tuple[int,int]]:
        cy, cx = center
        nodes: Set[Tuple[int,int]] = set()
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                y = (cy + dy) % cache.height
                x = (cx + dx) % cache.width
                if (y, x) not in cache.blocked_set:
                    nodes.add((y, x))
        return nodes

    def articulation_fraction(nodes: Set[Tuple[int,int]]) -> float:
        # Build adjacency restricted to nodes
        idx: Dict[Tuple[int,int], int] = {p:i for i,p in enumerate(nodes)}
        g: List[List[int]] = [[] for _ in range(len(nodes))]
        nodes_list = list(nodes)
        for i, (y, x) in enumerate(nodes_list):
            for ny, nx in _neighbors(y, x, cache.width, cache.height):
                if (ny, nx) in nodes:
                    g[i].append(idx[(ny, nx)])

        # Tarjan AP
        n = len(nodes_list)
        if n <= 2: return 0.0
        timer = 0
        tin = [-1]*n
        low = [-1]*n
        ap = [False]*n

        def dfs(v: int, p: int):
            nonlocal timer
            timer += 1
            tin[v] = low[v] = timer
            children = 0
            for to in g[v]:
                if to == p:
                    continue
                if tin[to] != -1:
                    low[v] = min(low[v], tin[to])
                else:
                    dfs(to, v)
                    low[v] = min(low[v], low[to])
                    if low[to] >= tin[v] and p != -1:
                        ap[v] = True
                    children += 1
            if p == -1 and children > 1:
                ap[v] = True

        for v in range(n):
            if tin[v] == -1:
                dfs(v, -1)

        frac = sum(1 for x in ap if x) / float(n)
        return min(1.0, frac)

    me_nodes  = local_nodes(cache.me_head)
    opp_nodes = local_nodes(cache.opp_head)
    return (articulation_fraction(me_nodes), articulation_fraction(opp_nodes))


def _frontier_safety(state: Any, cache: _EvalCache, depth: int = 2) -> float:
    """
    Score frontier tiles (adjacent to walls) we can reach in <= depth before opp,
    normalized by board size. Encourages safe expansion along walls.
    """
    w, h = cache.width, cache.height
    blocked = cache.blocked_set

    def dist_cap(src: Tuple[int,int]) -> Dict[Tuple[int,int], int]:
        d: Dict[Tuple[int,int], int] = {src: 0}
        q: deque[Tuple[int,int]] = deque([src])
        while q:
            y, x = q.popleft()
            dd = d[(y, x)]
            for ny, nx in _neighbors(y, x, w, h):
                if (ny, nx) in blocked or (ny, nx) in d: continue
                nd = dd + 1
                if nd > depth: continue
                d[(ny, nx)] = nd
                q.append((ny, nx))
        return d

    dm = dist_cap(cache.me_head)
    do = dist_cap(cache.opp_head)
    good = 0
    for (y, x), d in dm.items():
        # frontier if at least one neighbor is blocked
        if any(((ny, nx) in blocked) for (ny, nx) in _neighbors(y, x, w, h)):
            if (y, x) not in do or d < do[(y, x)]:
                good += 1
    return good / max(1, w * h)


def _deadend_depth(state: Any, cache: _EvalCache, cap: int = 8) -> int:
    """
    Depth (in steps) to the nearest forced dead-end (walking only degree<=2 nodes).
    Higher is worse; normalized in caller.
    """
    w, h = cache.width, cache.height
    blocked = cache.blocked_set
    start = cache.me_head

    seen = {start}
    q = deque([(start, 0)])
    while q:
        p, d = q.popleft()
        if d >= cap:
            return cap
        # forced chain => all neighbors count <= 2; dead-end when count <= 1
        deg = 0
        nbrs = []
        for ny, nx in _neighbors(p[0], p[1], w, h):
            if (ny, nx) in blocked:
                continue
            deg += 1
            nbrs.append((ny, nx))
        if deg <= 1:
            return d
        if deg <= 2:
            for qn in nbrs:
                if qn not in seen:
                    seen.add(qn)
                    q.append((qn, d+1))
    return cap

# ---------- Helpers used by scoring / ordering / threats ----------

def _ordering_key(s: Any, croot: _EvalCache) -> Tuple:
    cache = _EvalCache(croot.width, croot.height, _blocked(s), _me_head(s), _opp_head(s))
    deg = _empty_degree(s, cache.me_head, cache)
    area = _reachable_area(s, "me", cache)
    tunnel = _tunnel_len(cache.me_head, cache, MAX_TUNNEL_PROBE)
    # Lower is better (Python sorts ascending)
    return (-area, -deg, tunnel)


def _tiles_opp_can_enter_next(state: Any, cache: _EvalCache, inflate_boost: bool=False) -> Set[Tuple[int,int]]:
    if cache._opp_next_cache is not None and not inflate_boost:
        return cache._opp_next_cache
    res = set()
    w, h = cache.width, cache.height
    for m in _legal_dirs_opp(state):
        s2 = _simulate_opp(state, m)
        if s2 is None or not _is_legal_state(s2):
            continue
        res.add(_opp_head(s2))
        if inflate_boost and _opp_boosts(state) > 0:
            # include one more step envelope
            oh = _opp_head(s2)
            for ny, nx in _neighbors(oh[0], oh[1], w, h):
                if (ny, nx) not in cache.blocked_set:
                    res.add((ny, nx))
    if not inflate_boost:
        cache._opp_next_cache = res
    return res


def _tiles_me_can_enter_next(state: Any, cache: _EvalCache) -> Set[Tuple[int,int]]:
    if cache._me_next_cache is not None:
        return cache._me_next_cache
    res = set()
    for m in _legal_dirs_me(state):
        s2 = _simulate_me(state, m)
        if s2 is None or not _is_legal_state(s2):
            continue
        res.add(_me_head(s2))
    cache._me_next_cache = res
    return res

# =========================
# Utility & Adapters
# =========================

def _dims(state: Any) -> Tuple[int,int]:
    W = getattr(state, "W", None)
    H = getattr(state, "H", None)
    if W is not None and H is not None:
        return (int(W), int(H))
    w = getattr(state, "width", None)
    h = getattr(state, "height", None)
    if w is None and isinstance(state, dict):
        w = state.get("width"); h = state.get("height")
    if w is None or h is None:
        return (20, 18)
    return (int(w), int(h))


def _me_head(state: Any) -> Tuple[int,int]:
    for k in ("me_head", "my_head", "head_me", "meHead"):
        v = getattr(state, k, None)
        if v is not None: return tuple(v)
    if isinstance(state, dict):
        me = state.get("me") or state.get("my")
        if isinstance(me, dict) and "head" in me:
            return tuple(me["head"])
        if "me_head" in state:
            return tuple(state["me_head"])
    v = getattr(state, "head", None)
    if v is not None: return tuple(v)
    return (0, 0)


def _opp_head(state: Any) -> Tuple[int,int]:
    for k in ("opp_head", "other_head", "opponent_head", "oppHead"):
        v = getattr(state, k, None)
        if v is not None: return tuple(v)
    if isinstance(state, dict):
        opp = state.get("opp") or state.get("opponent")
        if isinstance(opp, dict) and "head" in opp:
            return tuple(opp["head"])
        if "opp_head" in state:
            return tuple(state["opp_head"])
    return (0, 0)


def _blocked(state: Any) -> Set[Tuple[int,int]]:
    b = getattr(state, "board", None)
    if b is not None:
        try:
            import numpy as np  # noqa
            arr = np.asarray(b)
            ys, xs = (arr != 0).nonzero()
            return {(int(y), int(x)) for y, x in zip(ys, xs)}
        except Exception:
            pass
    for k in ("blocked", "walls", "occupied", "trails"):
        v = getattr(state, k, None)
        if isinstance(v, set):
            return set(v)
        if isinstance(v, list) and v and isinstance(v[0], (tuple, list)):
            return {tuple(p) for p in v}
    if isinstance(state, dict):
        for k in ("blocked", "walls", "occupied", "trails"):
            v = state.get(k)
            if isinstance(v, set):
                return set(v)
            if isinstance(v, list) and v and isinstance(v[0], (tuple, list)):
                return {tuple(p) for p in v}
        to_union: List[Tuple[int,int]] = []
        me = state.get("me") or state.get("my")
        if isinstance(me, dict):
            for kk in ("trail", "body", "occupied"):
                if kk in me: to_union.extend(me[kk])
        opp = state.get("opp") or state.get("opponent")
        if isinstance(opp, dict):
            for kk in ("trail", "body", "occupied"):
                if kk in opp: to_union.extend(opp[kk])
        if to_union:
            return {tuple(p) for p in to_union}
    return {_me_head(state), _opp_head(state)}


def _legal_dirs_me(state: Any) -> List[str]:
    for name in ("legal_dirs_me", "legal_me", "legalMovesMe", "legal_my"):
        fn = getattr(state, name, None)
        if callable(fn): return list(fn())
    if isinstance(state, dict):
        for k in ("legal_me", "legal_dirs_me", "legalMovesMe", "my_legal"):
            v = state.get(k)
            if isinstance(v, (list, tuple)):
                return [str(x) for x in v]
    return list(DIRS)


def _legal_dirs_opp(state: Any) -> List[str]:
    for name in ("legal_dirs_opp", "legal_opp", "legalMovesOpp", "legal_other"):
        fn = getattr(state, name, None)
        if callable(fn): return list(fn())
    if isinstance(state, dict):
        for k in ("legal_opp", "legal_dirs_opp", "legalMovesOpp", "opp_legal"):
            v = state.get(k)
            if isinstance(v, (list, tuple)):
                return [str(x) for x in v]
    return list(DIRS)


def _simulate_me(state: Any, move: str) -> Optional[Any]:
    for name in ("after_me", "after", "step_me", "step"):
        fn = getattr(state, name, None)
        if callable(fn):
            try:
                return fn(move)
            except TypeError:
                try:
                    return fn(move, "me")
                except Exception:
                    pass
    return None


def _simulate_opp(state: Any, move: str) -> Optional[Any]:
    for name in ("after_opp", "after_other", "step_opp", "step"):
        fn = getattr(state, name, None)
        if callable(fn):
            try:
                return fn(move)
            except TypeError:
                try:
                    return fn(move, "opp")
                except Exception:
                    pass
    return None


def _simulate_me_with_boost(state: Any, move: str) -> Optional[Any]:
    for name in ("after_me_boost", "step_me_boost", "after_boost"):
        fn = getattr(state, name, None)
        if callable(fn):
            try:
                return fn(move)
            except Exception:
                pass
    return None


def _is_legal_state(state: Any) -> bool:
    width, height = _dims(state)
    me = _me_head(state)
    opp = _opp_head(state)
    blocked = _blocked(state)
    return (me not in blocked) and (opp not in blocked) and (0 <= me[0] < height) and (0 <= me[1] < width)


def _me_boosts(state: Any) -> int:
    for k in ("me_boosts", "boosts_me", "my_boosts"):
        v = getattr(state, k, None)
        if isinstance(v, int):
            return v
    if isinstance(state, dict):
        me = state.get("me") or state.get("my")
        if isinstance(me, dict) and isinstance(me.get("boosts"), int):
            return int(me["boosts"])
        if isinstance(state.get("me_boosts"), int):
            return int(state["me_boosts"])
    return 0


def _opp_boosts(state: Any) -> int:
    for k in ("opp_boosts", "boosts_opp", "other_boosts"):
        v = getattr(state, k, None)
        if isinstance(v, int):
            return v
    if isinstance(state, dict):
        opp = state.get("opp") or state.get("opponent")
        if isinstance(opp, dict) and isinstance(opp.get("boosts"), int):
            return int(opp["boosts"])
        if isinstance(state.get("opp_boosts"), int):
            return int(state["opp_boosts"])
    return 0


def _turn_number(state: Any) -> Optional[int]:
    for k in ("turn", "t", "ply", "move_number", "turn_count"):
        v = getattr(state, k, None)
        if isinstance(v, int):
            return v
    if isinstance(state, dict):
        for k in ("turn", "t", "ply", "move_number", "turn_count"):
            v = state.get(k)
            if isinstance(v, int):
                return v
    return None


def _center_bias(pt: Tuple[int,int], w: int, h: int) -> float:
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    dist = _euclid(pt[1], pt[0], cx, cy)
    return (dist - 0.5 * (w + h) / 2.0) / max(1.0, max(w, h))


def _torus_add(pt: Tuple[int,int], delta: Tuple[int,int], w: int, h: int) -> Tuple[int,int]:
    dy, dx = delta
    return ((pt[0] + dy) % h, (pt[1] + dx) % w)


def _euclid(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)
