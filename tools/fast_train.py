# tools/fast_train.py
# Fast in-process self-play + CPU-only param search (no HTTP/judge sleeps)
# Candidate fights a frozen baseline so trials aren't deterministic mirrors.

import os, sys, json, random, argparse, multiprocessing as mp
from pathlib import Path

# ---- Make repo root importable ----
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Sanity check
for req in ("case_closed_game.py", "state.py", "heuristics.py"):
    if not (ROOT / req).exists():
        raise FileNotFoundError(f"Expected {req} in {ROOT}, but not found.")

import numpy as np
from case_closed_game import Game, Direction, GameResult
from state import parse_state
import heuristics as H  # we'll override weight globals per-role via context

# --------- tiny opening book to avoid early dumb crashes ----------
OPENINGS = {
    ((1, 2), (17, 15)): ["RIGHT","RIGHT","DOWN","DOWN"],
    ((17, 15), (1, 2)): ["LEFT","LEFT","UP","UP"],
}

def opening_move_if_any(state_dict, role: str):
    turn = int(state_dict.get("turn_count", 0))
    if turn >= 4:
        return None
    my_trail = state_dict.get("agent1_trail" if role=="agent1" else "agent2_trail", [])
    opp_trail = state_dict.get("agent2_trail" if role=="agent1" else "agent1_trail", [])
    if not my_trail or not opp_trail:
        return None
    my_start = tuple(my_trail[0]); opp_start = tuple(opp_trail[0])
    seq = OPENINGS.get((my_start, opp_start))
    if seq and turn < len(seq):
        return seq[turn]
    return None

# --- prevent 180° reversals at engine level (mirrors HTTP judge behavior) ---
def fix_opposite(req_dir: Direction, cur_dir: Direction) -> Direction:
    rx, ry = req_dir.value
    cx, cy = cur_dir.value
    return cur_dir if (rx, ry) == (-cx, -cy) else req_dir

# ---------- Weight plumbing ----------
# Defaults (and expose important knobs so candidates can perturb them)
DEFAULT_WEIGHTS = {
 "W_SPACE":1.30,"W_SPACE_REPLY":0.35,"W_BRANCH":0.22,"W_TUNNEL":-0.20,"W_STRAIGHT":0.05,
 "W_CHOKE":-0.45,"W_HEADON_WIN":0.30,"W_VORONOI":0.40,"W_VORONOI_RPLY":0.20,
 "W_ARTICULATE":-0.50,"W_THREAT2":-0.25,
 "EPS_NEAR_TIE":3.0,"BOOST_DELTA_T":8,
 "BEAM_K_ME":3, "BEAM_K_OPP":3, "MOVE_BUDGET_SEC":0.03
}

# Snapshot the heuristics module's current values as the initial BASELINE
BASELINE_WEIGHTS = dict(DEFAULT_WEIGHTS)
for k in DEFAULT_WEIGHTS:
    if hasattr(H, k):
        BASELINE_WEIGHTS[k] = getattr(H, k)

TUNE_KEYS = [
    "W_SPACE","W_VORONOI","W_BRANCH","W_TUNNEL","W_CHOKE",
    "W_VORONOI_RPLY","W_SPACE_REPLY","W_HEADON_WIN","W_ARTICULATE"
]

class WeightsCtx:
    """Context manager to temporarily set H.<weight>=value then restore."""
    def __init__(self, weights: dict):
        self.weights = weights
        self.prev = {}

    def __enter__(self):
        for k, v in self.weights.items():
            if hasattr(H, k):
                self.prev[k] = getattr(H, k)
                setattr(H, k, v)
        return self

    def __exit__(self, exc_type, exc, tb):
        for k, v in self.prev.items():
            setattr(H, k, v)

def decide_with_weights(game: Game, role: str, weights: dict):
    """Build a judge-like payload, set weights temporarily, and decide."""
    board = game.board.grid
    a1_trail = list(game.agent1.get_trail_positions())
    a2_trail = list(game.agent2.get_trail_positions())
    payload = dict(
        board=board,
        agent1_trail=a1_trail,
        agent2_trail=a2_trail,
        agent1_length=game.agent1.length,
        agent2_length=game.agent2.length,
        agent1_alive=game.agent1.alive,
        agent2_alive=game.agent2.alive,
        agent1_boosts=game.agent1.boosts_remaining,
        agent2_boosts=game.agent2.boosts_remaining,
        turn_count=game.turns,
    )
    om = opening_move_if_any(payload, role)
    if om:
        return om

    s = parse_state(payload, role=role)
    with WeightsCtx(weights):
        base = H.choose_by_heuristic(s)
        final = H.maybe_apply_boost(s, base)
    return final

def parse_move_str(move_str: str):
    m = move_str.upper().split(":")
    d = m[0]
    boost = (len(m) == 2 and m[1] == "BOOST")
    dmap = {"UP":Direction.UP, "DOWN":Direction.DOWN, "LEFT":Direction.LEFT, "RIGHT":Direction.RIGHT}
    return dmap.get(d, Direction.RIGHT), boost

# ----------- single fast self-play game: candidate vs baseline -----------
def play_one_game(candidate_weights: dict, baseline_weights: dict, swap_sides: bool, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    g = Game()

    for _ in range(600):  # hard cap
        if not swap_sides:
            m1 = decide_with_weights(g, "agent1", candidate_weights)
            m2 = decide_with_weights(g, "agent2", baseline_weights)
        else:
            # swap: candidate controls agent2 this time
            m1 = decide_with_weights(g, "agent1", baseline_weights)
            m2 = decide_with_weights(g, "agent2", candidate_weights)

        d1, b1 = parse_move_str(m1)
        d2, b2 = parse_move_str(m2)

        d1 = fix_opposite(d1, g.agent1.direction)
        d2 = fix_opposite(d2, g.agent2.direction)

        result = g.step(d1, d2, b1, b2)
        if result is not None:
            return result
    return GameResult.DRAW

# ----------- evaluation -----------
def eval_candidate(candidate: dict, games: int, seed: int = 0, baseline: dict = None):
    baseline = baseline or BASELINE_WEIGHTS
    wins_cand = wins_base = draws = 0

    # half games no-swap (cand as agent1), half swapped
    half = games // 2
    rest = games - half

    for i in range(half):
        res = play_one_game(candidate, baseline, swap_sides=False, seed=seed + i)
        if res == GameResult.AGENT1_WIN:
            wins_cand += 1
        elif res == GameResult.AGENT2_WIN:
            wins_base += 1
        else:
            draws += 1

    for i in range(rest):
        res = play_one_game(candidate, baseline, swap_sides=True, seed=seed + 10000 + i)
        # when swapped, agent2 is candidate
        if res == GameResult.AGENT2_WIN:
            wins_cand += 1
        elif res == GameResult.AGENT1_WIN:
            wins_base += 1
        else:
            draws += 1

    total = max(1, wins_cand + wins_base + draws)
    wr_cand = wins_cand / total
    return wr_cand, {"candidate": wins_cand, "baseline": wins_base, "draw": draws}

def perturb(base: dict, rnd: random.Random) -> dict:
    cand = dict(base)
    for k in TUNE_KEYS:
        scale = 1.0 + rnd.uniform(-0.25, 0.25)
        cand[k] = round(cand[k] * scale, 3)
    return cand

def worker_trial(start_weights: dict, games: int, seed: int, baseline: dict):
    rnd = random.Random(seed)
    cand = perturb(start_weights, rnd)
    wr, stats = eval_candidate(cand, games, seed*7 + 13, baseline=baseline)
    return (wr, cand, stats)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()-2))
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--baseline", type=str, default="weights.json",
                    help="Optional path to baseline weights JSON; falls back to heuristics defaults if missing.")
    args = ap.parse_args()

    # Load external baseline if provided
    baseline = dict(BASELINE_WEIGHTS)
    bp = ROOT / args.baseline
    if bp.exists():
        try:
            with open(bp) as f:
                ext = json.load(f)
            for k,v in ext.items():
                if k in baseline:
                    baseline[k] = v
            print(f"[INFO] Loaded baseline from {bp}")
        except Exception as e:
            print(f"[WARN] Could not load baseline {bp}: {e}")

    # Start weights = current heuristics (or defaults)
    start = dict(BASELINE_WEIGHTS)

    if args.baseline_only:
        wr, stats = eval_candidate(start, args.games, seed=1337, baseline=baseline)
        print(f"[BASELINE vs BASELINE] WR(candidate)={wr:.3f}  stats={stats}")
        return

    print(f"[FAST-TRAIN] trials={args.trials}  games/cand={args.games}  workers={args.workers}")
    with mp.Pool(processes=args.workers) as pool:
        jobs = [pool.apply_async(worker_trial, (start, args.games, 1000+i, baseline)) for i in range(args.trials)]
        results = [j.get() for j in jobs]

    best = max(results, key=lambda t: t[0])
    wr, cand, stats = best
    print(f"[BEST] WR(candidate)={wr:.3f}  stats={stats}")
    print(f"[BEST] weights={cand}")

    out_path = ROOT / "weights_best.json"
    with open(out_path,"w") as f:
        json.dump(cand, f, indent=2)
    print(f"Wrote {out_path}")

    wr2, stats2 = eval_candidate(cand, max(300, args.games), seed=4242, baseline=baseline)
    print(f"[RECHECK] WR(candidate)={wr2:.3f}  stats={stats2}")

if __name__ == "__main__":
    mp.freeze_support()  # Windows
    main()
