#!/usr/bin/env python3
"""
Evolution mode: search over BOOST_DELTA_T to maximize P1 win rate vs baseline P2.

Usage:
  python tools/evolve.py --p1 http://127.0.0.1:5008 --p2 http://127.0.0.1:5009 \
    --trials 6 --games 60 --out tools/out

It will:
  - Try several BOOST_DELTA_T variants (integers)
  - For each, run a short self-play set
  - Pick the best by win-rate
  - Write tools/out/best_env.txt with BOOST_DELTA_T to keep
"""
import os, sys, json, argparse, subprocess, random, time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def run_batch(p1, p2, seeds, boost_delta_t, py):
    env = os.environ.copy()
    env["BOOST_DELTA_T"] = str(boost_delta_t)
    wins = losses = draws = 0
    for s in seeds:
        cmd = [py, os.path.join("tools","fast_judge.py"), "--p1", p1, "--p2", p2, "--seed", str(s)]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        rc = proc.returncode
        if rc == 1: wins += 1
        elif rc == 2: losses += 1
        else: draws += 1
    total = max(1, wins+losses+draws)
    return {
        "boost_delta_t": boost_delta_t,
        "wins": wins, "losses": losses, "draws": draws,
        "win_rate": round(wins/total, 4)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", default="http://127.0.0.1:5008")
    ap.add_argument("--p2", default="http://127.0.0.1:5009")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default=os.path.join("tools","out"))
    ap.add_argument("--min_t", type=int, default=4)
    ap.add_argument("--max_t", type=int, default=14)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    random_ts = [random.randint(args.min_t, args.max_t) for _ in range(args.trials)]
    seeds = list(range(1, args.games+1))

    results = []
    for T in random_ts:
        res = run_batch(args.p1, args.p2, seeds, T, args.python)
        results.append(res)
        print(json.dumps(res))

    best = max(results, key=lambda r: r["win_rate"])
    best_env = os.path.join(args.out, "best_env.txt")
    with open(best_env, "w") as fh:
        fh.write(f"BOOST_DELTA_T={best['boost_delta_t']}\n")

    summary = {"best": best, "env_file": best_env}
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
