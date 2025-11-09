#!/usr/bin/env python3
"""
Self-play harness that runs many fast_judge matches in parallel
and reports W/L/D. Requires tools/fast_judge.py in this repo.

Usage:
  python tools/selfplay.py --p1 http://127.0.0.1:5008 --p2 http://127.0.0.1:5009 -n 500 -w 8 --out tools/out

It writes:
  - tools/out/selfplay_results.csv  (seed, result)
  - tools/out/selfplay_summary.txt  (W/L/D, rates)
"""
import os, sys, csv, json, argparse, subprocess, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def run_one(seed, p1, p2, py="python"):
    """Run a single match via fast_judge. Returns ('p1'|'p2'|'draw', seed, rc, stdout, stderr)."""
    cmd = [py, os.path.join("tools","fast_judge.py"), "--p1", p1, "--p2", p2, "--seed", str(seed)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    rc  = proc.returncode

    # Map return code → result (fallback: parse stdout)
    if rc == 1:
        result = "p1"
    elif rc == 2:
        result = "p2"
    elif rc == 0:
        result = "draw"
    else:
        # Try to parse any obvious marker in stdout, else call it 'draw' but keep rc
        lowered = out.lower()
        if "p1" in lowered and "wins" in lowered:
            result = "p1"
        elif "p2" in lowered and "wins" in lowered:
            result = "p2"
        elif "draw" in lowered or "tie" in lowered:
            result = "draw"
        else:
            result = "draw"
    return result, seed, rc, out, err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", default="http://127.0.0.1:5008")
    ap.add_argument("--p2", default="http://127.0.0.1:5009")
    ap.add_argument("-n", "--num_games", type=int, default=200)
    ap.add_argument("-w", "--workers", type=int, default=8)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default=os.path.join("tools","out"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results_csv = os.path.join(args.out, "selfplay_results.csv")
    summary_txt = os.path.join(args.out, "selfplay_summary.txt")

    seeds = list(range(1, args.num_games+1))
    random.shuffle(seeds)

    W=L=D=0
    rows = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, s, args.p1, args.p2, args.python) for s in seeds]
        for f in as_completed(futs):
            result, seed, rc, out, err = f.result()
            if result == "p1": W += 1
            elif result == "p2": L += 1
            else: D += 1
            rows.append({"seed": seed, "result": result, "rc": rc})

    rows.sort(key=lambda r: r["seed"])
    with open(results_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["seed","result","rc"])
        w.writeheader()
        w.writerows(rows)

    total = max(1, W+L+D)
    summary = {
        "timestamp": datetime.utcnow().isoformat()+"Z",
        "p1": args.p1, "p2": args.p2, "num_games": total,
        "wins_p1": W, "wins_p2": L, "draws": D,
        "win_rate_p1": round(W/total, 4),
        "win_rate_p2": round(L/total, 4),
        "draw_rate": round(D/total, 4)
    }
    with open(summary_txt, "w") as fh:
        fh.write(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
