#!/usr/bin/env python3
"""
Elo updater. Reads tools/out/selfplay_results.csv and updates tools/out/elo.json.

Usage:
  python tools/elo.py --p1 AgentA --p2 AgentB --k 24

Assumptions:
  - p1 was the bot at --p1 URL used in selfplay.py
  - p2 was the bot at --p2 URL used in selfplay.py
  - results.csv contains rows with result in {'p1','p2','draw'}
"""
import os, sys, json, argparse, math, csv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def expected(rA, rB):
    return 1.0 / (1.0 + 10 ** ((rB - rA)/400))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1_name", default="AgentA")
    ap.add_argument("--p2_name", default="AgentB")
    ap.add_argument("--csv", default=os.path.join("tools","out","selfplay_results.csv"))
    ap.add_argument("--elo", default=os.path.join("tools","out","elo.json"))
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--draw_score", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.elo), exist_ok=True)
    if os.path.exists(args.elo):
        with open(args.elo, "r") as fh:
            ratings = json.load(fh)
    else:
        ratings = {}

    rA = ratings.get(args.p1_name, 1000.0)
    rB = ratings.get(args.p2_name, 1000.0)

    if not os.path.exists(args.csv):
        print(f"No results at {args.csv}. Run selfplay first.")
        return

    with open(args.csv, "r") as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            res = row["result"]
            EA = expected(rA, rB)
            EB = 1.0 - EA
            if res == "p1":
                SA, SB = 1.0, 0.0
            elif res == "p2":
                SA, SB = 0.0, 1.0
            else:
                SA = SB = args.draw_score
            rA += args.k * (SA - EA)
            rB += args.k * (SB - EB)

    ratings[args.p1_name] = rA
    ratings[args.p2_name] = rB
    with open(args.elo, "w") as fh:
        json.dump(ratings, fh, indent=2)

    print(json.dumps({"updated": True, "ratings": ratings}, indent=2))

if __name__ == "__main__":
    main()
