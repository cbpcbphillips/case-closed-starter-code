#!/usr/bin/env python3
"""
Tiny softmax policy head trained from logs/*.jsonl.
Outputs tools/out/policy_weights.json usable by the agent.

Usage:
  python tools/policy_head.py --logs logs --epochs 6 --lr 0.1

Integration (agent.py), minimal example inside send_move():
  import json, os, numpy as np
  Wfile = os.environ.get("POLICY_WEIGHTS", "tools/out/policy_weights.json")
  if os.path.exists(Wfile):
      W = np.asarray(json.load(open(Wfile))["W"])
      # logits = W @ features(s) ; choose argmax in legal moves
"""
import os, sys, json, argparse, glob
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ACTIONS = ["UP","DOWN","LEFT","RIGHT"]
A2I = {a:i for i,a in enumerate(ACTIONS)}

def feature_from_log(rec):
    """
    Build a simple feature vector from one log record.
    We DO NOT have the board; we use coarse info:
      - bias (1)
      - my_len, opp_len (normalized)
      - my_boosts
      - me_head (r,c) relative to center (10x20 guessed—adjust if needed)
      - last_move one-hot (4)
    """
    me = rec["me"]; opp = rec["opp"]
    r,c = me["head"]
    # If your board is not 20x20, tweak these means
    center_r, center_c = 10, 10

    my_len = me.get("len", 0) / 120.0
    opp_len = opp.get("len", 0) / 120.0
    boosts = me.get("boosts", 0) / 5.0
    dr = (r - center_r) / 10.0
    dc = (c - center_c) / 10.0

    # last move if present (infer from consecutive logs is better; here we use rec["move"])
    last = rec.get("move","UP").split(":")[0]
    last_oh = [1.0 if last==a else 0.0 for a in ACTIONS]
    return np.array([1.0, my_len, opp_len, boosts, dr, dc, *last_oh], dtype=np.float32)

def load_logs(log_dir):
    X=[]; y=[]
    for path in glob.glob(os.path.join(log_dir, "*.jsonl")):
        with open(path, "r") as fh:
            for line in fh:
                line=line.strip()
                if not line: continue
                try:
                    rec = json.loads(line)
                    mv = rec.get("move","UP").split(":")[0]
                    if mv not in A2I: continue
                    X.append(feature_from_log(rec))
                    y.append(A2I[mv])
                except Exception:
                    continue
    if not X:
        raise SystemExit("No training examples found in logs.")
    return np.stack(X), np.array(y, dtype=np.int64)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def train(X, y, epochs=6, lr=0.1, reg=1e-4, seed=0):
    np.random.seed(seed)
    n, d = X.shape
    k = len(ACTIONS)
    W = 0.01*np.random.randn(k, d).astype(np.float32)

    for _ in range(epochs):
        logits = X @ W.T              # [n,k]
        P = softmax(logits)
        Y = np.eye(k)[y]              # [n,k]
        grad = ((P - Y).T @ X) / n    # [k,d]
        W -= lr*(grad + reg*W)
    return W

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default=os.path.join("tools","out","policy_weights.json"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    X,y = load_logs(args.logs)
    W = train(X,y, epochs=args.epochs, lr=args.lr)
    with open(args.out, "w") as fh:
        json.dump({"W": W.tolist(), "actions": ACTIONS, "dim": int(X.shape[1])}, fh)
    print(json.dumps({"saved": args.out, "examples": int(X.shape[0]), "dim": int(X.shape[1])}, indent=2))

if __name__ == "__main__":
    main()
