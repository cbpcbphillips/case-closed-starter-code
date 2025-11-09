# tools/simple_collect_and_train.py
from __future__ import annotations
import os, json, math, random
from typing import List, Tuple

import numpy as np

from case_closed_game import Game, Direction, GameResult
from state import parse_state
from heuristics import choose_by_heuristic
from tools.shared_features import feature_vector_from_state

ACTIONS = ["UP","DOWN","LEFT","RIGHT"]
A2I = {a:i for i,a in enumerate(ACTIONS)}
DIR_MAP = {"UP": Direction.UP,"DOWN":Direction.DOWN,"LEFT":Direction.LEFT,"RIGHT":Direction.RIGHT}

# --- tiny helpers -------------------------------------------------------------

def game_to_raw(game: Game, player_num: int) -> dict:
    return {
        "board": game.board.grid,
        "agent1_trail": game.agent1.get_trail_positions(),
        "agent2_trail": game.agent2.get_trail_positions(),
        "agent1_length": game.agent1.length,
        "agent2_length": game.agent2.length,
        "agent1_alive": game.agent1.alive,
        "agent2_alive": game.agent2.alive,
        "agent1_boosts": game.agent1.boosts_remaining,
        "agent2_boosts": game.agent2.boosts_remaining,
        "turn_count": game.turns,
        "player_number": player_num,
    }

def rtg_weights(result: GameResult, steps: int, discount: float) -> np.ndarray:
    """Signed returns-to-go weights; draws = zeros."""
    if result == GameResult.DRAW:
        return np.zeros(steps, dtype=np.float32)
    sign = +1.0 if result == GameResult.AGENT1_WIN else -1.0
    w = np.power(discount, np.arange(steps-1, -1, -1, dtype=np.float32)) * sign
    # normalize to keep gradients stable
    norm = max(1.0, np.linalg.norm(w) / math.sqrt(max(1, steps)))
    w /= norm
    return w

def sample_move(state_obj, rng: random.Random, eps: float) -> str:
    mv = choose_by_heuristic(state_obj, rng)
    if isinstance(mv, Direction): mv = mv.name
    mv = str(mv).upper()
    if rng.random() < eps:
        mv = rng.choice(ACTIONS)
    return mv

# --- simple numpy softmax head ------------------------------------------------

def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

def train_softmax_linear(
    X: np.ndarray, y: np.ndarray, w: np.ndarray,
    lr: float = 0.3, epochs: int = 10, weight_decay: float = 1e-4, clip: float = 5.0
) -> np.ndarray:
    """
    Train a 4xD weight matrix using weighted NLL with signed weights.
    W shape: (4, D). Returns W.
    """
    n, D = X.shape
    K = 4
    W = np.zeros((K, D), dtype=np.float32)

    for ep in range(epochs):
        # full-batch GD for simplicity/stability on small datasets
        logits = X @ W.T                     # (n, K)
        P = softmax_np(logits)               # (n, K)

        # build (p - onehot) with per-sample signed weights w
        grad = P
        grad[np.arange(n), y] -= 1.0         # (n, K)
        grad *= w[:, None]                   # weight each sample by signed RTG

        # grad wrt W: sum_i grad_i[:, None] * x_i[None, :]
        dW = grad.T @ X                      # (K, D)

        # L2 weight decay
        dW += weight_decay * W

        # clip
        gnorm = np.linalg.norm(dW)
        if gnorm > clip:
            dW *= (clip / (gnorm + 1e-12))

        # update
        W -= lr * dW / max(1, n)

        # report simple loss (for sanity)
        nll = - (w * np.log(P[np.arange(n), y] + 1e-12)).mean()
        print(f"[train] epoch {ep+1}/{epochs}  loss={nll:.6f}  ||grad||={gnorm:.3f}")

    return W

# --- collection + train driver -----------------------------------------------

def main():
    # small, quick defaults to sanity-check the loop
    GAMES        = int(os.environ.get("GAMES", "600"))
    EPSILON      = float(os.environ.get("EPSILON", "0.12"))
    DISCOUNT     = float(os.environ.get("DISCOUNT", "0.997"))
    SEED         = int(os.environ.get("SEED", "1337"))
    MAX_TURNS    = int(os.environ.get("MAX_TURNS", "200"))  # Game caps itself too

    LR           = float(os.environ.get("LR", "0.35"))
    EPOCHS       = int(os.environ.get("EPOCHS", "12"))
    WD           = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
    CLIP         = float(os.environ.get("CLIP", "5.0"))

    OUT_WEIGHTS  = os.environ.get("OUT_WEIGHTS", "tools/out/policy_weights.json")

    rng = random.Random(SEED)

    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ws: List[np.ndarray] = []

    for gi in range(1, GAMES+1):
        game = Game()

        p1_feats: List[List[float]] = []
        p2_feats: List[List[float]] = []
        p1_acts:  List[int] = []
        p2_acts:  List[int] = []

        while True:
            s1 = parse_state(game_to_raw(game, 1), role="agent1", H=game.board.height, W=game.board.width)
            s2 = parse_state(game_to_raw(game, 2), role="agent2", H=game.board.height, W=game.board.width)

            mv1 = sample_move(s1, rng, EPSILON)
            mv2 = sample_move(s2, rng, EPSILON)

            f1 = feature_vector_from_state(s1, s1.me_last_dir)
            f2 = feature_vector_from_state(s2, s2.me_last_dir)
            p1_feats.append(f1); p2_feats.append(f2)
            p1_acts.append(A2I[mv1.split(":")[0]])
            p2_acts.append(A2I[mv2.split(":")[0]])

            res = game.step(DIR_MAP[mv1.split(":")[0]], DIR_MAP[mv2.split(":")[0]],
                            ":BOOST" in mv1, ":BOOST" in mv2)
            if res is not None:
                # weights for each side
                w1 = rtg_weights(res, len(p1_feats), DISCOUNT)
                res2 = GameResult.DRAW
                if res == GameResult.AGENT1_WIN: res2 = GameResult.AGENT2_WIN
                elif res == GameResult.AGENT2_WIN: res2 = GameResult.AGENT1_WIN
                w2 = rtg_weights(res2, len(p2_feats), DISCOUNT)

                Xs.append(np.asarray(p1_feats, np.float32)); ys.append(np.asarray(p1_acts, np.int64)); ws.append(w1)
                Xs.append(np.asarray(p2_feats, np.float32)); ys.append(np.asarray(p2_acts, np.int64)); ws.append(w2)
                break

        if gi % 50 == 0:
            print(f"[collect] {gi}/{GAMES} games...")

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    w = np.concatenate(ws, axis=0)

    # filter out draws (zero weights)
    m = (w != 0)
    X, y, w = X[m], y[m], w[m]
    print(f"[dataset] X={X.shape}, positives={np.sum(w>0)}, negatives={np.sum(w<0)}")

    # train W (4 x D)
    W = train_softmax_linear(X, y, w, lr=LR, epochs=EPOCHS, weight_decay=WD, clip=CLIP)

    # save
    os.makedirs(os.path.dirname(OUT_WEIGHTS), exist_ok=True)
    with open(OUT_WEIGHTS, "w", encoding="utf-8") as f:
        json.dump({"actions": ACTIONS, "dim": int(X.shape[1]), "W": W.tolist()}, f)
    print(f"[done] wrote {OUT_WEIGHTS}")

if __name__ == "__main__":
    main()
