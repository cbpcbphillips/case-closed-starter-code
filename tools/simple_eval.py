# tools/simple_eval.py
from __future__ import annotations
import os, json, random
from typing import Optional

import numpy as np

from case_closed_game import Game, Direction, GameResult
from state import parse_state
from heuristics import choose_by_heuristic
from tools.shared_features import feature_vector_from_state

ACTIONS = ["UP","DOWN","LEFT","RIGHT"]
DIR_MAP = {"UP": Direction.UP,"DOWN":Direction.DOWN,"LEFT":Direction.LEFT,"RIGHT":Direction.RIGHT}

def load_policy(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    W = np.asarray(data["W"], dtype=np.float32)  # (4, D)
    actions = data.get("actions", ACTIONS)
    return W, actions

def policy_argmax(legal: list[str], feats: np.ndarray, W: np.ndarray, actions: list[str]) -> Optional[str]:
    logits = W @ feats.astype(np.float32)
    # mask
    scores = []
    for a_idx, a in enumerate(actions):
        scores.append(logits[a_idx] if a in legal else -1e9)
    idx = int(np.argmax(scores))
    return actions[idx]

def legal_from_state_dict(state_dict: dict) -> list[str]:
    legal = state_dict.get("legal_moves") or state_dict.get("legalMoves")
    if legal:
        return [str(m).upper() for m in legal]
    return ACTIONS

def main():
    POLICY = os.environ.get("POLICY", "tools/out/policy_weights.json")
    GAMES  = int(os.environ.get("EVAL_GAMES", "500"))
    rng = random.Random(42)

    W, actions = load_policy(POLICY)

    hpp_wins = h_wins = draws = 0

    for _ in range(GAMES):
        g = Game()
        while True:
            s1d = {
                "board": g.board.grid,
                "agent1_trail": g.agent1.get_trail_positions(),
                "agent2_trail": g.agent2.get_trail_positions(),
                "agent1_length": g.agent1.length,
                "agent2_length": g.agent2.length,
                "agent1_alive": g.agent1.alive,
                "agent2_alive": g.agent2.alive,
                "agent1_boosts": g.agent1.boosts_remaining,
                "agent2_boosts": g.agent2.boosts_remaining,
                "turn_count": g.turns,
            }
            s2d = dict(s1d)

            s1 = parse_state(s1d, role="agent1", H=g.board.height, W=g.board.width)
            s2 = parse_state(s2d, role="agent2", H=g.board.height, W=g.board.width)

            # Agent1: heuristic -> policy refinement
            mv1 = choose_by_heuristic(s1, rng)
            if isinstance(mv1, Direction): mv1 = mv1.name
            mv1 = str(mv1).upper()
            feats1 = np.asarray(feature_vector_from_state(s1, s1.me_last_dir), dtype=np.float32)
            legal1 = ACTIONS  # judge doesn't give legals; rely on engine to reject opposite-direction, etc.
            pmv1 = policy_argmax(legal1, feats1, W, actions)
            if pmv1 in legal1:
                mv1 = pmv1

            # Agent2: pure heuristic
            mv2 = choose_by_heuristic(s2, rng)
            if isinstance(mv2, Direction): mv2 = mv2.name
            mv2 = str(mv2).upper()

            res = g.step(DIR_MAP[mv1], DIR_MAP[mv2], False, False)
            if res is not None:
                if res == GameResult.AGENT1_WIN: hpp_wins += 1
                elif res == GameResult.AGENT2_WIN: h_wins += 1
                else: draws += 1
                break

    total = hpp_wins + h_wins + draws
    print(f"[eval] policy+heuristic vs heuristic  |  W: {hpp_wins}  L: {h_wins}  D: {draws}  (N={total})")

if __name__ == "__main__":
    main()
