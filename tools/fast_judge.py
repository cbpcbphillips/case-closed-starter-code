#!/usr/bin/env python3
"""
Fast local judge for self-play training.

Usage:
    python tools/fast_judge.py --p1 http://127.0.0.1:5008 --p2 http://127.0.0.1:5009 --seed 42

Returns:
    exit code 0 -> draw
    exit code 1 -> agent1 win
    exit code 2 -> agent2 win
"""

#!/usr/bin/env python3

# --- FORCE PROJECT ROOT ON PATH (drop-in fix, DO NOT EDIT) ---
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# --------------------------------------------------------------

import requests, argparse
from case_closed_game import Game, Direction, GameResult

TIMEOUT = 2.5  # lower for speed but safe

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--p1", required=True)
    p.add_argument("--p2", required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def send_state(url, game, player):
    payload = {
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
        "player_number": player
    }
    try:
        requests.post(f"{url}/send-state", json=payload, timeout=TIMEOUT)
    except:
        pass

def get_move(url, player, game):
    params = {
        "player_number": player,
        "turn_count": game.turns,
        "attempt_number": 1,
        "random_moves_left": 0
    }
    try:
        r = requests.get(f"{url}/send-move", params=params, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json().get("move")
    except:
        pass
    return None

def parse_move(move):
    if not move:
        return None, False
    parts = move.upper().split(":")
    dir_str = parts[0]
    use_boost = len(parts) > 1 and parts[1] == "BOOST"
    mapping = {
        "UP": Direction.UP,
        "DOWN": Direction.DOWN,
        "LEFT": Direction.LEFT,
        "RIGHT": Direction.RIGHT
    }
    return mapping.get(dir_str), use_boost

def main():
    args = parse_args()

    game = Game()
    # Seed not currently used by Game engine; optional future usage
    # random.seed(args.seed); numpy if needed

    # initial state push
    send_state(args.p1, game, 1)
    send_state(args.p2, game, 2)

    while True:
        # request moves
        m1 = get_move(args.p1, 1, game)
        m2 = get_move(args.p2, 2, game)

        d1, b1 = parse_move(m1)
        d2, b2 = parse_move(m2)

        # fallback to keep game going
        if d1 is None: d1, b1 = game.agent1.direction, False
        if d2 is None: d2, b2 = game.agent2.direction, False

        result = game.step(d1, d2, b1, b2)

        # update state
        send_state(args.p1, game, 1)
        send_state(args.p2, game, 2)

        if result is not None:
            if result == GameResult.DRAW:
                sys.exit(0)
            elif result == GameResult.AGENT1_WIN:
                sys.exit(1)
            elif result == GameResult.AGENT2_WIN:
                sys.exit(2)

        if game.turns > 500:
            sys.exit(0)

if __name__ == "__main__":
    main()
