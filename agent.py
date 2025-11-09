# agent.py
import os
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from state import parse_state
from heuristics import choose_by_heuristic, maybe_apply_boost

from case_closed_game import Game

# -----------------------------
# Flask API server setup
# -----------------------------
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}
game_lock = Lock()

# You can override these via env if you want different names on 5008/5009
PARTICIPANT = os.getenv("PARTICIPANT", "ParticipantX")
AGENT_NAME = os.getenv("AGENT_NAME", "AgentX")

# -----------------------------
# Small opening book (prevents early center collisions)
# Keys use (my_start_xy, opp_start_xy) with XY order (x, y) from judge payloads
# Values are the first few moves to play on turns 0..N-1
# -----------------------------
OPENINGS = {
    ((1, 2), (17, 15)): ["RIGHT", "RIGHT", "DOWN", "DOWN"],
    ((17, 15), (1, 2)): ["LEFT", "LEFT", "UP", "UP"],
}

def opening_move_if_any(state_dict, role: str):
    """
    Return a scripted opening move for early turns if we recognize the spawn pair.
    role: "agent1" or "agent2"
    """
    turn = int(state_dict.get("turn_count", 0))

    # only use for first few plies
    if turn >= 4:
        return None

    # trails are provided in (x, y)
    my_trail = state_dict.get("agent1_trail" if role == "agent1" else "agent2_trail", [])
    opp_trail = state_dict.get("agent2_trail" if role == "agent1" else "agent1_trail", [])

    if not my_trail or not opp_trail:
        return None

    my_start = tuple(my_trail[0])
    opp_start = tuple(opp_trail[0])

    seq = OPENINGS.get((my_start, opp_start))
    if not seq:
        return None

    if turn < len(seq):
        return seq[turn]
    return None

# -----------------------------
# HTTP endpoints
# -----------------------------
@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity."""
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge."""
    from case_closed_game import Direction  # local import avoids circulars

    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"])
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"])
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])

        # Best effort to keep directions reasonable if the judge ever expands payload
        # (We keep the existing direction if not provided.)
        # If you later add last_dir to payload, you can set it here.


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge pushes the current game state to the agent."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


def choose_move(s):
    """
    Single entrypoint for decision each tick:
    - choose_by_heuristic (with internal short beam/reply awareness)
    - then maybe apply a boost if it's beneficial/safe
    """
    base_move = choose_by_heuristic(s)           # "UP"/"DOWN"/"LEFT"/"RIGHT"
    final_move = maybe_apply_boost(s, base_move) # returns "DIR" or "DIR:BOOST"
    return final_move


@app.route("/send-move", methods=["GET"])
def send_move():
    """
    Judge requests the agent's move for the current tick.

    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT and :BOOST is optional.
    """
    player_number = request.args.get("player_number", default=1, type=int)

    # Snapshot the last posted state
    with game_lock:
        state = dict(LAST_POSTED_STATE)

    # If we haven't received a state yet, be defensive
    if not state:
        return jsonify({"move": "RIGHT"}), 200

    # Decide role for this tick
    role = "agent1" if player_number == 1 else "agent2"

    # Opening move first (prevents early symmetric collisions)
    om = opening_move_if_any(state, role)
    if om:
        return jsonify({"move": om}), 200

    # Parse to typed State and decide
    s = parse_state(state, role=role)
    move = choose_move(s)

    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state."""
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    # default port 5008; run second copy on 5009 with PORT=5009
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)
