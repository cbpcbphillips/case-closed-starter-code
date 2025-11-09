#!/usr/bin/env python3
import os, json, random
from datetime import datetime
from flask import Flask, request, jsonify

from config import cfg
from state import parse_state
from heuristics import choose_by_heuristic, maybe_apply_boost

try:
    from case_closed_game import Direction
    HAS_DIRECTION = True
except Exception:
    HAS_DIRECTION = False

# --------------------------------------------------------------------------
# Champion / env config
# --------------------------------------------------------------------------
PARTICIPANT   = os.environ.get("PARTICIPANT",   str(cfg("PARTICIPANT", "ParticipantX")))
AGENT_NAME    = os.environ.get("AGENT_NAME",    str(cfg("AGENT_NAME", "AgentX")))
AGENT_VARIANT = os.environ.get("AGENT_VARIANT", str(cfg("AGENT_VARIANT", "A")))
PORT          = int(os.environ.get("PORT", str(os.environ.get("PORT") or 5008)))

LOG_FILE      = os.environ.get("LOG_FILE", str(cfg("LOG_FILE", ""))).strip()
POLICY_PATH   = os.environ.get("POLICY_WEIGHTS", str(cfg("POLICY_WEIGHTS", ""))).strip()
ENABLE_POLICY = int(os.environ.get("ENABLE_POLICY_HEAD", str(cfg("ENABLE_POLICY_HEAD", 1))))

app = Flask(__name__)

_POLICY = None
_POLICY_META = None

def _try_load_policy():
    global _POLICY, _POLICY_META
    if _POLICY is not None or not POLICY_PATH or not ENABLE_POLICY:
        return
    try:
        with open(POLICY_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        W = data.get("W")
        if isinstance(W, list) and W and isinstance(W[0], list):
            _POLICY = W
            _POLICY_META = {
                "actions": data.get("actions", ["UP","DOWN","LEFT","RIGHT"]),
                "dim": data.get("dim", len(W[0]))
            }
    except Exception:
        _POLICY = None
        _POLICY_META = None

def _safe_jsonl(path, record):
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _feature_vector_from_state(s, last_move_hint="UP"):
    try:
        r, c = s.me_head
        center_r, center_c = (s.H - 1)/2.0, (s.W - 1)/2.0
        my_len  = float(s.me_len)  / 120.0
        opp_len = float(s.opp_len) / 120.0
        boosts  = float(s.me_boosts) / 5.0
        dr = (r - center_r) / max(1.0, s.H/2.0)
        dc = (c - center_c) / max(1.0, s.W/2.0)
        actions = ["UP","DOWN","LEFT","RIGHT"]
        last = last_move_hint if last_move_hint in actions else (s.me_last_dir or "UP")
        last_oh = [1.0 if last == a else 0.0 for a in actions]
        return [1.0, my_len, opp_len, boosts, dr, dc, *last_oh]
    except Exception:
        return [1.0] + [0.0]*9

def _policy_argmax(legal_moves, features):
    if _POLICY is None or _POLICY_META is None:
        return None
    try:
        actions = _POLICY_META.get("actions", ["UP","DOWN","LEFT","RIGHT"])
        def dot(a, b): return sum(x*y for x, y in zip(a, b))
        logits = [dot(row, features) for row in _POLICY]
        NEG_INF = -1e9
        masked = [logits[i] if a in legal_moves else NEG_INF for i, a in enumerate(actions)]
        idx = max(range(len(masked)), key=lambda k: masked[k])
        return actions[idx]
    except Exception:
        return None

def _legal_from_payload(p):
    legal = p.get("legal_moves") or p.get("legalMoves") or ["UP","DOWN","LEFT","RIGHT"]
    return [str(m).upper() for m in legal]

# --------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------
def decide_move(raw_state):
    s = parse_state(raw_state)
    legal = _legal_from_payload(raw_state)

    rng = random.Random(hash((getattr(s, "turn", 0), s.me_head, s.opp_head)) & 0xFFFFFFFF)
    move = choose_by_heuristic(s, rng)
    if isinstance(move, Direction) and HAS_DIRECTION:
        move = move.name
    move = str(move).upper()

    _try_load_policy()
    if ENABLE_POLICY and _POLICY is not None:
        feats = _feature_vector_from_state(s, last_move_hint=move)
        pmove = _policy_argmax(legal, feats)
        if pmove in legal:
            move = pmove

    try:
        move2 = maybe_apply_boost(s, move)
        if move2:
            if isinstance(move2, Direction) and HAS_DIRECTION:
                move2 = move2.name
            move = str(move2).upper()
    except Exception:
        pass

    if move not in legal:
        move = legal[0]

    return move, s, legal

# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def info():
    return jsonify({
        "participant": PARTICIPANT,
        "agent_name": AGENT_NAME,
        "variant": AGENT_VARIANT,
        "policy_loaded": bool(_POLICY is not None),
        "champion_tag": cfg("CHAMPION_TAG", "none")
    }), 200

@app.route("/move", methods=["POST"])
def move():
    payload = request.get_json(silent=True) or {}
    mv, s, legal = decide_move(payload)

    # Minimal but useful rollout log for training
    rec = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "participant": PARTICIPANT,
        "agent": AGENT_NAME,
        "variant": AGENT_VARIANT,
        "turn": getattr(s, "turn", 0),
        "H": getattr(s, "H", None),
        "W": getattr(s, "W", None),

        "me": {
            "head": getattr(s, "me_head", None),
            "len": getattr(s, "me_len", None),
            "boosts": getattr(s, "me_boosts", None),
            "last": getattr(s, "me_last_dir", None)
        },
        "opp": {
            "head": getattr(s, "opp_head", None),
            "len": getattr(s, "opp_len", None),
            "boosts": getattr(s, "opp_boosts", None),
            "last": getattr(s, "opp_last_dir", None)
        },

        "legal": legal,
        "action": mv
    }
    _safe_jsonl(LOG_FILE, rec)

    return jsonify({"move": mv}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
