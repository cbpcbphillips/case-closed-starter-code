# tools/shared_features.py
from __future__ import annotations

from typing import Optional, List

ACTIONS: List[str] = ["UP", "DOWN", "LEFT", "RIGHT"]

def feature_vector_from_state(s, last_move_hint: Optional[str] = None) -> list[float]:
    """
    Mirrors agent.py's _feature_vector_from_state for perfect compatibility.
    Output length = 10: [bias, my_len, opp_len, boosts, dr, dc, one-hot(last 4)]
    """
    try:
        r, c = s.me_head
        center_r, center_c = (s.H - 1) / 2.0, (s.W - 1) / 2.0

        my_len  = float(s.me_len)  / 120.0
        opp_len = float(s.opp_len) / 120.0
        boosts  = float(s.me_boosts) / 5.0
        dr = (r - center_r) / max(1.0, s.H / 2.0)
        dc = (c - center_c) / max(1.0, s.W / 2.0)

        last = last_move_hint if last_move_hint in ACTIONS else (s.me_last_dir or "UP")
        last_oh = [1.0 if last == a else 0.0 for a in ACTIONS]

        return [1.0, my_len, opp_len, boosts, dr, dc, *last_oh]
    except Exception:
        # Fallback: bias + 9 zeros
        return [1.0] + [0.0] * 9
