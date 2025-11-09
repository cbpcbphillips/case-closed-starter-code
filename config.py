# config.py
import json, os
from functools import lru_cache
from typing import Any

_CHAMPION_PATH = os.environ.get("CHAMPION_PATH", "champion.json")

@lru_cache(maxsize=1)
def _load_file() -> dict:
    try:
        with open(_CHAMPION_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def cfg(key: str, default: Any = None) -> Any:
    """
    Resolve config precedence:
      1) Environment variable (exact key)
      2) champion.json (CHAMPION_PATH; default 'champion.json')
      3) provided default
    """
    if key in os.environ:
        return os.environ[key]
    return _load_file().get(key, default)
