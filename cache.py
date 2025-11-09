# cache.py
import os, sqlite3, threading, hashlib, json, time
from functools import lru_cache

_DB = os.getenv("CACHE_DB", "cache/cc_cache.sqlite")
os.makedirs(os.path.dirname(_DB), exist_ok=True)
_LOCK = threading.Lock()

def _conn():
    c = sqlite3.connect(_DB, check_same_thread=False)
    c.execute("""CREATE TABLE IF NOT EXISTS kv(
        k TEXT PRIMARY KEY,
        v TEXT NOT NULL,
        ts INTEGER NOT NULL
    )""")
    return c

_CONN = _conn()
_ENABLE = bool(int(os.getenv("CACHE_ENABLE", "1")))
_READONLY = bool(int(os.getenv("CACHE_READONLY", "0")))

def hash_key(parts) -> str:
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        if isinstance(p, (bytes, bytearray)):
            h.update(p)
        else:
            h.update(str(p).encode())
        h.update(b"|")
    return h.hexdigest()

@lru_cache(maxsize=50000)
def _mem_get(k: str):
    if not _ENABLE: return None
    with _LOCK:
        row = _CONN.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
    return json.loads(row[0]) if row else None

def get(k: str):
    return _mem_get(k)

def put(k: str, value: dict):
    if not _ENABLE or _READONLY: return
    s = json.dumps(value, separators=(",", ":"))
    now = int(time.time())
    with _LOCK:
        _CONN.execute("INSERT OR REPLACE INTO kv(k,v,ts) VALUES(?,?,?)", (k, s, now))
        _CONN.commit()
