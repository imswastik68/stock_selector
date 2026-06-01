"""
Simple date-keyed disk cache for expensive daily fetches.

Cache files live in cache/YYYY-MM-DD_{key}.pkl — one file per key per day.
Old files (prior days) are cleaned up automatically on each save.
"""

from __future__ import annotations

import pickle
from datetime import date
from pathlib import Path

_CACHE_DIR = Path(__file__).parent.parent / "cache"


def load_today(key: str):
    """Return today's cached value for key, or None on cache miss."""
    path = _CACHE_DIR / f"{date.today().isoformat()}_{key}.pkl"
    if path.exists():
        try:
            data = pickle.loads(path.read_bytes())
            print(f"[cache] HIT  {key}  ({path.name})")
            return data
        except Exception:
            return None
    return None


def load_latest(key: str):
    """Return the most recent cached value for key regardless of date (for pre-market reuse)."""
    if not _CACHE_DIR.exists():
        return None
    files = sorted(_CACHE_DIR.glob(f"*_{key}.pkl"), reverse=True)
    if not files:
        return None
    try:
        data = pickle.loads(files[0].read_bytes())
        print(f"[cache] HIT  {key}  ({files[0].name}, pre-market reuse)")
        return data
    except Exception:
        return None


def save_today(key: str, data) -> None:
    """Save data to today's cache for key. Deletes prior-day files for the same key."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    # Remove stale files for this key (prior days)
    for old in _CACHE_DIR.glob(f"*_{key}.pkl"):
        if not old.name.startswith(today):
            old.unlink(missing_ok=True)
    path = _CACHE_DIR / f"{today}_{key}.pkl"
    path.write_bytes(pickle.dumps(data))
    print(f"[cache] SAVE {key}  ({path.name})")
