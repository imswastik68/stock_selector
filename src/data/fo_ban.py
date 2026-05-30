"""F&O ban list delta tracker — uses archives.nseindia.com (no JS auth needed)."""

import json
import re
import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CACHE_FILE = DATA_DIR / "fo_ban_prev.json"

ARCHIVE_URL = "https://archives.nseindia.com/content/fo/fo_secban.csv"
ARCHIVE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,*/*",
    "Accept-Encoding": "gzip, deflate, br",
}


def _fetch_current_ban_list() -> list[str]:
    """
    Fetch today's F&O ban list from NSE archives.
    Returns list of ticker symbols (no .NS suffix).
    """
    try:
        resp = requests.get(ARCHIVE_URL, headers=ARCHIVE_HEADERS, timeout=15)
        resp.raise_for_status()
        text = resp.text.strip()

        # Format 1: "Securities in Ban For Trade Date 01-JUN-2026: NIL"
        if "NIL" in text.upper():
            print("[fo_ban] no securities in ban today")
            return []

        # Format 2: "Securities in Ban For Trade Date DD-MON-YYYY:\nSYM1\nSYM2"
        # or comma-separated: "SYM1,SYM2,SYM3"
        # Strip the header line if present
        lines = text.splitlines()
        symbols = []
        for line in lines:
            line = line.strip()
            if not line or "Securities in Ban" in line or "Trade Date" in line:
                continue
            # Could be comma-separated on one line
            for sym in re.split(r"[,\s]+", line):
                sym = sym.strip().upper()
                if sym and re.match(r"^[A-Z0-9&\-]+$", sym):
                    symbols.append(sym)

        print(f"[fo_ban] {len(symbols)} securities in ban")
        return symbols

    except Exception as exc:
        print(f"[fo_ban] fetch failed: {exc}")
        return []


def _load_prev() -> list[str]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            pass
    return []


def _save(symbols: list[str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(symbols))


def fetch_fo_ban_delta() -> list[str]:
    """
    Return tickers newly removed from the F&O ban list (liquidity unlock signal).
    Updates cache for next run. Returns list with .NS suffix.
    """
    prev = set(_load_prev())
    current = _fetch_current_ban_list()
    current_set = set(current)

    removed = prev - current_set
    _save(current)

    removed_ns = [f"{sym}.NS" for sym in sorted(removed)]
    print(f"[fo_ban] current ban: {len(current)}, removed today: {len(removed_ns)}")
    return removed_ns
