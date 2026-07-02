"""
F&O-eligible stock list — gates short-side picks.

Indian cash-market shorting is intraday-only; holding a short for 5-10 trading
days requires stock futures, which only exist for F&O-listed individual
securities (~210 names). This module is the source of truth for that set.

Source: NSE's fo_mktlots.csv (individual-securities lot-size table). The
old archives.nseindia.com path now serves a PDF; nsearchives.nseindia.com
still serves the real CSV as of 2026-07.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

DATA_DIR   = Path(__file__).parent.parent.parent / "data"
STATIC_CSV = DATA_DIR / "fo_eligible.csv"

FO_MKTLOTS_URL = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
ARCHIVE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/csv,*/*",
    "Accept-Encoding": "gzip, deflate, br",
}

# Index/underlying rows that appear at the top of fo_mktlots.csv — not tradeable
# stocks, exclude from the eligible-symbol set.
_INDEX_UNDERLYINGS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "NIFTYNXT50", "MIDCPNIFTY", "SENSEX", "BANKEX"}


def _parse_mktlots(raw_text: str) -> pd.DataFrame:
    """Parse fo_mktlots.csv text -> DataFrame[symbol, lot_size], individual stocks only."""
    df = pd.read_csv(StringIO(raw_text))
    df.columns = [c.strip() for c in df.columns]
    if len(df.columns) < 3:
        return pd.DataFrame(columns=["symbol", "lot_size"])

    sym_col = next((c for c in df.columns if "symbol" in c.lower()), None)
    if sym_col is None:
        return pd.DataFrame(columns=["symbol", "lot_size"])
    lot_col = df.columns[2]   # near-month lot size; header text (e.g. "JUL-26") changes monthly

    out = pd.DataFrame({
        "symbol":   df[sym_col].astype(str).str.strip(),
        "lot_size": pd.to_numeric(df[lot_col], errors="coerce"),
    })
    out = out[out["symbol"].str.upper() != "SYMBOL"]                 # repeated sub-header row
    out = out[~out["symbol"].str.upper().isin(_INDEX_UNDERLYINGS)]   # index contracts
    out = out[out["symbol"] != ""]
    out = out.dropna(subset=["symbol", "lot_size"])
    return out.drop_duplicates(subset="symbol").reset_index(drop=True)


def _fetch_live() -> pd.DataFrame:
    resp = requests.get(FO_MKTLOTS_URL, headers=ARCHIVE_HEADERS, timeout=15)
    resp.raise_for_status()
    return _parse_mktlots(resp.text)


def refresh_static_csv() -> bool:
    """Fetch live and overwrite data/fo_eligible.csv. Returns True on success. Never raises."""
    try:
        df = _fetch_live()
        if len(df) < 100:
            print(f"[fo_list] suspiciously small list ({len(df)}) — not overwriting static CSV")
            return False
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(STATIC_CSV, index=False)
        print(f"[fo_list] refreshed {STATIC_CSV.name}: {len(df)} F&O-eligible symbols")
        return True
    except Exception as exc:
        print(f"[fo_list] refresh failed: {exc}")
        return False


def fetch_fo_eligible() -> set[str]:
    """
    Set of .NS tickers eligible for stock futures (individual securities only).
    Order: today's cache -> live fetch -> committed static CSV fallback.
    Never returns empty if data/fo_eligible.csv exists (fail-closed for callers:
    an empty result routes all sells to watch-only, which is safe, not silent).
    """
    from src.cache import load_today, save_today

    cached = load_today("fo_eligible")
    if cached:
        return set(cached)

    try:
        df = _fetch_live()
        if len(df) >= 100:
            symbols = {f"{s}.NS" for s in df["symbol"]}
            save_today("fo_eligible", sorted(symbols))
            return symbols
        print(f"[fo_list] live fetch too small ({len(df)}), falling back to static CSV")
    except Exception as exc:
        print(f"[fo_list] live fetch failed: {exc}, falling back to static CSV")

    if STATIC_CSV.exists():
        df = pd.read_csv(STATIC_CSV)
        symbols = {f"{s}.NS" for s in df["symbol"].dropna()}
        print(f"[fo_list] loaded {len(symbols)} symbols from static fallback")
        return symbols

    print("[fo_list] WARNING: no live data and no static CSV — returning empty set")
    return set()


def fetch_fo_lot_sizes() -> dict[str, int]:
    """Ticker (.NS) -> current lot size. For futures-aware position sizing (later phase)."""
    if STATIC_CSV.exists():
        df = pd.read_csv(STATIC_CSV)
        return {
            f"{row.symbol}.NS": int(row.lot_size)
            for row in df.itertuples()
            if pd.notna(row.lot_size)
        }
    return {}
