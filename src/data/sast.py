"""
SAST (Substantial Acquisition of Shares and Takeovers) insider filing fetcher.

SEBI mandates filing within 2 trading days of acquisition. Any SAST filing in the
last 5 calendar days indicates an institutional entity is accumulating stake.
Catches creeping acquisitions that never appear in daily bulk deals.

Returns {ticker: True} for tickers with at least one filing in the look-back window.
Fails gracefully on any error — returns empty dict.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta


def _pnsea_available() -> bool:
    try:
        import pnsea  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


def _sast_one(symbol: str, from_dt: str, to_dt: str) -> tuple[str, bool]:
    try:
        from pnsea import NSE
        nse = NSE()
        df = nse.insider.getSastData(symbol, from_date=from_dt, to_date=to_dt)
        if df is None:
            return symbol, False
        if hasattr(df, "empty") and df.empty:
            return symbol, False
        return symbol, bool(len(df) > 0)
    except Exception as exc:
        print(f"[sast] {symbol}: {exc}")
        return symbol, False


def fetch_sast_signals(tickers: list[str]) -> dict[str, bool]:
    """
    Return {ticker: True} for tickers with SAST filings in the last 5 calendar days.
    Strips .NS suffix and [PENNY] prefix for NSE API compatibility.
    """
    if not tickers:
        return {}

    if not _pnsea_available():
        print("[sast] pnsea not installed — skipping SAST signals (run: pip install pnsea)")
        return {}

    today   = date.today()
    from_dt = (today - timedelta(days=5)).strftime("%d-%m-%Y")
    to_dt   = today.strftime("%d-%m-%Y")

    clean: dict[str, str] = {
        t: t.replace(".NS", "").replace("[PENNY]", "")
        for t in tickers
    }

    raw: dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_sast_one, sym, from_dt, to_dt): orig
            for orig, sym in clean.items()
        }
        for fut in as_completed(futures):
            orig = futures[fut]
            _, hit = fut.result()
            raw[orig] = hit

    hits = [t for t, v in raw.items() if v]
    print(f"[sast] checked {len(tickers)} tickers, {len(hits)} with recent SAST filing")
    return raw
