"""
Promoter / insider buying tracker.

Two layers:
  1. Bulk/block deal detection (same-day, from bulk_deals data passed in)
  2. Quarterly SHP delta (promoter % change vs prior quarter via yfinance)

fetch_promoter_signals(tickers, bulk_deals) → {ticker: {"promoter_pct": float, "promoter_bought": bool}}
"""

from __future__ import annotations

import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yfinance as yf

DATA_DIR = Path(__file__).parent.parent.parent / "data"
SHP_CACHE = DATA_DIR / "promoter_shp.json"


def _load_shp_cache() -> dict[str, float]:
    """Load saved promoter % from last run."""
    if SHP_CACHE.exists():
        try:
            return json.loads(SHP_CACHE.read_text())
        except Exception:
            pass
    return {}


def _save_shp_cache(data: dict[str, float]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SHP_CACHE.write_text(json.dumps(data))


def _get_promoter_pct(ticker: str) -> float | None:
    """
    Fetch insider/promoter holding % from yfinance major_holders.
    Returns % held by insiders (proxy for promoters in Indian market), or None on failure.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = yf.Ticker(ticker)
            holders = t.major_holders
        if holders is None or holders.empty:
            return None
        # yfinance 1.x major_holders: single column "Value", row index = metric name
        # e.g. index "insidersPercentHeld" -> Value 0.511 (fraction)
        for metric, row in holders.iterrows():
            if "insider" in str(metric).lower():
                try:
                    pct = float(row.iloc[0])
                    if pct <= 1.0:
                        pct *= 100  # convert fraction to percent
                    return round(pct, 2)
                except (ValueError, TypeError):
                    pass
    except Exception:
        pass
    return None


def fetch_promoter_signals(
    tickers: list[str],
    bulk_deals: list[dict],
) -> dict[str, dict]:
    """
    For each ticker return:
      promoter_pct     — current insider/promoter holding %
      promoter_bought  — True if stake increased vs prior quarter OR bulk deal by promoter today
    """
    prev_cache = _load_shp_cache()
    new_cache: dict[str, float] = {}
    results: dict[str, dict] = {}

    # Bulk deal promoter detection (free, same-day signal)
    bulk_promoter: set[str] = {
        d["ticker"] for d in bulk_deals
        if d.get("is_promoter") and d.get("quantity", 0) > 0
    }

    # Parallel fetch — yfinance is I/O bound, 8 workers saves ~8× time
    def _fetch(ticker: str) -> tuple[str, float | None]:
        return ticker, _get_promoter_pct(ticker)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch, t): t for t in tickers}
        pct_map: dict[str, float | None] = {}
        for fut in as_completed(futures):
            t, pct = fut.result()
            pct_map[t] = pct

    for ticker in tickers:
        pct = pct_map.get(ticker)
        prev_pct = prev_cache.get(ticker)

        shp_increased = (
            pct is not None
            and prev_pct is not None
            and pct - prev_pct >= 0.5
        )

        promoter_bought = ticker in bulk_promoter or shp_increased

        results[ticker] = {
            "promoter_pct": pct,
            "promoter_bought": promoter_bought,
            "promoter_deal_today": ticker in bulk_promoter,
            "shp_delta_pct": round(pct - prev_pct, 2) if (pct and prev_pct) else None,
        }

        if pct is not None:
            new_cache[ticker] = pct

    # Merge new readings into cache (don't wipe tickers not fetched this run)
    merged = {**prev_cache, **new_cache}
    _save_shp_cache(merged)

    bought = [t for t, v in results.items() if v["promoter_bought"]]
    print(f"[promoter] fetched {len(results)} tickers, {len(bought)} with promoter buying signal")
    return results
