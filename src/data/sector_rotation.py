"""
Sector rotation signal: fetch constituents of top-2 performing sector indices.

Flow:
  1. market_context.py already computes sector_heatmap: {index_code: 5d_pct}
  2. This module finds the top-2 sectors by 5d return
  3. Fetches NSE constituent lists for those sectors via NSE equity-stockIndices API
  4. Returns set of NSE tickers in hot sectors

The set is passed to scorer.py as hot_sector_tickers.
Stocks in a hot sector get +1 (sector_in_momentum signal).

NSE constituent API: https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20BANK
Requires NSE cookie session (same pattern as options.py).
"""

from __future__ import annotations

import time

import requests

_NSE_HOME = "https://www.nseindia.com/"
_CONSTITUENTS_URL = "https://www.nseindia.com/api/equity-stockIndices?index={index}"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": _NSE_HOME,
}

# Map yfinance sector index codes → NSE index name for the constituents API
_SECTOR_INDEX_TO_NSE: dict[str, str] = {
    "^CNXAUTO":    "NIFTY AUTO",
    "^CNXBANK":    "NIFTY BANK",
    "^CNXIT":      "NIFTY IT",
    "^CNXPHARMA":  "NIFTY PHARMA",
    "^CNXFMCG":    "NIFTY FMCG",
    "^CNXMETAL":   "NIFTY METAL",
    "^CNXREALTY":  "NIFTY REALTY",
    "^CNXENERGY":  "NIFTY ENERGY",
    "^CNXINFRA":   "NIFTY INFRA",
    "^CNXPSUBANK": "NIFTY PSU BANK",
    "^CNXMEDIA":   "NIFTY MEDIA",
}

_TOP_N = 2  # number of top sectors to track


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get(_NSE_HOME, timeout=10)
        time.sleep(0.3)
        s.get("https://www.nseindia.com/market-data/equity-derivatives-watch", timeout=10)
        time.sleep(0.3)
    except Exception:
        pass
    return s


def _fetch_constituents(session: requests.Session, nse_index: str) -> set[str]:
    """Fetch ticker symbols for a given NSE index. Returns set of 'SYMBOL.NS' strings."""
    url = _CONSTITUENTS_URL.format(index=requests.utils.quote(nse_index))
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        records = data.get("data", []) if isinstance(data, dict) else []
        tickers = set()
        for row in records:
            symbol = row.get("symbol", "")
            # NSE stock tickers never have spaces; index names do (e.g. "NIFTY BANK", "NIFTY AUTO").
            # Skip any row that's the index itself rather than a constituent stock.
            if symbol and " " not in symbol:
                tickers.add(f"{symbol.upper()}.NS")
        return tickers
    except Exception as exc:
        print(f"[sector] fetch_constituents({nse_index}) error: {exc}")
        return set()


def fetch_hot_sector_tickers(sector_heatmap: dict[str, float]) -> set[str]:
    """
    Given the sector heatmap {index_code: 5d_pct_change}, find top-2 performing sectors
    and return all constituent tickers as a set of 'SYMBOL.NS' strings.

    Returns empty set if heatmap is unavailable or API calls fail.
    """
    if not sector_heatmap:
        return set()

    # Sort sectors by 5d return, take top N
    ranked = sorted(
        [(code, pct) for code, pct in sector_heatmap.items() if pct > 0],
        key=lambda x: x[1],
        reverse=True,
    )[:_TOP_N]

    if not ranked:
        print("[sector] no sectors with positive 5d return — no hot sector signal")
        return set()

    top_sectors = [(code, _SECTOR_INDEX_TO_NSE.get(code, ""), pct) for code, pct in ranked]
    print(f"[sector] hot sectors: " + ", ".join(f"{name}({pct:+.1f}%)" for _, name, pct in top_sectors))

    session = _make_session()
    hot_tickers: set[str] = set()

    for code, nse_name, pct in top_sectors:
        if not nse_name:
            continue
        tickers = _fetch_constituents(session, nse_name)
        print(f"[sector] {nse_name}: {len(tickers)} constituents fetched")
        hot_tickers.update(tickers)

    print(f"[sector] total hot-sector tickers: {len(hot_tickers)}")
    return hot_tickers
