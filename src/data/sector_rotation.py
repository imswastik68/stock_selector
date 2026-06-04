"""
Sector rotation signal: fetch constituents of top-2 performing sector indices.

Flow:
  1. market_context.py computes sector_heatmap: {index_code: 5d_pct} via yfinance
  2. This module finds the top-2 sectors by 5d return
  3. Fetches NSE constituent lists from NSE archives CSV (no cookie required)
  4. Returns set of NSE tickers in hot sectors

The set is passed to scorer.py as hot_sector_tickers.
Stocks in a hot sector get +1 (sector_in_momentum signal).

NSE archives: https://archives.nseindia.com/content/indices/<filename>.csv
CSV format: Company Name,Industry,Symbol,Series,ISIN Code
"""

from __future__ import annotations

import requests

_ARCHIVES_BASE = "https://archives.nseindia.com/content/indices/"

# Map yfinance sector index codes → NSE archives CSV filename
_SECTOR_INDEX_TO_CSV: dict[str, str] = {
    "^CNXAUTO":    "ind_niftyautolist.csv",
    "^CNXBANK":    "ind_niftybanklist.csv",
    "^CNXIT":      "ind_niftyitlist.csv",
    "^CNXPHARMA":  "ind_niftypharmalist.csv",
    "^CNXFMCG":    "ind_niftyfmcglist.csv",
    "^CNXMETAL":   "ind_niftymetallist.csv",
    "^CNXREALTY":  "ind_niftyrealtylist.csv",
    "^CNXENERGY":  "ind_niftyenergylist.csv",
    "^CNXINFRA":   "ind_niftyinfralist.csv",
    "^CNXPSUBANK": "ind_niftypsubanklist.csv",
    "^CNXMEDIA":   "ind_niftymedialist.csv",
}

# Human-readable name for logging (derived from CSV filename, not needed for API calls)
_SECTOR_INDEX_TO_NAME: dict[str, str] = {
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

_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}


def _fetch_constituents(csv_file: str, sector_name: str) -> set[str]:
    """
    Download NSE archives CSV and parse Symbol column (index 2).
    Returns set of 'SYMBOL.NS' strings. No session/cookie required.
    """
    url = _ARCHIVES_BASE + csv_file
    try:
        resp = requests.get(url, timeout=15, headers=_HEADERS)
        resp.raise_for_status()
        tickers: set[str] = set()
        for line in resp.text.strip().split("\n")[1:]:  # skip header row
            parts = line.split(",")
            if len(parts) > 2:
                symbol = parts[2].strip()
                if symbol and " " not in symbol:  # exclude index rows (have spaces)
                    tickers.add(f"{symbol.upper()}.NS")
        print(f"[sector] {sector_name}: {len(tickers)} constituents fetched")
        return tickers
    except Exception as exc:
        print(f"[sector] fetch_constituents({sector_name}) error: {exc}")
        return set()


def fetch_hot_sector_tickers(sector_heatmap: dict[str, float]) -> set[str]:
    """
    Given the sector heatmap {index_code: 5d_pct_change}, find top-2 performing sectors
    and return all constituent tickers as a set of 'SYMBOL.NS' strings.

    Returns empty set if heatmap is unavailable or downloads fail.
    """
    if not sector_heatmap:
        return set()

    # Sort sectors by 5d return, take top N with positive return
    ranked = sorted(
        [(code, pct) for code, pct in sector_heatmap.items() if pct > 0],
        key=lambda x: x[1],
        reverse=True,
    )[:_TOP_N]

    if not ranked:
        print("[sector] no sectors with positive 5d return — no hot sector signal")
        return set()

    print(
        "[sector] hot sectors: "
        + ", ".join(
            f"{_SECTOR_INDEX_TO_NAME.get(code, code)}({pct:+.1f}%)"
            for code, pct in ranked
        )
    )

    hot_tickers: set[str] = set()
    for code, _pct in ranked:
        csv_file = _SECTOR_INDEX_TO_CSV.get(code)
        if not csv_file:
            continue
        name = _SECTOR_INDEX_TO_NAME.get(code, code)
        hot_tickers.update(_fetch_constituents(csv_file, name))

    print(f"[sector] total hot-sector tickers: {len(hot_tickers)}")
    return hot_tickers
