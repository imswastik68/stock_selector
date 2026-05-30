"""
Download NIFTY 500 and NSE SME stock lists into data/.
Run once before first scan: python3 scripts/download_universe.py

Strategy (tried in order):
  1. archives.nseindia.com  — static CSV, no JS/cookie required
  2. nsetools.get_stock_codes() — returns all NSE equities
  3. Embedded NIFTY100 seed list — bare minimum fallback
"""

import io
import sys
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# archives.nseindia.com serves static files; no JS session needed
ARCHIVE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Minimal NIFTY100 seed so the pipeline runs even when all network fetches fail
NIFTY100_SEED = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
    "BHARTIARTL", "KOTAKBANK", "ITC", "LT", "AXISBANK", "ASIANPAINT", "MARUTI",
    "SUNPHARMA", "TITAN", "ULTRACEMCO", "BAJFINANCE", "NESTLEIND", "WIPRO",
    "HCLTECH", "POWERGRID", "ONGC", "NTPC", "COALINDIA", "INDUSINDBK", "M&M",
    "BAJAJFINSV", "TECHM", "ADANIPORTS", "JSWSTEEL", "TATASTEEL", "GRASIM",
    "CIPLA", "DRREDDY", "DIVISLAB", "BRITANNIA", "EICHERMOT", "BPCL", "HEROMOTOCO",
    "SHREECEM", "APOLLOHOSP", "TATACONSUM", "SBILIFE", "HDFCLIFE", "UPL",
    "BAJAJ-AUTO", "HINDALCO", "VEDL", "ADANIENT", "ICICIPRULI", "PIDILITIND",
    "DABUR", "MARICO", "COLPAL", "MCDOWELL-N", "BERGEPAINT", "BANDHANBNK",
    "AUROPHARMA", "TORNTPHARM", "LUPIN", "BIOCON", "HAVELLS", "VOLTAS",
    "MPHASIS", "COFORGE", "PERSISTENT", "LTIM", "OFSS", "INFY", "ZOMATO",
    "PAYTM", "NYKAA", "DELHIVERY", "IRCTC", "TATAPOWER", "ADANIGREEN",
    "ADANITRANS", "ADANIPORTS", "SIEMENS", "ABB", "CUMMINSIND", "THERMAX",
    "BHEL", "BEL", "HAL", "COCHINSHIP", "GRSE", "MAZDA", "DCBBANK",
    "RBLBANK", "FEDERALBNK", "IDFCFIRSTB", "BANKBARODA", "PNB", "CANBK",
    "UNIONBANK", "INDIANB", "CENTRALBK", "UCOBANK",
]


def _try_archives_csv(index_filename: str) -> list[str]:
    """Fetch an NSE index CSV from the archives subdomain."""
    url = f"https://archives.nseindia.com/content/indices/{index_filename}"
    try:
        resp = requests.get(url, headers=ARCHIVE_HEADERS, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        sym_col = next((c for c in df.columns if "symbol" in c.lower()), None)
        if sym_col is None:
            return []
        return df[sym_col].dropna().str.strip().str.upper().tolist()
    except Exception as exc:
        print(f"[universe] archives fetch ({index_filename}) failed: {exc}")
        return []


def _try_nsetools_all_stocks() -> list[str]:
    """Use nsetools to get all NSE-listed stock codes."""
    try:
        from nsetools import Nse
        nse = Nse()
        codes = nse.get_stock_codes()  # returns {symbol: company_name}
        symbols = [s.upper() for s in codes if s and s != "SYMBOL"]
        print(f"[universe] nsetools: got {len(symbols)} stock codes")
        return symbols
    except Exception as exc:
        print(f"[universe] nsetools fallback failed: {exc}")
        return []


def download_nifty500() -> int:
    out = DATA_DIR / "nifty500.csv"

    # 1. Try archives subdomain (static CSV, most reliable)
    symbols = _try_archives_csv("ind_nifty500list.csv")

    # 2. Try nsetools (all NSE equities — wider universe, also fine)
    if not symbols:
        print("[universe] trying nsetools for all NSE equities...")
        symbols = _try_nsetools_all_stocks()

    # 3. Seed fallback — small but non-empty
    if not symbols:
        print("[universe] using embedded NIFTY100 seed list")
        symbols = NIFTY100_SEED

    df = pd.DataFrame({"Symbol": symbols})
    df.to_csv(out, index=False)
    print(f"[universe] NIFTY500: saved {len(symbols)} symbols → {out}")
    return len(symbols)


def download_sme() -> int:
    out = DATA_DIR / "sme_list.csv"

    symbols = _try_archives_csv("ind_niftysme500list.csv")
    if not symbols:
        # SME list is nice-to-have; skip silently
        print("[universe] SME list unavailable — skipping (non-fatal)")
        return 0

    df = pd.DataFrame({"Symbol": symbols})
    df.to_csv(out, index=False)
    print(f"[universe] SME: saved {len(symbols)} symbols → {out}")
    return len(symbols)


if __name__ == "__main__":
    n500 = download_nifty500()
    sme = download_sme()

    if n500 == 0:
        print("\nERROR: all fetch strategies failed")
        sys.exit(1)

    print(f"\nDone. Universe: {n500 + sme} tickers ({n500} main + {sme} SME)")
    if n500 == len(NIFTY100_SEED):
        print("WARNING: using seed fallback list only. For full coverage:")
        print("  Download ind_nifty500list.csv from nseindia.com and save to data/nifty500.csv")
