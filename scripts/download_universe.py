"""
Download NSE stock universe into data/.
Run once before first scan: python3 scripts/download_universe.py

Strategy:
  1. NSE archives (no JS/cookie): Nifty500 + Midcap150 + Smallcap250 + SME
  2. Optional screener.in export (set SCREENER_EMAIL + SCREENER_PASSWORD in .env)
     — fetches a quality-filtered screen (PE<50, positive ROE, low debt)
  3. nsetools.get_stock_codes() fallback
  4. Embedded NIFTY100 seed list — bare minimum

Result: data/nifty500.csv + data/sme_list.csv (merged, deduplicated universe)
"""

import io
import os
import sys
import time
import requests
import pandas as pd
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

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
    """Fetch Nifty500 + Midcap150 + Smallcap250 merged into nifty500.csv."""
    out = DATA_DIR / "nifty500.csv"

    # Pull all three NSE index constituent lists
    all_symbols: list[str] = []
    for index_file in ["ind_nifty500list.csv", "ind_niftymidcap150list.csv", "ind_niftysmallcap250list.csv"]:
        syms = _try_archives_csv(index_file)
        if syms:
            print(f"[universe] {index_file}: {len(syms)} symbols")
            all_symbols.extend(syms)
        else:
            print(f"[universe] {index_file}: unavailable (non-fatal)")

    # Deduplicate preserving order
    symbols = list(dict.fromkeys(all_symbols))

    # 2. nsetools fallback
    if not symbols:
        print("[universe] trying nsetools for all NSE equities...")
        symbols = _try_nsetools_all_stocks()

    # 3. Seed fallback
    if not symbols:
        print("[universe] using embedded NIFTY100 seed list")
        symbols = NIFTY100_SEED

    # 4. Optional screener.in quality-filtered additions
    screener_syms = _try_screener_in()
    if screener_syms:
        before = len(symbols)
        symbols = list(dict.fromkeys(symbols + screener_syms))
        print(f"[universe] screener.in added {len(symbols) - before} new quality symbols")

    df = pd.DataFrame({"Symbol": symbols})
    df.to_csv(out, index=False)
    print(f"[universe] MAIN: saved {len(symbols)} symbols → {out}")
    return len(symbols)


def _try_nse_full_equity_list() -> list[str]:
    """
    Download NSE's full equity list (includes SME/SME-IPO stocks not in Nifty indices).
    URL: https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv
    """
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        resp = requests.get(url, headers=ARCHIVE_HEADERS, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        sym_col = next((c for c in df.columns if "symbol" in c.lower()), None)
        if sym_col is None:
            return []
        # Filter out debentures, ETFs, etc. — keep only EQ series
        if "SERIES" in df.columns:
            df = df[df["SERIES"].str.strip() == "EQ"]
        symbols = df[sym_col].dropna().str.strip().str.upper().tolist()
        print(f"[universe] NSE full equity list: {len(symbols)} EQ symbols")
        return symbols
    except Exception as exc:
        print(f"[universe] NSE full equity list failed: {exc}")
        return []


def download_sme() -> int:
    out = DATA_DIR / "sme_list.csv"

    # Try NSE full equity list (covers SME stocks not in Nifty indices)
    symbols = _try_nse_full_equity_list()
    if not symbols:
        print("[universe] SME list unavailable — skipping (non-fatal)")
        return 0

    df = pd.DataFrame({"Symbol": symbols})
    df.to_csv(out, index=False)
    print(f"[universe] SME: saved {len(symbols)} symbols → {out}")
    return len(symbols)


# ── screener.in optional integration ─────────────────────────────────────────

def _try_screener_in() -> list[str]:
    """
    Log in to screener.in with SCREENER_EMAIL + SCREENER_PASSWORD from env/.env
    and export a quality-filtered screen (PE<50, positive ROE, debt/equity<2).
    Returns list of NSE symbols, or [] if credentials not set or request fails.
    """
    email = os.environ.get("SCREENER_EMAIL", "")
    password = os.environ.get("SCREENER_PASSWORD", "")
    if not email or not password:
        print("[universe] SCREENER_EMAIL/PASSWORD not set — skipping screener.in")
        return []

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,*/*",
        "Referer": "https://www.screener.in/",
    })

    try:
        # Step 1: get CSRF token
        r = session.get("https://www.screener.in/login/", timeout=15)
        r.raise_for_status()
        csrf = ""
        for line in r.text.splitlines():
            if "csrfmiddlewaretoken" in line and 'value="' in line:
                csrf = line.split('value="')[1].split('"')[0]
                break
        if not csrf:
            print("[universe] screener.in: could not extract CSRF token")
            return []

        # Step 2: log in
        login_data = {
            "csrfmiddlewaretoken": csrf,
            "username": email,
            "password": password,
        }
        r = session.post("https://www.screener.in/login/", data=login_data,
                         headers={"Referer": "https://www.screener.in/login/"}, timeout=15)
        if "logout" not in r.text.lower() and r.url == "https://www.screener.in/login/":
            print("[universe] screener.in: login failed — check credentials")
            return []

        time.sleep(0.5)

        # Step 3: export quality screen as CSV
        # Query: PE < 50 AND Return on equity > 10 AND Debt to equity < 2
        query = "PE+Ratio+%3C+50+AND+Return+on+equity+%3E+10+AND+Debt+to+equity+%3C+2"
        export_url = f"https://www.screener.in/screens/new/?sort=&order=&query={query}&limit=200&format=csv"
        r = session.get(export_url, headers={"Referer": "https://www.screener.in/"}, timeout=30)
        r.raise_for_status()

        df = pd.read_csv(io.StringIO(r.text))
        # screener.in CSV has a "Symbol" column with NSE tickers
        sym_col = next((c for c in df.columns if "symbol" in c.lower()), None)
        if sym_col is None:
            print(f"[universe] screener.in: unexpected CSV columns: {list(df.columns)}")
            return []

        symbols = df[sym_col].dropna().str.strip().str.upper().tolist()
        print(f"[universe] screener.in quality screen: {len(symbols)} symbols")
        return symbols

    except Exception as exc:
        print(f"[universe] screener.in fetch failed: {exc}")
        return []


def refresh_fo_eligible() -> None:
    """Keep data/fo_eligible.csv fresh — short-side picks are gated against it."""
    try:
        from src.data.fo_list import refresh_static_csv
        refresh_static_csv()
    except Exception as exc:
        print(f"[universe] fo_eligible refresh failed (non-fatal): {exc}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

    n500 = download_nifty500()
    sme = download_sme()
    refresh_fo_eligible()

    if n500 == 0:
        print("\nERROR: all fetch strategies failed")
        sys.exit(1)

    print(f"\nDone. Universe: {n500 + sme} tickers ({n500} main + {sme} SME)")
    if n500 <= len(NIFTY100_SEED):
        print("WARNING: using seed fallback list only. For full coverage run again with internet.")
    print("\nFor screener.in quality filter, add to .env:")
    print("  SCREENER_EMAIL=your@email.com")
    print("  SCREENER_PASSWORD=yourpassword")
