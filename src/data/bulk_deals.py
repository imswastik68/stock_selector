"""NSE bulk/block deals fetcher — uses archives.nseindia.com (no JS auth needed)."""

import io
import requests
import pandas as pd
from datetime import date

ARCHIVE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,*/*",
    "Accept-Encoding": "gzip, deflate, br",
}

BULK_URL = "https://archives.nseindia.com/content/equities/bulk.csv"
BLOCK_URL = "https://archives.nseindia.com/content/equities/block.csv"

FII_DII_KEYWORDS = [
    "fii", "fpi", "foreign", "lic", "sbi life", "hdfc life", "icici pru",
    "mutual fund", " mf", "insurance", "pension", "provident", "nps",
    "axis mf", "nippon", "kotak mf", "aditya birla", "dsp", "mirae",
    "motilal", "invesco", "franklin", "hsbc mf", "uti mf", "tata mf",
    "wasatch", "apt universal", "bnp paribas", "goldman", "morgan stanley",
    "jp morgan", "blackrock", "vanguard", "fidelity", "nomura",
]


def _is_fii_dii(actor: str) -> bool:
    a = actor.lower()
    return any(kw in a for kw in FII_DII_KEYWORDS)


def _fetch_csv(url: str, deal_type: str) -> list[dict]:
    try:
        resp = requests.get(url, headers=ARCHIVE_HEADERS, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = df.columns.str.strip()

        results = []
        for _, row in df.iterrows():
            ticker = str(row.get("Symbol") or "").strip().upper()
            actor = str(row.get("Client Name") or "").strip()
            qty = str(row.get("Quantity Traded") or "0").replace(",", "")
            price = str(row.get("Trade Price / Wght. Avg. Price") or "0").replace(",", "")

            if not ticker:
                continue

            try:
                results.append({
                    "ticker": f"{ticker}.NS",
                    "actor": actor,
                    "deal_type": deal_type,
                    "quantity": float(qty or 0),
                    "price": float(price or 0),
                    "is_fii_dii": _is_fii_dii(actor),
                })
            except ValueError:
                continue

        return results

    except Exception as exc:
        print(f"[bulk_deals] {deal_type} fetch failed: {exc}")
        return []


def fetch_bulk_deals(target_date: date | None = None) -> list[dict]:
    """
    Return today's NSE bulk + block deals from NSE archives CSV.
    Each dict: {ticker, actor, deal_type, quantity, price, is_fii_dii}
    """
    results = _fetch_csv(BULK_URL, "bulk") + _fetch_csv(BLOCK_URL, "block")
    print(f"[bulk_deals] {len(results)} deals fetched")
    return results
