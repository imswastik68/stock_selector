"""NSE bulk/block deals fetcher — uses archives.nseindia.com (no JS auth needed)."""

import io
import requests
import pandas as pd

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

# Entities that are typically promoter / promoter-group buyers
PROMOTER_KEYWORDS = [
    "promoter", "director", "chairman", "managing director",
    "whole-time director", "wtd", "family trust", "family office",
    "holding", "ventures pvt", "industries pvt", "enterprises pvt",
    "private limited", "pvt. ltd", "pvt ltd",
]


def _is_fii_dii(actor: str) -> bool:
    a = actor.lower()
    return any(kw in a for kw in FII_DII_KEYWORDS)


def _is_promoter(actor: str) -> bool:
    a = actor.lower()
    return any(kw in a for kw in PROMOTER_KEYWORDS) and not _is_fii_dii(actor)


def _fetch_csv(url: str, deal_type: str) -> list[dict]:
    try:
        resp = requests.get(url, headers=ARCHIVE_HEADERS, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = df.columns.str.strip()

        side_col = next((c for c in df.columns if "buy" in c.lower() or "sell" in c.lower()), None)

        results = []
        for _, row in df.iterrows():
            ticker = str(row.get("Symbol") or "").strip().upper()
            actor = str(row.get("Client Name") or "").strip()
            qty = str(row.get("Quantity Traded") or "0").replace(",", "")
            price = str(row.get("Trade Price / Wght. Avg. Price") or "0").replace(",", "")
            side = str(row.get(side_col) or "").strip().upper() if side_col else ""

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
                    "is_promoter": _is_promoter(actor),
                    "side": side if side in ("BUY", "SELL") else "",
                })
            except ValueError:
                continue

        return results

    except Exception as exc:
        print(f"[bulk_deals] {deal_type} fetch failed: {exc}")
        return []


def fetch_bulk_deals() -> list[dict]:
    """
    Return today's NSE bulk + block deals from NSE archives CSV.
    Each dict: {ticker, actor, deal_type, quantity, price, is_fii_dii, is_promoter, side}
    side is "BUY"/"SELL" from NSE's Buy/Sell column, or "" if the column is missing
    (treated as BUY by callers — preserves pre-side-tracking behavior).
    """
    results = _fetch_csv(BULK_URL, "bulk") + _fetch_csv(BLOCK_URL, "block")
    print(f"[bulk_deals] {len(results)} deals fetched")
    return results
