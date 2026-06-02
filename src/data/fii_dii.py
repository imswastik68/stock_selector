"""
NSE provisional FII/DII cash market data.

NSE publishes daily provisional data by ~6 PM IST.
URL: https://www.nseindia.com/api/fiidiiTradeReact

Returns net buy/sell amounts for FII (Foreign) and DII (Domestic) in cash segment.
Used as market-wide signal: heavy FII buying = reduce downtrend headwind.

Thresholds (₹ crore):
  FII net buy  > +500cr  → fii_buying_strong  (reduce headwind by 1)
  FII net sell > -500cr  → fii_selling_strong (increase headwind by 1)
  DII net buy  > +500cr  → dii_buying_strong  (moderately bullish, DII often counter-cyclical)
"""

from __future__ import annotations

import time

import requests

_NSE_HOME = "https://www.nseindia.com/"
_FII_DII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": _NSE_HOME,
}

_THRESHOLD_CR = 500.0  # ₹500 crore threshold for "strong" signal

_EMPTY = {
    "fii_net_cr": None,
    "dii_net_cr": None,
    "fii_buying_strong": False,
    "fii_selling_strong": False,
    "dii_buying_strong": False,
    "available": False,
}


def fetch_fii_dii_data() -> dict:
    """
    Fetch today's provisional FII/DII cash segment data from NSE.
    Returns dict with net amounts and signal flags.
    Available after ~6 PM IST; returns _EMPTY gracefully before publication.
    """
    session = requests.Session()
    session.headers.update(_HEADERS)
    try:
        session.get(_NSE_HOME, timeout=10)
        time.sleep(0.3)
    except Exception:
        pass

    try:
        resp = session.get(_FII_DII_URL, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"[fii_dii] fetch error: {exc}")
        return dict(_EMPTY)

    # NSE returns a list of dicts with keys like:
    # {"category": "FII/FPI *", "buyValue": "12345.67", "sellValue": "11234.56", "netValue": "1111.11"}
    fii_net = None
    dii_net = None

    for row in (data if isinstance(data, list) else []):
        cat = str(row.get("category", "")).upper()
        try:
            net = float(str(row.get("netValue", "0")).replace(",", ""))
        except (ValueError, TypeError):
            continue
        if "FII" in cat or "FPI" in cat:
            fii_net = net
        elif "DII" in cat:
            dii_net = net

    if fii_net is None and dii_net is None:
        print("[fii_dii] no data in response (market not yet closed or API changed)")
        return dict(_EMPTY)

    result = {
        "fii_net_cr": round(fii_net, 1) if fii_net is not None else None,
        "dii_net_cr": round(dii_net, 1) if dii_net is not None else None,
        "fii_buying_strong":  fii_net is not None and fii_net > _THRESHOLD_CR,
        "fii_selling_strong": fii_net is not None and fii_net < -_THRESHOLD_CR,
        "dii_buying_strong":  dii_net is not None and dii_net > _THRESHOLD_CR,
        "available": True,
    }
    print(
        f"[fii_dii] FII net={result['fii_net_cr']}cr  DII net={result['dii_net_cr']}cr"
        f"  fii_buying={result['fii_buying_strong']}  fii_selling={result['fii_selling_strong']}"
    )
    return result
