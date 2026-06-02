"""
Gift Nifty (GIFT City) futures gap fetcher.

Gift Nifty trades 6 AM–11:30 PM IST and is the primary pre-market indicator
for Nifty 50 direction. A gap of > +1% = strong buy bias; < -1% = sell bias.

Used in pre-market and EOD scans to modulate the headwind penalty applied to candidates.
In EOD context, Gift Nifty reflects where tomorrow's open is expected to be.

NSE API endpoint: https://www.nseindia.com/api/liveNSEGIFT
NOTE: This endpoint must be verified against the live NSE site — if it returns
a 404 or unexpected schema, the function returns _EMPTY gracefully (no crash,
feature simply disabled). Verify via browser devtools: Network tab → search "GIFT".
"""

from __future__ import annotations

import time

import requests

_NSE_HOME = "https://www.nseindia.com/"
_GIFT_URL  = "https://www.nseindia.com/api/liveNSEGIFT"
_NIFTY_URL = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": _NSE_HOME,
}

_EMPTY = {
    "gift_price": None,
    "nifty_prev_close": None,
    "gap_pct": None,
    "gap_up_strong": False,   # gap > +1%
    "gap_down_strong": False, # gap < -1%
    "available": False,
}


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get(_NSE_HOME, timeout=10)
        time.sleep(0.3)
    except Exception:
        pass
    return s


def fetch_gift_nifty() -> dict:
    """
    Fetch current Gift Nifty price and compute gap vs prior Nifty close.
    Returns gap_pct (positive = gap up, negative = gap down).
    Only meaningful before 9:15 AM IST (pre-market). After market open,
    Nifty itself is the better reference.
    """
    session = _make_session()

    gift_price = None
    nifty_prev = None

    # Fetch Gift Nifty current price
    try:
        resp = session.get(_GIFT_URL, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # Try multiple response shapes (NSE sometimes changes schema)
        records = data.get("data", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        if isinstance(records, list) and records:
            item = records[0]
            # Try known NSE field names in priority order
            for field in ("lastPrice", "last", "ltp", "previousClose", "price"):
                val = item.get(field)
                if val is not None and val != 0:
                    gift_price = float(str(val).replace(",", ""))
                    break
    except Exception as exc:
        print(f"[gift_nifty] Gift Nifty fetch error: {exc}")

    # Fetch Nifty 50 previous close
    try:
        resp = session.get(_NIFTY_URL, timeout=15)
        resp.raise_for_status()
        idx_data = resp.json()
        # Returns {"data": [{"symbol": "NIFTY 50", "previousClose": 22200.5, ...}]}
        records = idx_data.get("data", []) if isinstance(idx_data, dict) else []
        for row in records:
            if "NIFTY 50" in str(row.get("symbol", "")).upper() or \
               "NIFTY50" in str(row.get("indexSymbol", "")).upper():
                nifty_prev = float(str(row.get("previousClose", 0)).replace(",", ""))
                break
    except Exception as exc:
        print(f"[gift_nifty] Nifty prev close fetch error: {exc}")

    if gift_price is None or nifty_prev is None or nifty_prev == 0:
        print("[gift_nifty] insufficient data — skipping gap signal")
        return dict(_EMPTY)

    gap_pct = round((gift_price / nifty_prev - 1) * 100, 2)
    result = {
        "gift_price": round(gift_price, 2),
        "nifty_prev_close": round(nifty_prev, 2),
        "gap_pct": gap_pct,
        "gap_up_strong":   gap_pct > 1.0,
        "gap_down_strong": gap_pct < -1.0,
        "available": True,
    }
    print(
        f"[gift_nifty] Gift={result['gift_price']}  prev_close={result['nifty_prev_close']}"
        f"  gap={result['gap_pct']:+.2f}%"
        f"  gap_up={result['gap_up_strong']}  gap_down={result['gap_down_strong']}"
    )
    return result
