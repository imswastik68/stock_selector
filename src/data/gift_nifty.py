"""
Gift Nifty / Nifty intraday gap fetcher.

NSE's liveNSEGIFT and equity-stockIndices endpoints were retired.
We now use the allIndices endpoint which returns live Nifty 50 last price
and previousClose — sufficient to compute the intraday gap signal.

Gap interpretation:
  pre-market (before 9:15 AM): last == previousClose → gap = 0 (neutral, correct)
  EOD (after 3:30 PM): last = today's close → gap = full day % move (better signal)
  > +1%  = gap_up_strong
  < -1%  = gap_down_strong
"""

from __future__ import annotations

import time

import requests

_NSE_HOME    = "https://www.nseindia.com/"
_ALL_INDICES = "https://www.nseindia.com/api/allIndices"

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
    "gap_up_strong": False,
    "gap_down_strong": False,
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
    Fetch Nifty 50 last price and previousClose from allIndices.
    Computes intraday gap: (last / previousClose - 1) * 100.
    In pre-market this equals 0 (neutral); in EOD it reflects today's full move.
    """
    session = _make_session()
    try:
        resp = session.get(_ALL_INDICES, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        indices = data.get("data", [])

        nifty_row = next(
            (idx for idx in indices if idx.get("indexSymbol") == "NIFTY 50"),
            None,
        )
        if nifty_row is None:
            print("[gift_nifty] NIFTY 50 not found in allIndices")
            return dict(_EMPTY)

        last       = float(nifty_row.get("last", 0) or 0)
        prev_close = float(nifty_row.get("previousClose", 0) or 0)

        if last == 0 or prev_close == 0:
            print("[gift_nifty] zero price in allIndices — skipping gap signal")
            return dict(_EMPTY)

        gap_pct = round((last / prev_close - 1) * 100, 2)
        result = {
            "gift_price": round(last, 2),
            "nifty_prev_close": round(prev_close, 2),
            "gap_pct": gap_pct,
            "gap_up_strong":   gap_pct > 1.0,
            "gap_down_strong": gap_pct < -1.0,
            "available": True,
        }
        print(
            f"[gift_nifty] Nifty={result['gift_price']}  prev={result['nifty_prev_close']}"
            f"  gap={result['gap_pct']:+.2f}%"
            f"  gap_up={result['gap_up_strong']}  gap_down={result['gap_down_strong']}"
        )
        return result

    except Exception as exc:
        print(f"[gift_nifty] fetch error: {exc}")
        return dict(_EMPTY)
