"""
NSE options chain reader: Put-Call Ratio (PCR) and OI buildup signals.

PCR interpretation (contrarian extremes only):
  > 1.5 : Extreme fear / put-heavy → contrarian bullish signal (pcr_fear)
  < 0.5 : Extreme complacency / call-heavy → warning signal (pcr_greed)
  0.5-1.5: Normal range — no signal

OI buildup:
  price_up  + OI_up   → long_buildup  (bullish — fresh longs entering)
  price_up  + OI_down → short_covering (bullish — shorts exiting)
  price_down + OI_up  → short_buildup  (bearish — fresh shorts entering)
  price_down + OI_down→ long_unwinding (bearish — longs exiting)

For scoring we use:
  pcr_fear       : PCR > 1.5 (extreme fear = contrarian bullish)     → +2
  long_buildup   : price up + OI up = fresh longs entering            → +1
  short_buildup  : bearish OI signal                                  → −2
  pcr_greed      : PCR < 0.5 (extreme complacency = warning)         → −1

All NSE API calls require a valid cookie session — obtained by hitting the
homepage first. Fails gracefully: all signals default to False on any error.
"""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

_NSE_HOME = "https://www.nseindia.com/"
_OC_EQUITIES = "https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"

_MIN_CALL_OI = 5000  # ignore options signals on illiquid chains (SME / thinly-traded stocks)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": _NSE_HOME,
}

_EMPTY = {
    "pcr": None,
    "pcr_fear": False,    # PCR > 1.5 — extreme fear = contrarian bullish
    "pcr_greed": False,   # PCR < 0.5 — extreme complacency = warning
    "long_buildup": False,
    "short_buildup": False,
    "short_covering": False,
    "long_unwinding": False,
    "total_put_oi": 0,
    "total_call_oi": 0,
}


def _make_session() -> requests.Session:
    """Create an NSE-cookie-initialised session."""
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get(_NSE_HOME, timeout=10)
        time.sleep(0.3)
        # Second hit to consolidate cookies (NSE often requires this)
        s.get("https://www.nseindia.com/market-data/equity-derivatives-watch", timeout=10)
        time.sleep(0.3)
    except Exception:
        pass
    return s


def _parse_option_chain(data: dict, prev_close: float | None) -> dict:
    """
    Parse NSE option-chain response and return signal dict.
    prev_close is used for OI buildup direction (today close vs yesterday close).
    """
    records = data.get("records", {}).get("data", [])
    if not records:
        return dict(_EMPTY)

    total_put_oi  = sum(r.get("PE", {}).get("openInterest", 0) for r in records if "PE" in r)
    total_call_oi = sum(r.get("CE", {}).get("openInterest", 0) for r in records if "CE" in r)
    total_put_chg = sum(r.get("PE", {}).get("changeinOpenInterest", 0) for r in records if "PE" in r)
    total_call_chg = sum(r.get("CE", {}).get("changeinOpenInterest", 0) for r in records if "CE" in r)

    pcr = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else None

    # Underlying close vs prev close
    ul = data.get("records", {}).get("underlyingValue") or 0
    # We rely on caller to pass prev_close; fallback to OI change direction only
    price_up = (prev_close is not None and prev_close > 0 and ul > prev_close)
    price_dn = (prev_close is not None and prev_close > 0 and ul < prev_close)

    # OI expanding or contracting
    net_oi_chg = total_call_chg + total_put_chg
    oi_up = net_oi_chg > 0

    long_buildup   = bool(price_up and oi_up)
    short_covering = bool(price_up and not oi_up)
    short_buildup  = bool(price_dn and oi_up)
    long_unwinding = bool(price_dn and not oi_up)

    # Gate all signals on minimum liquidity — illiquid option chains produce noise
    liquid = total_call_oi >= _MIN_CALL_OI

    # Contrarian extremes only; normal PCR range (0.5-1.5) carries no signal
    pcr_fear  = bool(liquid and pcr is not None and pcr > 1.5)
    pcr_greed = bool(liquid and pcr is not None and pcr < 0.5)

    # OI signals only meaningful on liquid chains
    long_buildup   = bool(liquid and long_buildup)
    short_buildup  = bool(liquid and short_buildup)
    short_covering = bool(liquid and short_covering)
    long_unwinding = bool(liquid and long_unwinding)

    return {
        "pcr": pcr,
        "pcr_fear": pcr_fear,
        "pcr_greed": pcr_greed,
        "long_buildup": long_buildup,
        "short_buildup": short_buildup,
        "short_covering": short_covering,
        "long_unwinding": long_unwinding,
        "total_put_oi": total_put_oi,
        "total_call_oi": total_call_oi,
    }


def _fetch_one(ticker: str, session: requests.Session, prev_close: float | None) -> tuple[str, dict]:
    symbol = ticker.replace(".NS", "").replace("[PENNY]", "")
    try:
        url = _OC_EQUITIES.format(symbol=symbol)
        resp = session.get(url, timeout=15)
        if resp.status_code == 403:
            # Session expired — return empty (caller handles retry)
            return ticker, dict(_EMPTY)
        resp.raise_for_status()
        data = resp.json()
        return ticker, _parse_option_chain(data, prev_close)
    except Exception as exc:
        print(f"[options] {ticker}: {exc}")
        return ticker, dict(_EMPTY)


def fetch_options_signals(
    tickers: list[str],
    prev_closes: dict[str, float] | None = None,
    max_workers: int = 4,
) -> dict[str, dict]:
    """
    Fetch NSE options data for a list of tickers.

    tickers     : list of NSE tickers e.g. ["SBIN.NS", "RELIANCE.NS"]
    prev_closes : {ticker: yesterday_close} — used for OI buildup direction
    Returns     : {ticker: signal_dict}

    Only F&O-eligible stocks have options chains. Non-F&O tickers silently return _EMPTY.
    """
    if not tickers:
        return {}

    prev_closes = prev_closes or {}
    session = _make_session()
    results: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_one, t, session, prev_closes.get(t)): t
            for t in tickers
        }
        for fut in as_completed(futures):
            ticker, sig = fut.result()
            results[ticker] = sig
            if sig.get("pcr") is not None:
                print(f"[options] {ticker}: PCR={sig['pcr']} "
                      f"long_buildup={sig['long_buildup']} pcr_fear={sig['pcr_fear']} pcr_greed={sig['pcr_greed']}")

    no_data = sum(1 for v in results.values() if v["pcr"] is None)
    print(f"[options] {len(tickers)} tickers, {len(tickers) - no_data} with options data, "
          f"{no_data} no data (non-F&O or fetch error)")
    return results
