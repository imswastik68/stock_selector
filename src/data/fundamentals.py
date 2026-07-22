"""
Fundamental quality gate — soft score adjustment using yfinance .info.

Signals derived per ticker (top-20 only; missing data → neutral):
  fundamental_strong: ROE>15% AND D/E<1 AND EPS-growth>0  → +2 points
  fundamental_weak:   net loss (trailingEps<0) OR D/E>3   → -2 points
  (both False = neutral, never hard-drops)

Results cached in outputs/fundamentals_cache.json with 7-day TTL
(.info calls are slow and fundamentals rarely change day-to-day).

Usage:
  from src.data.fundamentals import fetch_fundamental_signals
  signals = fetch_fundamental_signals(["RELIANCE.NS", "TCS.NS"])
  # {"RELIANCE.NS": {"fundamental_strong": True, "fundamental_weak": False,
  #                   "roe": 0.18, "de_ratio": 0.5, "eps_growth": 0.12}}
"""

from __future__ import annotations

import json
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

_CACHE_FILE = Path(__file__).parent.parent.parent / "outputs" / "fundamentals_cache.json"
_CACHE_TTL_DAYS = 7
_FETCH_DELAY_SECONDS = 1.2  # spacing between .info calls; see the fetch loop


def _load_cache() -> dict:
    if _CACHE_FILE.exists():
        try:
            return json.loads(_CACHE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_cache(data: dict) -> None:
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(data, indent=2, default=str))


def _is_fresh(entry: dict) -> bool:
    fetched = entry.get("fetched_date")
    if not fetched:
        return False
    try:
        return (date.today() - date.fromisoformat(fetched)).days < _CACHE_TTL_DAYS
    except Exception:
        return False


def _derive_signals(info: dict) -> dict:
    """Derive fundamental_strong / fundamental_weak from yfinance .info dict."""
    roe        = info.get("returnOnEquity")         # decimal, e.g. 0.18
    de_ratio   = info.get("debtToEquity")            # percentage, e.g. 36.2 = 0.362x
    eps_growth = info.get("earningsQuarterlyGrowth") # decimal, e.g. 0.12
    eps        = info.get("trailingEps")             # absolute earnings
    margins    = info.get("profitMargins")

    # Normalize D/E: yfinance's debtToEquity is always a percentage (Yahoo "Total
    # Debt/Equity (mrq)", e.g. 36.2 meaning a true ratio of 0.362x), never a raw
    # ratio. A conditional >10 check left true D/E 0.03-0.10 (reported as 3-10)
    # unscaled, misclassifying low-debt companies as fundamental_weak (confirmed
    # live: TATAELXSI.NS reported D/E=5.3, true D/E ~0.05).
    de_normalized = None
    if de_ratio is not None:
        de_normalized = de_ratio / 100

    strong = False
    weak   = False

    if roe is not None and de_normalized is not None:
        if roe > 0.15 and de_normalized < 1.0 and (eps_growth is None or eps_growth > 0):
            strong = True
        elif (eps is not None and eps < 0) or (de_normalized is not None and de_normalized > 3.0):
            weak = True
    elif eps is not None and eps < 0:
        weak = True

    return {
        "fundamental_strong": strong,
        "fundamental_weak":   weak,
        "roe":        round(float(roe), 4) if roe is not None else None,
        "de_ratio":   round(float(de_normalized), 3) if de_normalized is not None else None,
        "eps_growth": round(float(eps_growth), 4) if eps_growth is not None else None,
        "profit_margins": round(float(margins), 4) if margins is not None else None,
    }


def fetch_fundamental_signals(tickers: list[str]) -> dict[str, dict]:
    """
    Fetch fundamental signals for a list of tickers.
    Uses 7-day cache; missing/failed data → neutral (both signals False).
    Returns {ticker: {fundamental_strong, fundamental_weak, roe, de_ratio, ...}}
    """
    try:
        import yfinance as yf
    except ImportError:
        print("[fundamentals] yfinance not installed — fundamentals disabled")
        return {}

    cache = _load_cache()
    today = date.today().isoformat()
    results: dict[str, dict] = {}
    to_fetch: list[str] = []

    for t in tickers:
        cached = cache.get(t)
        if cached and _is_fresh(cached):
            results[t] = {k: v for k, v in cached.items() if k != "fetched_date"}
        else:
            to_fetch.append(t)

    if to_fetch:
        print(f"[fundamentals] fetching .info for {len(to_fetch)} tickers ...")
        n_failed = 0
        for i, t in enumerate(to_fetch):
            # yfinance throttles .info hard (HTTP 429 "Too Many Requests") when
            # called in a tight loop -- a bare 20-ticker loop reliably fails ALL
            # 20. Space the calls out; _FETCH_DELAY_SECONDS x 20 is only ~24s
            # against a scan that already takes minutes.
            if i:
                time.sleep(_FETCH_DELAY_SECONDS)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    info = yf.Ticker(t).info
                sig = _derive_signals(info)
                cache[t] = {**sig, "fetched_date": today}
                results[t] = sig
                label = "strong" if sig["fundamental_strong"] else ("weak" if sig["fundamental_weak"] else "neutral")
                roe_str = f" ROE={sig['roe']:.1%}" if sig['roe'] is not None else ""
                de_str  = f" D/E={sig['de_ratio']:.1f}" if sig['de_ratio'] is not None else ""
                print(f"[fundamentals] {t:<20} {label}{roe_str}{de_str}")
            except Exception as exc:
                n_failed += 1
                print(f"[fundamentals] {t}: error ({exc}) — neutral")
                # Neutral for THIS run only -- deliberately NOT written to cache.
                # Caching a fetch failure would stamp it with today's date and
                # the 7-day TTL would then serve "no fundamental data" for a week
                # without ever retrying, silently disabling fundamental_strong /
                # fundamental_weak on exactly the tickers that failed. A failure
                # is an absence of data, not a measurement of it.
                results[t] = {"fundamental_strong": False, "fundamental_weak": False,
                              "roe": None, "de_ratio": None, "eps_growth": None, "profit_margins": None}

        if n_failed:
            print(f"[fundamentals] {n_failed}/{len(to_fetch)} failed — not cached, will retry next run")
        _save_cache(cache)

    return results
