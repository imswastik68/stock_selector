"""
NSE daily bhav copy — delivery volume signal.

Delivery % (DELIV_PER) measures what fraction of traded volume resulted in actual
delivery vs. intraday square-off. Sustained high delivery across multiple sessions
indicates institutional accumulation rather than speculative activity.

Signal: delivery_surge = DELIV_PER >= 50% on at least 3 of the last 5 trading sessions.
Single-day spikes are excluded — they can be block deal settlement artifacts.
"""

from __future__ import annotations

import io
import time
from datetime import date, timedelta

import pandas as pd
import requests

_NSE_HOME = "https://www.nseindia.com/"
_BHAV_URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date}.csv"

_MIN_DELIVERY_PCT   = 50.0  # absolute floor: below this = mostly retail/intraday
_MIN_DELIVERY_SPIKE = 10.0  # today must exceed own 5-day avg by >= 10pp (fresh entry)
_LOOKBACK_DAYS      = 5     # trading sessions to fetch

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


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get(_NSE_HOME, timeout=10)
        time.sleep(0.3)
    except Exception:
        pass
    return s


def _fetch_one_day(session: requests.Session, d: date) -> dict[str, float] | None:
    """Fetch bhav copy for one date. Returns {SYMBOL.NS: delivery_pct} or None on failure."""
    url = _BHAV_URL.format(date=d.strftime("%d%m%Y"))
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200 or len(resp.content) < 1000:
            return None
        df = pd.read_csv(io.BytesIO(resp.content))
        df.columns = df.columns.str.strip().str.upper()
        if not {"SYMBOL", "SERIES", "DELIV_PER"}.issubset(df.columns):
            return None
        df = df[df["SERIES"].str.strip() == "EQ"].copy()
        result: dict[str, float] = {}
        for _, row in df.iterrows():
            symbol = str(row["SYMBOL"]).strip()
            if not symbol:
                continue
            try:
                result[f"{symbol}.NS"] = float(row["DELIV_PER"])
            except (ValueError, TypeError):
                pass  # "-" or blank — skip
        return result or None
    except Exception:
        return None


def fetch_delivery_signals() -> dict[str, dict]:
    """
    Fetch NSE bhav copy for the last 5 trading sessions and return delivery signals.
    delivery_surge = today's delivery% >= 50% AND today >= own 5-day avg + 10pp.
    Returns {ticker: {"delivery_pct": float, "delivery_surge": bool, "delivery_spike_pp": float}}.
    Empty dict on complete failure — callers handle gracefully.
    Results are cached to disk for the calendar day.
    """
    import os
    from src.cache import load_today, load_latest, save_today
    cached = load_today("delivery")
    if cached is not None:
        return cached
    if os.environ.get("SCAN_MODE") == "pre_market":
        cached = load_latest("delivery")
        if cached is not None:
            return cached  # pre-market: reuse yesterday's data, no new download

    session = _make_session()

    # Walk back calendar days, collecting up to _LOOKBACK_DAYS trading sessions.
    # A trading session is any day where the bhav copy exists and has data.
    daily_maps: list[dict[str, float]] = []
    cal_day = date.today()

    for _ in range(20):  # scan at most 20 calendar days back
        day_data = _fetch_one_day(session, cal_day)
        if day_data is not None:
            daily_maps.append(day_data)
        cal_day -= timedelta(days=1)
        if len(daily_maps) == _LOOKBACK_DAYS:
            break

    if not daily_maps:
        print("[delivery] bhav copy unavailable — delivery_surge signal disabled for this run")
        return {}

    sessions_fetched = len(daily_maps)
    print(f"[delivery] fetched {sessions_fetched} trading sessions for delivery analysis")

    # Aggregate across sessions: count days >= threshold per ticker
    all_tickers: set[str] = set()
    for day in daily_maps:
        all_tickers.update(day.keys())

    results: dict[str, dict] = {}
    for ticker in all_tickers:
        today_pct = daily_maps[0].get(ticker, 0.0)
        # 5-day avg excludes today (days 1-4) to measure spike vs recent baseline
        prior_pcts = [day.get(ticker, 0.0) for day in daily_maps[1:]]
        avg_prior   = sum(prior_pcts) / len(prior_pcts) if prior_pcts else today_pct
        spike_pp    = today_pct - avg_prior
        # Both conditions required: absolute institutional floor + relative spike vs own baseline
        delivery_surge = (today_pct >= _MIN_DELIVERY_PCT and spike_pp >= _MIN_DELIVERY_SPIKE)
        results[ticker] = {
            "delivery_pct":      round(today_pct, 2),
            "delivery_surge":    delivery_surge,
            "delivery_spike_pp": round(spike_pp, 1),
        }

    surge_count = sum(1 for v in results.values() if v["delivery_surge"])
    print(f"[delivery] {len(results)} equities — {surge_count} with delivery_surge "
          f"(>={_MIN_DELIVERY_PCT}% today AND spike >={_MIN_DELIVERY_SPIKE}pp above own 5d avg)")
    save_today("delivery", results)
    return results
