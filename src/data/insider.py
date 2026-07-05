"""
Promoter open-market purchase detector (promoter_open_mkt_buy) -- SOTA Round
Phase 3. sast_insider_buying fires on ANY SAST filing (pledges, creeping
acquisitions, inter-se transfers) and backtested net-harmful (ret_lift=
-0.407, n=2010). This isolates promoters buying their OWN stock in the OPEN
MARKET -- a genuine "insiders think it's cheap" signal, only visible via the
PIT (Prohibition of Insider Trading) disclosure feed, not SAST.

Backtested SHIP verdict (scripts/backtest_events.py, outputs/event_backtest.
json, 260-week window): n=617, ret_lift=+1.063 (wr_lift +5.38pp), 70/30
holdout sign-consistent (train +1.152, holdout +0.866).

Filter constants below are the SINGLE SOURCE OF TRUTH -- scripts/
backtest_events.py imports them (never hardcodes its own copy), confirmed
live against the corporates-pit feed (2026-07-05, 1192-row sample):
acqMode has exactly one value meaning a genuine open-market trade, "Market
Purchase" -- distinct from "Off Market" (which also contains the substring
"market"), "ESOP", "Pledge Creation", "Preferential Offer", "Gift", etc.
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
import requests

from src.data.delivery import _make_session

PIT_URL = "https://www.nseindia.com/api/corporates-pit?index=equities&from_date={frm}&to_date={to}"
PIT_PRIME_URL = "https://www.nseindia.com/companies-listing/corporate-filings-insider-trading"

PIT_ACQ_MODE = "Market Purchase"
PIT_PERSON_CATEGORIES = {"Promoters", "Promoter Group"}
PIT_MIN_VALUE = 1e7  # >= Rs 1 crore


def _make_www_session() -> requests.Session:
    session = _make_session()
    try:
        session.get(PIT_PRIME_URL, timeout=15)
        time.sleep(1.0)
    except Exception:
        pass
    return session


def _www_get_json(session: requests.Session, url: str, retries: int = 4):
    for _ in range(retries):
        try:
            r = session.get(url, timeout=25)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(2.5)
    return None


def _matches_filter(row: dict) -> bool:
    if row.get("personCategory") not in PIT_PERSON_CATEGORIES:
        return False
    if row.get("tdpTransactionType") != "Buy":
        return False
    if row.get("acqMode") != PIT_ACQ_MODE:
        return False
    try:
        sec_val = float(str(row.get("secVal")).strip())
    except (ValueError, TypeError):
        return False
    return sec_val >= PIT_MIN_VALUE


def fetch_promoter_open_mkt_buys(lookback_days: int = 2) -> list[dict]:
    """
    Return NIFTY500 (or any listed) tickers with a genuine promoter
    open-market purchase intimated in the last `lookback_days` days.
    Global feed (one call covers every listed company, unlike shareholding-
    master's per-symbol calls). Same-day cache, same pattern as
    src.data.breakouts.fetch_breakouts.
    """
    import os
    from src.cache import load_today, load_latest, save_today
    cached = load_today("promoter_open_mkt")
    if cached is not None:
        return cached
    if os.environ.get("SCAN_MODE") == "pre_market":
        cached = load_latest("promoter_open_mkt")
        if cached is not None:
            return cached

    end = date.today()
    start = end - timedelta(days=lookback_days)
    session = _make_www_session()
    url = PIT_URL.format(frm=start.strftime("%d-%m-%Y"), to=end.strftime("%d-%m-%Y"))
    data = _www_get_json(session, url)
    if data is None:
        print("[insider] corporates-pit API unreachable -- skipping this scan")
        return []

    rows = data if isinstance(data, list) else (data or {}).get("data", data)
    if not isinstance(rows, list):
        return []

    seen: set[tuple] = set()
    results: list[dict] = []
    for row in rows:
        if not isinstance(row, dict) or not _matches_filter(row):
            continue
        sym = str(row.get("symbol") or "").strip().upper()
        if not sym:
            continue
        dt_str = str(row.get("date") or "").strip()
        try:
            d = pd.to_datetime(dt_str[:11], format="%d-%b-%Y").date()
        except Exception:
            continue
        key = (d, sym)
        if key in seen:
            continue
        seen.add(key)
        ticker = f"{sym}.NS"
        results.append({
            "ticker": ticker,
            "intimation_date": d.isoformat(),
            "value_rs": float(str(row.get("secVal")).strip()),
        })

    print(f"[insider] {len(results)} promoter open-market buy(s) found")
    save_today("promoter_open_mkt", results)
    return results
