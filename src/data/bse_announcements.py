"""
NSE corporate announcement fetcher — catches post-3:30 PM filings before market open.

Uses NSE's public corporate-announcements API (no auth required).
Targets: quarterly results, order/contract wins, buybacks, dividends.

Returns a flat list of {ticker, signal_key, headline, filed_at} dicts.
score_candidates() converts this to a per-ticker lookup for signal scoring.
"""

from __future__ import annotations

import re
import requests
from datetime import datetime, timedelta, timezone

_NSE_URL = "https://www.nseindia.com/api/corporate-announcements?index=equities"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nseindia.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

IST = timezone(timedelta(hours=5, minutes=30))

# Maps announcement description keywords → signal_key
# Checked in order; first match wins.
_KEYWORD_MAP: list[tuple[list[str], str]] = [
    # Buyback: strong promoter confidence signal
    (["buy-back", "buyback", "buy back"], "buyback_announced"),
    # Dividend declaration
    (["interim dividend", "final dividend", "dividend"], "dividend_announced"),
    # Results — "financial results" or "quarterly results" or outcome mentioning results
    (["financial results", "quarterly result", "annual result", "half yearly result"], "results_beat_announced"),
    # Order/contract win — high-impact for mid/small caps
    (
        [
            "award of order", "receipt of order", "new order", "letter of award",
            "letter of intent", "work order", "order win", "secured order",
            "order receipt", "order/win", "contract awarded", "contract secured",
        ],
        "contract_win",
    ),
]


def _parse_nse_dt(dt_str: str) -> datetime | None:
    """Parse '02-Jun-2026 23:49:32' → aware datetime (IST)."""
    try:
        dt = datetime.strptime(dt_str.strip(), "%d-%b-%Y %H:%M:%S")
        return dt.replace(tzinfo=IST)
    except ValueError:
        return None


def _match_signal(desc: str, headline: str) -> str | None:
    text = (desc + " " + headline).lower()
    for keywords, signal_key in _KEYWORD_MAP:
        if any(kw in text for kw in keywords):
            return signal_key
    return None


def fetch_bse_announcements(hours_back: int = 20) -> list[dict]:
    """
    Return NSE corporate filings posted in the last `hours_back` hours.
    Each dict: {ticker (with .NS), signal_key, headline, filed_at (ISO str)}

    Safe to call at any time; returns [] on any failure.
    """
    try:
        resp = requests.get(_NSE_URL, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        items = resp.json()
        if not isinstance(items, list):
            print("[bse_announcements] unexpected API response type")
            return []
    except Exception as exc:
        print(f"[bse_announcements] fetch failed: {exc}")
        return []

    cutoff = datetime.now(IST) - timedelta(hours=hours_back)
    results: list[dict] = []

    for item in items:
        symbol = str(item.get("symbol") or "").strip().upper()
        if not symbol:
            continue

        dt_str = str(item.get("an_dt") or "").strip()
        filed_at = _parse_nse_dt(dt_str)
        if filed_at is None or filed_at < cutoff:
            continue

        desc = str(item.get("desc") or "")
        # attchmntText has a one-line summary from the company filing
        headline = str(item.get("attchmntText") or desc)

        signal_key = _match_signal(desc, headline)
        if signal_key is None:
            continue

        results.append({
            "ticker": f"{symbol}.NS",
            "signal_key": signal_key,
            "headline": headline[:200],
            "filed_at": filed_at.isoformat(),
        })

    print(f"[bse_announcements] {len(results)} actionable announcements in last {hours_back}h")
    return results


# PEAD v2 (Alpha Round Phase 2/5): results_beat_announced fires on ANY results
# filing and backtested net-harmful (ret_lift=-1.234, n=8824 -- see
# outputs/event_backtest.json / scripts/backtest_events.py). The version that
# shipped conditions on the announcement-day price REACTION as a surprise
# proxy. Same pre-declared rule as the backtest, applied live:
#   reaction day R = filing day if filed by 15:30 IST, else the next
#     available trading bar in the ticker's own OHLCV;
#   r_R = close_R / close_{R-1} - 1
#   r_R >= +3%  -> "positive" (pead_positive_surprise)
#   r_R <= -3%  -> "negative" (pead_negative_surprise)
_PEAD_REACTION_CUTOFF = 15 * 60 + 30  # 15:30 IST, in minutes-since-midnight


def classify_pead_reaction(filed_at: str, df) -> str | None:
    """`filed_at` is an ISO datetime string (as produced by
    fetch_bse_announcements's `filed_at`), `df` is the ticker's OHLCV
    DataFrame (datetime index, must include a `Close` column). Returns
    "positive", "negative", or None (middle band / not enough data)."""
    if df is None or df.empty:
        return None
    try:
        filed = datetime.fromisoformat(filed_at)
    except Exception:
        return None
    filed_date = filed.date()
    filed_minutes = filed.hour * 60 + filed.minute

    idx = df.index
    # normalise to plain dates for comparison against filed_date
    dates = [ts.date() if hasattr(ts, "date") else ts for ts in idx]

    if filed_minutes <= _PEAD_REACTION_CUTOFF and filed_date in dates:
        r_pos = dates.index(filed_date)
    else:
        r_pos = next((i for i, d in enumerate(dates) if d > filed_date), None)
    if r_pos is None or r_pos == 0:
        return None

    try:
        close_r = float(df["Close"].iloc[r_pos])
        close_prev = float(df["Close"].iloc[r_pos - 1])
    except Exception:
        return None
    if close_prev <= 0:
        return None

    r_react = close_r / close_prev - 1
    if r_react >= 0.03:
        return "positive"
    if r_react <= -0.03:
        return "negative"
    return None
