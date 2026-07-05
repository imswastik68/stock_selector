"""
Tests for src/data/bse_announcements.classify_pead_reaction -- the live PEAD v2
surprise classifier (Alpha Round Phase 2/5). Mirrors the exact rule backtested
in scripts/backtest_events.py's collect_pead_events: reaction day R = filing
day if filed by 15:30 IST else the next available bar; r_R = close_R/close_
{R-1}-1; >=+3% positive, <=-3% negative, else no classification.
"""

from __future__ import annotations

import pandas as pd

from src.data.bse_announcements import classify_pead_reaction


def _df(date_close: dict) -> pd.DataFrame:
    days = sorted(date_close.keys())
    closes = [date_close[d] for d in days]
    return pd.DataFrame({"Close": closes}, index=pd.to_datetime(days))


def test_pre_cutoff_filing_reacts_same_day_positive():
    df = _df({"2026-01-02": 100.0, "2026-01-05": 104.0})
    assert classify_pead_reaction("2026-01-05T10:00:00+05:30", df) == "positive"


def test_after_cutoff_filing_reacts_next_trading_day_negative():
    df = _df({"2026-01-05": 100.0, "2026-01-06": 94.0})
    assert classify_pead_reaction("2026-01-05T16:00:00+05:30", df) == "negative"


def test_middle_band_reaction_is_none():
    df = _df({"2026-01-02": 100.0, "2026-01-05": 101.0})
    assert classify_pead_reaction("2026-01-05T10:00:00+05:30", df) is None


def test_no_prior_bar_returns_none():
    """Filing on the very first cached bar has no R-1 to react against."""
    df = _df({"2026-01-02": 100.0})
    assert classify_pead_reaction("2026-01-02T10:00:00+05:30", df) is None


def test_empty_or_missing_df_returns_none():
    assert classify_pead_reaction("2026-01-05T10:00:00+05:30", None) is None
    assert classify_pead_reaction("2026-01-05T10:00:00+05:30", pd.DataFrame()) is None


def test_unparseable_filed_at_returns_none():
    df = _df({"2026-01-02": 100.0, "2026-01-05": 104.0})
    assert classify_pead_reaction("not-a-date", df) is None


def test_no_future_bar_available_returns_none():
    """Filed after cutoff but the OHLCV cache has no bar after the filing date
    (e.g. a stale/short-lived cache) -- must not crash or misfire."""
    df = _df({"2026-01-05": 100.0})
    assert classify_pead_reaction("2026-01-05T16:00:00+05:30", df) is None
