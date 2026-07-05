"""
Tests for src/data/bse_announcements.classify_pead_reaction -- the live PEAD v2
surprise classifier (Alpha Round Phase 2/5). Mirrors the exact rule backtested
in scripts/backtest_events.py's collect_pead_events: reaction day R = filing
day if filed by 15:30 IST else the next available bar; r_R = close_R/close_
{R-1}-1; >=+3% positive, <=-3% negative, else no classification.
"""

from __future__ import annotations

from datetime import date
from unittest import mock

import pandas as pd

import backtest_events as be
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


# ── permanent backtest<->live parity (Phase 0b) ──────────────────────────────
# The backtest (collect_pead_events) and the live path (classify_pead_reaction)
# use DIFFERENT mechanisms to find the reaction day -- the backtest walks a
# cached Nifty trading-day calendar, live walks the ticker's own OHLCV index.
# They must never silently diverge. This was verified ad-hoc once; pinning it
# here makes it a permanent regression, with all three classification outcomes.

def _shared_day_set():
    return [date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]


def _both_classify(symbol: str, an_dt: str, filed_at_iso: str, closes: dict):
    """Run the SAME (filing, OHLCV) input through both the backtest engine and
    the live classifier; return (backtest_signal_or_None, live_classification)."""
    day_set = _shared_day_set()
    df = _df({d.isoformat(): c for d, c in closes.items()})
    item = {"symbol": symbol, "desc": "Financial Results Q3", "attchmntText": "", "an_dt": an_dt}

    with mock.patch.object(be, "_fetch_announcement_items", return_value=([item], None)), \
         mock.patch.object(be, "trading_days", return_value=day_set):
        out, reason = be.collect_pead_events(156, [f"{symbol}.NS"], {f"{symbol}.NS": df})
    assert reason is None

    bt_signal = None
    for sig in ("pead_positive_surprise", "pead_negative_surprise"):
        if any(t == f"{symbol}.NS" for _, t in out[sig]):
            bt_signal = sig

    live_cls = classify_pead_reaction(filed_at_iso, df)
    return bt_signal, live_cls


def test_parity_positive_reaction_pre_cutoff():
    bt_signal, live_cls = _both_classify(
        "PARX", "05-Jan-2026 10:00:00", "2026-01-05T10:00:00+05:30",
        {date(2026, 1, 2): 100.0, date(2026, 1, 5): 104.0, date(2026, 1, 6): 104.0, date(2026, 1, 7): 104.0},
    )
    assert bt_signal == "pead_positive_surprise"
    assert live_cls == "positive"


def test_parity_negative_reaction_post_cutoff():
    bt_signal, live_cls = _both_classify(
        "PARY", "05-Jan-2026 16:00:00", "2026-01-05T16:00:00+05:30",
        {date(2026, 1, 2): 100.0, date(2026, 1, 5): 100.0, date(2026, 1, 6): 94.0, date(2026, 1, 7): 94.0},
    )
    assert bt_signal == "pead_negative_surprise"
    assert live_cls == "negative"


def test_parity_middle_band_no_signal_either_side():
    bt_signal, live_cls = _both_classify(
        "PARZ", "05-Jan-2026 10:00:00", "2026-01-05T10:00:00+05:30",
        {date(2026, 1, 2): 100.0, date(2026, 1, 5): 101.0, date(2026, 1, 6): 101.0, date(2026, 1, 7): 101.0},
    )
    assert bt_signal is None
    assert live_cls is None


# ── fetch_pead_signals: batch OHLCV fetch + classification (SOTA Round Phase 1) ─

import src.data.bse_announcements as ba


def test_fetch_pead_signals_empty_announcements_returns_empty():
    assert ba.fetch_pead_signals([]) == {}


def test_fetch_pead_signals_ignores_non_results_announcements():
    anns = [{"ticker": "FOO.NS", "signal_key": "buyback_announced",
             "an_dt": "05-Jan-2026 10:00:00", "filed_at": "2026-01-05T10:00:00+05:30"}]
    assert ba.fetch_pead_signals(anns) == {}


def test_fetch_pead_signals_multi_ticker_multiindex(monkeypatch):
    idx = pd.to_datetime(["2026-01-02", "2026-01-05"])
    # MultiIndex columns (ticker, field) -- mirrors yf.download(group_by="ticker")
    cols = pd.MultiIndex.from_product([["FOO.NS", "BAR.NS"], ["Close"]])
    raw = pd.DataFrame([[100.0, 100.0], [104.0, 94.0]], index=idx, columns=cols)

    monkeypatch.setattr(ba.yf, "download", lambda *a, **k: raw)
    anns = [
        {"ticker": "FOO.NS", "signal_key": "results_beat_announced",
         "filed_at": "2026-01-05T10:00:00+05:30"},
        {"ticker": "BAR.NS", "signal_key": "results_beat_announced",
         "filed_at": "2026-01-05T10:00:00+05:30"},
    ]
    out = ba.fetch_pead_signals(anns)
    assert out == {"FOO.NS": "positive", "BAR.NS": "negative"}


def test_fetch_pead_signals_single_ticker_flat_columns(monkeypatch):
    idx = pd.to_datetime(["2026-01-02", "2026-01-05"])
    raw = pd.DataFrame({"Close": [100.0, 104.0]}, index=idx)

    monkeypatch.setattr(ba.yf, "download", lambda *a, **k: raw)
    anns = [{"ticker": "FOO.NS", "signal_key": "results_beat_announced",
              "filed_at": "2026-01-05T10:00:00+05:30"}]
    out = ba.fetch_pead_signals(anns)
    assert out == {"FOO.NS": "positive"}


def test_fetch_pead_signals_download_error_returns_empty(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(ba.yf, "download", _boom)
    anns = [{"ticker": "FOO.NS", "signal_key": "results_beat_announced",
              "filed_at": "2026-01-05T10:00:00+05:30"}]
    assert ba.fetch_pead_signals(anns) == {}


def test_fetch_pead_signals_empty_download_returns_empty(monkeypatch):
    monkeypatch.setattr(ba.yf, "download", lambda *a, **k: pd.DataFrame())
    anns = [{"ticker": "FOO.NS", "signal_key": "results_beat_announced",
              "filed_at": "2026-01-05T10:00:00+05:30"}]
    assert ba.fetch_pead_signals(anns) == {}
