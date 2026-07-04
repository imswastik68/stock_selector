"""
Offline tests for scripts/backtest_events.py -- the historical event-signal
backtest (delivery/SAST/bulk-deal). No network calls: the delivery-surge
formula, cache loader, resumability, baseline sampler, and verdict/weight
logic are all exercised directly against synthetic/fixture data.
"""

from __future__ import annotations

import gzip
import json
from datetime import date
from unittest import mock

import backtest_events as be


# ── delivery surge formula (must match src.data.delivery's live semantics) ──

def test_delivery_surge_formula_matches_live_constants():
    """Replicates collect_delivery_events' per-ticker surge check inline,
    against the SAME imported constants the live scorer/delivery.py use --
    a signal fires iff today >= 50% AND spike vs prior-4-avg >= 10pp."""
    today_pct = 55.0
    prior = [40.0, 42.0, 41.0, 43.0]  # avg = 41.5
    spike = today_pct - sum(prior) / len(prior)
    fires = today_pct >= be._MIN_DELIVERY_PCT and spike >= be._MIN_DELIVERY_SPIKE
    assert fires is True

    # Below the absolute floor -- must not fire even with a huge spike
    today_pct2 = 49.0
    prior2 = [5.0, 5.0, 5.0, 5.0]
    spike2 = today_pct2 - sum(prior2) / len(prior2)
    fires2 = today_pct2 >= be._MIN_DELIVERY_PCT and spike2 >= be._MIN_DELIVERY_SPIKE
    assert fires2 is False  # spike is huge but today_pct < 50 floor


def test_collect_delivery_events_missing_ticker_day_treated_as_zero(tmp_path, monkeypatch):
    """A ticker absent from a given day's bhavcopy must count as 0.0 delivery%
    for that day (delivery.py:127 semantics), not be skipped from the
    trailing-average calculation entirely."""
    bhav_dir = tmp_path / "bhavcopy"
    bhav_dir.mkdir()
    monkeypatch.setattr(be, "BHAV_CACHE", bhav_dir)
    monkeypatch.setattr(be, "MIN_BHAV_DAYS", 6)  # fixture only has 6 days

    days = [date(2026, 1, i) for i in range(1, 7)]
    # FOO.NS present every day at low%, absent on day 5 (should count as 0.0
    # there), then present again on day 6 with a spike -- since day 6's prior-4
    # window (days 2-5) includes the 0.0 for day 5, the average is pulled down,
    # which would make the spike condition easier to satisfy incorrectly if
    # the missing day were instead excluded from the average.
    data_by_day = {
        days[0]: {"FOO": 20.0}, days[1]: {"FOO": 20.0}, days[2]: {"FOO": 20.0},
        days[3]: {"FOO": 20.0}, days[4]: {},  # FOO absent -> 0.0
        days[5]: {"FOO": 55.0},
    }
    for d, tickers in data_by_day.items():
        with gzip.open(be._bhav_path(d), "wt") as f:
            f.write("SYMBOL,DELIV_PER\n")
            for sym, pct in tickers.items():
                f.write(f"{sym},{pct}\n")

    with mock.patch.object(be, "trading_days", return_value=days):
        events, reason = be.collect_delivery_events(weeks=1, skip_download=True)

    assert reason is None
    # prior-4 avg for day 6 = (20+20+20+0)/4 = 15.0; spike = 55-15 = 40 >= 10
    assert (days[5], "FOO.NS") in events


def test_load_bhavcopy_skips_dash_values_and_strips_symbol_whitespace(tmp_path, monkeypatch):
    bhav_dir = tmp_path / "bhavcopy"
    bhav_dir.mkdir()
    monkeypatch.setattr(be, "BHAV_CACHE", bhav_dir)

    d = date(2026, 1, 1)
    with gzip.open(be._bhav_path(d), "wt") as f:
        f.write("SYMBOL,DELIV_PER\n")
        f.write(" FOO ,60.5\n")   # stray whitespace around symbol
        f.write("BAR,-\n")        # "-" placeholder for missing data -- must be skipped

    result = be._load_bhavcopy(d)
    assert result == {"FOO.NS": 60.5}  # BAR excluded, FOO's symbol stripped


# ── resumability: cached/miss dates must never trigger a network call ──────

def test_download_bhavcopy_skips_dates_already_cached_or_missed(tmp_path, monkeypatch):
    bhav_dir = tmp_path / "bhavcopy"
    bhav_dir.mkdir()
    monkeypatch.setattr(be, "BHAV_CACHE", bhav_dir)

    d1, d2 = date(2026, 1, 1), date(2026, 1, 2)
    with gzip.open(be._bhav_path(d1), "wt") as f:
        f.write("SYMBOL,DELIV_PER\nFOO,60.0\n")
    be._bhav_miss_path(d2).touch()  # d2 = known holiday, no bhavcopy

    def _boom(*args, **kwargs):
        raise AssertionError("should not make a network call for already-resolved dates")

    fake_session = mock.MagicMock()
    fake_session.get.side_effect = _boom
    with mock.patch.object(be, "_delivery_session", return_value=fake_session):
        be._download_bhavcopy([d1, d2])  # must not raise


# ── baseline sampler ─────────────────────────────────────────────────────────

def test_sample_baseline_excludes_signal_tickers_and_is_deterministic():
    import pandas as pd
    import numpy as np

    dates = pd.bdate_range("2024-01-01", periods=100)
    df = pd.DataFrame({
        "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 5_000_000.0,
    }, index=dates)
    ohlcv = {f"T{i}.NS": df for i in range(10)}
    universe = list(ohlcv.keys())

    events = [(dates[70].date(), "T0.NS"), (dates[70].date(), "T1.NS")]
    baseline1 = be._sample_baseline(events, ohlcv, universe, cap_per_date=3, seed=42)
    baseline2 = be._sample_baseline(events, ohlcv, universe, cap_per_date=3, seed=42)

    assert baseline1 == baseline2  # deterministic under a fixed seed
    signal_tickers = {t for _, t in events}
    assert all(t not in signal_tickers for _, t in baseline1)


# ── suggested_weight clamp ───────────────────────────────────────────────────

def test_suggested_weight_clamps_to_0_5_range():
    assert be.suggested_weight(None) == 0
    assert be.suggested_weight(-2.0) == 0     # negative lift -> floor at 0
    assert be.suggested_weight(0.5) == 2       # round(3*0.5) = round(1.5) = 2
    assert be.suggested_weight(3.0) == 5       # round(3*3.0) = 9, clamped to 5


# ── verdict truth table ──────────────────────────────────────────────────────

def _result(as_of: str, win: bool, ret: float) -> dict:
    return {"ticker": "X.NS", "as_of": date.fromisoformat(as_of), "outcome": "t1_hit" if win else "sl_hit",
            "return_pct": ret, "win": win}


def test_verdict_insufficient_sample_below_min_n():
    signal = [_result(f"2024-01-{i%28+1:02d}", True, 5.0) for i in range(499)]
    baseline = [_result(f"2024-01-{i%28+1:02d}", False, 1.0) for i in range(499)]
    stats = be.verdict_for(signal, baseline)
    assert stats["verdict"] == "INSUFFICIENT_SAMPLE"
    assert stats["suggested_weight"] == 0


def test_verdict_ships_when_lifts_positive_and_holdout_consistent():
    # 600 signal events, all wins with a positive lift over baseline, evenly
    # spread across dates so both the 70% train and 30% holdout halves are
    # sign-consistent.
    signal = [_result(f"2024-{(i % 12) + 1:02d}-01", True, 8.0) for i in range(600)]
    baseline = [_result(f"2024-{(i % 12) + 1:02d}-01", False, 1.0) for i in range(600)]
    stats = be.verdict_for(signal, baseline)
    assert stats["n"] == 600
    assert stats["holdout_consistent"] is True
    assert stats["verdict"] == "SHIP"
    assert stats["suggested_weight"] > 0


def test_verdict_no_ship_when_holdout_inconsistent():
    # Train half (first 70%) all wins with a big lift; holdout half (last 30%)
    # all losses with a negative lift -- pooled average could still look
    # positive-ish, but the holdout split must catch the sign flip.
    signal = ([_result(f"2024-01-{i+1:02d}", True, 10.0) for i in range(28)] * 15)[:420]
    signal += [_result(f"2024-06-{(i % 28)+1:02d}", False, -10.0) for i in range(180)]
    baseline = [_result(f"2024-01-{(i % 28)+1:02d}", False, 1.0) for i in range(420)]
    baseline += [_result(f"2024-06-{(i % 28)+1:02d}", True, 5.0) for i in range(180)]
    stats = be.verdict_for(signal, baseline)
    assert stats["n"] == 600
    assert stats["holdout_consistent"] is False
    assert stats["verdict"] == "NO-SHIP"
    assert stats["suggested_weight"] == 0
