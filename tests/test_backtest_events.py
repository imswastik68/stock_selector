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


# ── SAST event collection ─────────────────────────────────────────────────────

def test_collect_sast_events_records_a_hit_without_crashing(tmp_path, monkeypatch):
    """Regression: a real SAST hit crashed collect_sast_events with
    AttributeError ('datetime.date' object has no attribute 'date') --
    chunk_start/chunk_end are already `date` objects (from date.today()
    arithmetic), so their midpoint is also a `date`, not a `datetime`; calling
    .date() on it fails. Found by actually running this against the real
    pnsea API, not caught by any test beforehand."""
    sast_cache = tmp_path / "sast_events"
    monkeypatch.setattr(be, "SAST_CACHE", sast_cache)
    monkeypatch.setattr(be, "_pnsea_available", lambda: True)
    monkeypatch.setattr(be, "_sast_one", lambda symbol, from_dt, to_dt: (symbol, True))

    events, reason = be.collect_sast_events(weeks=4, universe=["FOO.NS"])

    assert reason is None
    assert len(events) >= 1
    d, ticker = events[0]
    assert ticker == "FOO.NS"
    assert isinstance(d, date)

    cache_file = sast_cache / "FOO.json"
    assert cache_file.exists()
    cached = json.loads(cache_file.read_text())
    assert len(cached["records"]) >= 1


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


# ── options (F&O bhavcopy) reconstruction ────────────────────────────────────

def _write_fo_slim(monkeypatch, tmp_path, day_rows: dict):
    """day_rows: {date: [{symbol, call_oi, put_oi, net_oi_chg, ul_price}, ...]}"""
    import pandas as pd
    fo_dir = tmp_path / "fo_bhavcopy"
    fo_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(be, "FO_CACHE", fo_dir)
    for d, rows in day_rows.items():
        with gzip.open(be._fo_path(d), "wt") as f:
            pd.DataFrame(rows).to_csv(f, index=False)


def test_options_pcr_fear_fires_above_1_5_and_respects_liquidity_gate(tmp_path, monkeypatch):
    d0, d1 = date(2026, 1, 1), date(2026, 1, 2)
    _write_fo_slim(monkeypatch, tmp_path, {
        d0: [{"symbol": "FEAR", "call_oi": 100000, "put_oi": 200000, "net_oi_chg": 0, "ul_price": 100.0},
             {"symbol": "ILLIQ", "call_oi": 100, "put_oi": 500, "net_oi_chg": 0, "ul_price": 50.0}],
        d1: [{"symbol": "FEAR", "call_oi": 100000, "put_oi": 200000, "net_oi_chg": 0, "ul_price": 100.0},
             {"symbol": "ILLIQ", "call_oi": 100, "put_oi": 500, "net_oi_chg": 0, "ul_price": 50.0}],
    })
    monkeypatch.setattr(be, "MIN_BHAV_DAYS", 2)
    with mock.patch.object(be, "trading_days", return_value=[d0, d1]):
        out, reason = be.collect_options_events(weeks=1, skip_download=True)
    assert reason is None
    fear = set(out["options_pcr_fear"])
    # FEAR: pcr=2.0>1.5 AND call_oi 100000 >= _MIN_CALL_OI(5000) -> fires on d1
    assert (d1, "FEAR.NS") in fear
    # ILLIQ: pcr=5.0 but call_oi 100 < gate -> never fires
    assert not any(t == "ILLIQ.NS" for _, t in fear)


def test_options_long_buildup_needs_price_up_and_oi_up(tmp_path, monkeypatch):
    d0, d1 = date(2026, 1, 1), date(2026, 1, 2)
    _write_fo_slim(monkeypatch, tmp_path, {
        d0: [{"symbol": "BULL", "call_oi": 100000, "put_oi": 50000, "net_oi_chg": 0, "ul_price": 100.0}],
        # d1: price up (110>100) AND net OI up (+5000) -> long_buildup
        d1: [{"symbol": "BULL", "call_oi": 100000, "put_oi": 50000, "net_oi_chg": 5000, "ul_price": 110.0}],
    })
    monkeypatch.setattr(be, "MIN_BHAV_DAYS", 2)
    with mock.patch.object(be, "trading_days", return_value=[d0, d1]):
        out, reason = be.collect_options_events(weeks=1, skip_download=True)
    assert (d1, "BULL.NS") in set(out["options_long_buildup"])
    # not short_covering (that needs OI DOWN)
    assert (d1, "BULL.NS") not in set(out["options_short_covering"])


# ── announcements classification (uses the live _ann_match_signal) ───────────

def test_announcement_classification_matches_live_keyword_map():
    # buyback keyword -> buyback_announced (live map, imported)
    assert be._ann_match_signal("Buy-back of equity shares", "") == "buyback_announced"
    assert be._ann_match_signal("Financial Results for Q1", "") == "results_beat_announced"
    assert be._ann_match_signal("Receipt of order worth 500cr", "") == "contract_win"
    assert be._ann_match_signal("Board meeting intimation", "") is None


# ── promoter quarter-over-quarter increase ───────────────────────────────────

def test_promoter_event_only_on_qoq_increase(tmp_path, monkeypatch):
    import pandas as pd
    shp_dir = tmp_path / "shareholding_hist"
    shp_dir.mkdir()
    monkeypatch.setattr(be, "SHP_CACHE", shp_dir)
    # FOO: promoter rises 50 -> 52 (increase, event) then 52 -> 51 (decrease, no event)
    (shp_dir / "FOO.json").write_text(json.dumps([
        {"date": "31-Mar-2025", "pr_and_prgrp": "50.0"},
        {"date": "30-Jun-2025", "pr_and_prgrp": "52.0"},
        {"date": "30-Sep-2025", "pr_and_prgrp": "51.0"},
    ]))
    with mock.patch.object(be, "_make_www_session", return_value=None):
        events, reason = be.collect_promoter_events(weeks=520, universe=["FOO.NS"])
    assert reason is None
    dates = {d for d, t in events}
    assert date(2025, 6, 30) in dates       # the increase quarter
    assert date(2025, 9, 30) not in dates   # the decrease quarter


# ── merge: running one source must not wipe another's verdict ────────────────

def test_results_merge_does_not_clobber_prior_verdicts(tmp_path, monkeypatch):
    out_file = tmp_path / "event_backtest.json"
    out_file.write_text(json.dumps({"signals": {
        "delivery_surge": {"verdict": "SHIP", "suggested_weight": 1, "n": 38279, "ret_lift": 0.316},
    }}))
    monkeypatch.setattr(be, "OUT_FILE", out_file)
    existing = be._load_existing_results()
    assert existing["delivery_surge"]["verdict"] == "SHIP"
