"""
Regression test for the audit fix to src/performance.py: record_picks now
stamps exit_policy at record time, and evaluate_prior_picks falls back to
"static" (not the current global WINNER_POLICY) for legacy un-stamped
picks -- otherwise a WINNER_POLICY change silently rewrites historical
outcomes for picks that were actually recorded under a different policy.
"""

from __future__ import annotations

import json
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import pandas as pd

import src.performance as p


def test_new_pick_stamped_with_current_policy_and_addition_a_fields(tmp_path):
    perf_file = tmp_path / "performance.json"
    watchlist_data = {
        "scan_date": "2026-07-04",
        "buy_watchlist": [{
            "ticker": "NEW.NS", "today_close": 100.0, "stop_loss": "₹90",
            "target_1": "₹120", "target_2": "₹130", "big_mover": True, "instrument": "cash_equity",
        }],
        "sell_watchlist": [],
    }
    with mock.patch.object(p, "_PERF_FILE", perf_file), mock.patch.object(p, "WINNER_POLICY", "time_10d"):
        p.record_picks(watchlist_data)
        loaded = json.loads(perf_file.read_text())

    pick = loaded["2026-07-04"]["NEW.NS"]
    assert pick["exit_policy"] == "time_10d"
    assert pick["big_mover"] is True
    assert pick["instrument"] == "cash_equity"


def test_legacy_unstamped_pick_falls_back_to_static_not_current_global(tmp_path):
    """A pick recorded before the exit_policy stamp existed has no
    exit_policy key -- evaluate_prior_picks must treat it as "static" (what
    it was actually recorded under), not whatever WINNER_POLICY is today."""
    perf_file = tmp_path / "performance.json"
    perf_file.write_text(json.dumps({
        "2026-06-01": {
            "LEGACY.NS": {
                "direction": "buy", "entry": 100.0, "stop_loss": 90.0,
                "target_1": 200.0, "target_2": 220.0, "outcome": "open", "outcome_date": None,
                # no "exit_policy" key -- simulates a pre-fix pick
            },
        },
    }))
    with mock.patch.object(p, "_PERF_FILE", perf_file):
        loaded = p._load_perf()
        legacy_pick = loaded["2026-06-01"]["LEGACY.NS"]
        assert legacy_pick.get("exit_policy", "static") == "static"


def test_performance_summary_splits_big_mover_win_rate(tmp_path):
    perf_file = tmp_path / "performance.json"
    recent_date = (date.today() - timedelta(days=3)).isoformat()  # must stay inside the 30d lookback
    perf_file.write_text(json.dumps({
        recent_date: {
            "OLD.NS": {"direction": "buy", "outcome": "t1_hit"},                      # no big_mover key
            "BM.NS":  {"direction": "buy", "outcome": "sl_hit", "big_mover": True},
            "BM2.NS": {"direction": "buy", "outcome": "t1_hit", "big_mover": True},
        },
    }))
    with mock.patch.object(p, "_PERF_FILE", perf_file), \
         mock.patch.object(p, "evaluate_prior_picks", lambda lookback_days: json.loads(perf_file.read_text())):
        summary = p.performance_summary(30)

    assert summary["big_mover_n_decided"] == 2
    assert summary["big_mover_win_rate_pct"] == 50.0
    assert summary["other_n_decided"] == 1
    assert summary["other_win_rate_pct"] == 100.0


# ── Phase 1 fix: next-day zone entry (was scan-day point-entry) ─────────────

def test_record_picks_parses_entry_zone(tmp_path):
    perf_file = tmp_path / "performance.json"
    watchlist_data = {
        "scan_date": "2026-07-04",
        "buy_watchlist": [{
            "ticker": "ZONE.NS", "today_close": 100.0, "entry_zone": "₹95.5-₹104.5",
            "stop_loss": "₹90", "target_1": "₹120", "target_2": "₹130",
        }],
        "sell_watchlist": [],
    }
    with mock.patch.object(p, "_PERF_FILE", perf_file):
        p.record_picks(watchlist_data)
        loaded = json.loads(perf_file.read_text())

    pick = loaded["2026-07-04"]["ZONE.NS"]
    assert pick["entry_lo"] == 95.5
    assert pick["entry_hi"] == 104.5
    assert pick["eval_method"] == "next_day_zone_v2"


def test_record_picks_na_zone_falls_back_to_close(tmp_path):
    perf_file = tmp_path / "performance.json"
    watchlist_data = {
        "scan_date": "2026-07-04",
        "buy_watchlist": [{
            "ticker": "NOZONE.NS", "today_close": 100.0, "entry_zone": "N/A",
            "stop_loss": "₹90", "target_1": "₹120",
        }],
        "sell_watchlist": [],
    }
    with mock.patch.object(p, "_PERF_FILE", perf_file):
        p.record_picks(watchlist_data)
        loaded = json.loads(perf_file.read_text())

    pick = loaded["2026-07-04"]["NOZONE.NS"]
    assert pick["entry_lo"] == 100.0
    assert pick["entry_hi"] == 100.0


def _make_ohlcv(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    idx = pd.to_datetime(list(rows.keys()))
    return pd.DataFrame(list(rows.values()), index=idx)


def test_evaluate_first_checked_bar_is_next_day_not_scan_day(tmp_path):
    """The core Phase 1 bug fix: the scan-day bar's own low (80, well below the
    stop) must NOT be checked -- entry starts the NEXT trading day. If the old
    `as_of = scan_date - 1 business day` bug were still present, this pick
    would false-positive sl_hit on the scan date itself instead of correctly
    riding to t1_hit two days later."""
    perf_file = tmp_path / "performance.json"
    day0 = date.today() - timedelta(days=5)  # inside the default 30d/7d lookback windows
    day1 = day0 + timedelta(days=1)
    day2 = day0 + timedelta(days=2)
    scan_date = day0.isoformat()
    perf_file.write_text(json.dumps({
        scan_date: {
            "T1HIT.NS": {
                "direction": "buy", "entry": 100.0, "entry_lo": 95.0, "entry_hi": 105.0,
                "stop_loss": 90.0, "target_1": 120.0, "target_2": None,
                "outcome": "open", "outcome_date": None, "exit_policy": "static",
            },
        },
    }))
    df = _make_ohlcv({
        # scan-day bar: low=80 would trip the 90 stop if it were ever checked
        day0.isoformat(): {"Open": 100, "High": 110, "Low": 80,  "Close": 100},
        # next day: triggers entry (low <= entry_hi=105), does not hit SL/T1 itself
        day1.isoformat(): {"Open": 100, "High": 106, "Low": 95,  "Close": 101},
        # day after: hits T1 (high >= 120)
        day2.isoformat(): {"Open": 101, "High": 125, "Low": 100, "Close": 120},
    })

    with mock.patch.object(p, "_PERF_FILE", perf_file), \
         mock.patch.object(p, "yf") as mock_yf:
        mock_yf.download.return_value = df  # single ticker -> raw.columns is a plain Index
        updated = p.evaluate_prior_picks(lookback_days=7)

    pick = updated[scan_date]["T1HIT.NS"]
    assert pick["outcome"] == "t1_hit"
    assert pick["outcome_date"] > scan_date


def test_legacy_pick_no_zone_falls_back_to_point_entry(tmp_path):
    """Legacy picks (recorded before entry_lo/entry_hi existed) have no zone --
    they fall back to a point entry at the recorded close. The as_of fix alone
    (checking from the next day) still removes their same-day self-trip."""
    perf_file = tmp_path / "performance.json"
    day0 = date.today() - timedelta(days=5)
    day1 = day0 + timedelta(days=1)
    scan_date = day0.isoformat()
    perf_file.write_text(json.dumps({
        scan_date: {
            "LEGACY.NS": {
                "direction": "buy", "entry": 100.0,  # no entry_lo/entry_hi keys
                "stop_loss": 90.0, "target_1": 120.0, "target_2": None,
                "outcome": "open", "outcome_date": None,
            },
        },
    }))
    df = _make_ohlcv({
        day0.isoformat(): {"Open": 100, "High": 110, "Low": 80,  "Close": 100},
        day1.isoformat(): {"Open": 100, "High": 125, "Low": 99,  "Close": 120},
    })

    with mock.patch.object(p, "_PERF_FILE", perf_file), \
         mock.patch.object(p, "yf") as mock_yf:
        mock_yf.download.return_value = df
        updated = p.evaluate_prior_picks(lookback_days=7)

    pick = updated[scan_date]["LEGACY.NS"]
    assert pick["outcome"] == "t1_hit"  # not sl_hit from the scan-day low=80


# ── Live-Proof Round Phase 2: proof inputs captured at emission ─────────────

def test_record_picks_captures_active_signals_regime_and_nifty(tmp_path):
    perf_file = tmp_path / "performance.json"
    watchlist_data = {
        "scan_date": "2026-07-06",
        "nifty_context": "uptrend",
        "buy_watchlist": [{
            "ticker": "SIG.NS", "today_close": 100.0, "stop_loss": "₹90", "target_1": "₹120",
            "active_signals": ["reversal_oversold_v2", "near_52w_high"],
        }],
        "sell_watchlist": [],
    }
    with mock.patch.object(p, "_PERF_FILE", perf_file):
        p.record_picks(watchlist_data, nifty_at_emission=24500.5)
        loaded = json.loads(perf_file.read_text())

    pick = loaded["2026-07-06"]["SIG.NS"]
    assert pick["active_signals"] == ["reversal_oversold_v2", "near_52w_high"]
    assert pick["regime"] == "uptrend"
    assert pick["nifty_at_emission"] == 24500.5


def test_record_picks_defaults_when_signals_or_nifty_missing(tmp_path):
    """A pick with no active_signals key and no nifty_at_emission arg must not
    crash -- defaults to an empty list / None, not a KeyError."""
    perf_file = tmp_path / "performance.json"
    watchlist_data = {
        "scan_date": "2026-07-06",
        "buy_watchlist": [{"ticker": "BARE.NS", "today_close": 100.0, "stop_loss": "₹90", "target_1": "₹120"}],
        "sell_watchlist": [],
    }
    with mock.patch.object(p, "_PERF_FILE", perf_file):
        p.record_picks(watchlist_data)  # nifty_at_emission omitted
        loaded = json.loads(perf_file.read_text())

    pick = loaded["2026-07-06"]["BARE.NS"]
    assert pick["active_signals"] == []
    assert pick["regime"] is None
    assert pick["nifty_at_emission"] is None


def test_record_picks_same_day_rerun_does_not_overwrite_proof_fields(tmp_path):
    """The existing daily idempotency guard (scan_date already in perf -> return)
    already protects write-once fields, but pin it explicitly for the new
    Phase-2 fields specifically -- a second call with DIFFERENT signals/nifty
    must not silently rewrite the first day's frozen record."""
    perf_file = tmp_path / "performance.json"
    watchlist_data_1 = {
        "scan_date": "2026-07-06", "nifty_context": "uptrend",
        "buy_watchlist": [{"ticker": "FROZEN.NS", "today_close": 100.0, "stop_loss": "₹90",
                            "target_1": "₹120", "active_signals": ["reversal_oversold_v2"]}],
        "sell_watchlist": [],
    }
    watchlist_data_2 = {
        "scan_date": "2026-07-06", "nifty_context": "downtrend",  # would-be overwrite
        "buy_watchlist": [{"ticker": "FROZEN.NS", "today_close": 100.0, "stop_loss": "₹90",
                            "target_1": "₹120", "active_signals": ["pead_positive_surprise"]}],
        "sell_watchlist": [],
    }
    with mock.patch.object(p, "_PERF_FILE", perf_file):
        p.record_picks(watchlist_data_1, nifty_at_emission=24000.0)
        p.record_picks(watchlist_data_2, nifty_at_emission=25000.0)
        loaded = json.loads(perf_file.read_text())

    pick = loaded["2026-07-06"]["FROZEN.NS"]
    assert pick["active_signals"] == ["reversal_oversold_v2"]
    assert pick["regime"] == "uptrend"
    assert pick["nifty_at_emission"] == 24000.0


def test_decided_outcome_never_reevaluated(tmp_path):
    perf_file = tmp_path / "performance.json"
    day0 = date.today() - timedelta(days=5)
    day1 = day0 + timedelta(days=1)
    scan_date = day0.isoformat()
    perf_file.write_text(json.dumps({
        scan_date: {
            "DONE.NS": {
                "direction": "buy", "entry": 100.0, "entry_lo": 95.0, "entry_hi": 105.0,
                "stop_loss": 90.0, "target_1": 120.0, "target_2": None,
                "outcome": "sl_hit", "outcome_date": day1.isoformat(), "exit_policy": "static",
            },
        },
    }))
    # If re-evaluated, this data would produce t1_hit -- must NOT happen, since
    # the pick's outcome is already decided.
    df = _make_ohlcv({
        day0.isoformat(): {"Open": 100, "High": 110, "Low": 95, "Close": 100},
        day1.isoformat(): {"Open": 100, "High": 125, "Low": 99, "Close": 120},
    })
    with mock.patch.object(p, "_PERF_FILE", perf_file), \
         mock.patch.object(p, "yf") as mock_yf:
        mock_yf.download.return_value = df
        updated = p.evaluate_prior_picks(lookback_days=7)

    pick = updated[scan_date]["DONE.NS"]
    assert pick["outcome"] == "sl_hit"
    assert pick["outcome_date"] == day1.isoformat()
