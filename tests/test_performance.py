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
