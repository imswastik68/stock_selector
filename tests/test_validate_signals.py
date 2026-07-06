"""
Tests for scripts/validate_signals.py -- the unified reconciliation harness.
The critical property is that the action logic is WEIGHT-SIGN-AWARE: a
disqualifier (negative weight) is validated by NEGATIVE lift, not positive.
An early version flagged every correctly-working disqualifier as DEMOTE.
"""

from __future__ import annotations

import json
from unittest import mock

import validate_signals as vs


def _run_collect(monkeypatch, tmp_path, ohlcv_signals=None, event_signals=None,
                 short_term=None, disq=None):
    stats_file = tmp_path / "backtest_signal_stats.json"
    event_file = tmp_path / "event_backtest.json"
    stats_file.write_text(json.dumps({"signals": ohlcv_signals or {}}))
    event_file.write_text(json.dumps({"signals": event_signals or {}}))
    monkeypatch.setattr(vs, "SIGNAL_STATS", stats_file)
    monkeypatch.setattr(vs, "EVENT_BT", event_file)
    monkeypatch.setattr(vs.scorer, "SHORT_TERM_WEIGHTS", short_term or {})
    monkeypatch.setattr(vs.scorer, "SWING_WEIGHTS", {})
    monkeypatch.setattr(vs.scorer, "DISQUALIFIER_WEIGHTS", disq or {})
    monkeypatch.setattr(vs.scorer, "BEARISH_EVENT_WEIGHTS", {})
    monkeypatch.setattr(vs.scorer, "PENDING_VALIDATION", {})
    rows = vs._collect()
    return {r["signal"]: r for r in rows}


def test_buy_signal_positive_lift_is_ok(monkeypatch, tmp_path):
    rows = _run_collect(monkeypatch, tmp_path,
                        ohlcv_signals={"good": {"n": 10000, "wr_lift_pp": 2.0, "ret_lift": 0.8}},
                        short_term={"good": 3})
    assert rows["good"]["action"] == "OK"


def test_buy_signal_negative_lift_is_demote(monkeypatch, tmp_path):
    rows = _run_collect(monkeypatch, tmp_path,
                        ohlcv_signals={"bad": {"n": 10000, "wr_lift_pp": -1.0, "ret_lift": -0.5}},
                        short_term={"bad": 2})
    assert rows["bad"]["action"] == "DEMOTE"


def test_disqualifier_negative_lift_is_ok_not_demote(monkeypatch, tmp_path):
    """The core fix: a disqualifier (negative weight) with negative lift is
    working correctly -- it must NOT be flagged as a demote."""
    rows = _run_collect(monkeypatch, tmp_path,
                        ohlcv_signals={"macd_bearish_cross": {"n": 68391, "wr_lift_pp": -3.1, "ret_lift": -0.84}},
                        disq={"macd_bearish_cross": -1})
    assert rows["macd_bearish_cross"]["action"] == "OK"


def test_disqualifier_strong_positive_lift_is_demote(monkeypatch, tmp_path):
    """A disqualifier that is actually HELPING (clearly positive lift) is a
    genuine mis-weight and should flag."""
    rows = _run_collect(monkeypatch, tmp_path,
                        ohlcv_signals={"mislabeled": {"n": 10000, "wr_lift_pp": 2.0, "ret_lift": 0.9}},
                        disq={"mislabeled": -1})
    assert rows["mislabeled"]["action"] == "DEMOTE"


def test_disqualifier_tiny_positive_lift_within_deadband_is_ok(monkeypatch, tmp_path):
    rows = _run_collect(monkeypatch, tmp_path,
                        ohlcv_signals={"noise": {"n": 10000, "wr_lift_pp": 0.1, "ret_lift": 0.05}},
                        disq={"noise": -1})
    assert rows["noise"]["action"] == "OK"


def test_unweighted_shipping_signal_is_promote(monkeypatch, tmp_path):
    rows = _run_collect(monkeypatch, tmp_path,
                        event_signals={"delivery_surge": {"n": 38279, "wr_lift_pp": 1.85,
                                        "ret_lift": 0.316, "verdict": "SHIP"}},
                        short_term={"delivery_surge": 0})
    assert rows["delivery_surge"]["action"] == "PROMOTE"


def test_untestable_event_signal_is_wait(monkeypatch, tmp_path):
    rows = _run_collect(monkeypatch, tmp_path,
                        event_signals={"bulk_deal_fii_dii": {"verdict": "UNTESTABLE",
                                        "n": None, "wr_lift_pp": None, "ret_lift": None}},
                        short_term={"bulk_deal_fii_dii": 0})
    assert rows["bulk_deal_fii_dii"]["action"] == "WAIT (no data)"


def test_insufficient_sample_is_wait(monkeypatch, tmp_path):
    rows = _run_collect(monkeypatch, tmp_path,
                        event_signals={"thin": {"verdict": "INSUFFICIENT_SAMPLE",
                                        "n": 120, "wr_lift_pp": 1.0, "ret_lift": 0.5}},
                        short_term={"thin": 0})
    assert rows["thin"]["action"].startswith("WAIT")


def test_backtest_only_column_names_mapped_to_live_keys(monkeypatch, tmp_path):
    """volume_surge (backtest col) -> volume_5x (live key) so reconciliation
    finds the live weight."""
    rows = _run_collect(monkeypatch, tmp_path,
                        ohlcv_signals={"volume_surge": {"n": 33214, "wr_lift_pp": -2.4, "ret_lift": -0.89}},
                        disq={"volume_5x": -1})
    assert "volume_5x" in rows
    assert rows["volume_5x"]["live_weight"] == -1
    assert rows["volume_5x"]["action"] == "OK"


def test_diagnostic_signal_is_never_a_promotion_candidate(monkeypatch, tmp_path):
    """A signal name containing '_diag' (backtest_events.py's convention for a
    deliberately non-promotable variant, e.g. reversal_oversold_diag_no_trend)
    must not be flagged PROMOTE even if it SHIPs -- it was excluded from the
    gate on purpose (anti-gate-shopping), not because it's unweighted by
    oversight."""
    rows = _run_collect(monkeypatch, tmp_path,
                        event_signals={"reversal_oversold_diag_no_trend": {
                            "n": 6843, "wr_lift_pp": 6.87, "ret_lift": 2.102, "verdict": "SHIP"}},
                        short_term={"reversal_oversold_diag_no_trend": 0})
    assert rows["reversal_oversold_diag_no_trend"]["action"] == "OK"


# ── --run subprocess invocation (SOTA Round Phase 4) ─────────────────────────

def test_run_flag_invokes_backtest_events_with_weeks_260(monkeypatch, tmp_path):
    """--run must pass --weeks 260 to backtest_events.py -- not silently fall
    back to that script's own 156w default, which would understate every
    event-signal sample size on the monthly automated revalidation."""
    stats_file = tmp_path / "backtest_signal_stats.json"
    event_file = tmp_path / "event_backtest.json"
    out_file = tmp_path / "signal_validation.json"
    stats_file.write_text(json.dumps({"signals": {}}))
    event_file.write_text(json.dumps({"signals": {}}))
    monkeypatch.setattr(vs, "SIGNAL_STATS", stats_file)
    monkeypatch.setattr(vs, "EVENT_BT", event_file)
    monkeypatch.setattr(vs, "OUT_FILE", out_file)

    calls = []
    monkeypatch.setattr(vs.subprocess, "run", lambda args, **k: calls.append(args))
    vs.main(do_run=True, as_json=False)

    assert len(calls) == 2
    backtest_events_call = calls[1]
    assert "backtest_events.py" in str(backtest_events_call[1])
    assert "--weeks" in backtest_events_call
    assert backtest_events_call[backtest_events_call.index("--weeks") + 1] == "260"
