"""
Regression tests for the Phase 4 discipline layer: src/risk.py's
drawdown_state() circuit breaker and src/performance.py's loss_streak_state()
cooldown. Both gate NEW position sizing in main.py; neither touches
already-open holdings.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from unittest import mock

import pytest

import src.risk as risk
import src.performance as perf


# ── drawdown_state ────────────────────────────────────────────────────────────

def _curve(*equities: float) -> list[dict]:
    return [{"date": f"2026-01-{i+1:02d}", "equity": e} for i, e in enumerate(equities)]


def test_drawdown_state_empty_history_is_normal():
    assert risk.drawdown_state([]) == {"state": "normal", "drawdown_pct": 0.0, "peak_equity": None}


def test_drawdown_state_normal_within_10pct():
    state = risk.drawdown_state(_curve(100_000, 98_000, 95_000))  # -5% from peak
    assert state["state"] == "normal"
    assert state["drawdown_pct"] == pytest.approx(-5.0, abs=0.01)


def test_drawdown_state_reduced_between_10_and_15pct():
    state = risk.drawdown_state(_curve(100_000, 88_000))  # -12%
    assert state["state"] == "reduced"


def test_drawdown_state_halted_beyond_15pct():
    state = risk.drawdown_state(_curve(100_000, 80_000))  # -20%
    assert state["state"] == "halted"


def test_drawdown_state_hysteresis_stays_halted_until_8pct_recovery():
    """Once drawdown breaches -15%, recovering only to -9% (still worse than
    the -8% reset line) must stay halted, even though -9% alone would
    otherwise read as 'normal' (better than the -10% reduced threshold)."""
    curve = _curve(100_000, 80_000, 91_000)  # peak 100k -> -20% -> recovers to -9%
    state = risk.drawdown_state(curve)
    assert state["state"] == "halted"


def test_drawdown_state_recovers_to_normal_below_8pct():
    curve = _curve(100_000, 80_000, 93_000)  # -20% then recovers to -7%
    state = risk.drawdown_state(curve)
    assert state["state"] == "normal"


def test_drawdown_state_nifty_recovery_overrides_halt():
    curve = _curve(100_000, 80_000, 88_000)  # -20% then only recovers to -12%
    state = risk.drawdown_state(curve, nifty_above_200dma=True)
    assert state["state"] == "reduced"  # halt lifted by Nifty signal, but still -12% -> reduced


def test_drawdown_state_peak_tracks_new_highs():
    curve = _curve(100_000, 110_000, 97_900)  # peak becomes 110k, current dd = -11%
    state = risk.drawdown_state(curve)
    assert state["peak_equity"] == 110_000
    assert state["state"] == "reduced"


# ── loss_streak_state ─────────────────────────────────────────────────────────

def _perf_with_outcomes(outcomes: list[str]) -> dict:
    """Build a performance.json-shaped dict with `outcomes` in chronological
    order, one pick per day, each already decided on that day."""
    today = date.today()
    data = {}
    for i, outcome in enumerate(outcomes):
        d = (today - timedelta(days=len(outcomes) - i)).isoformat()
        data[d] = {f"T{i}.NS": {
            "direction": "buy", "outcome": outcome, "outcome_date": d,
        }}
    return data


def test_loss_streak_no_cooldown_below_threshold(tmp_path):
    perf_file = tmp_path / "performance.json"
    perf_file.write_text(json.dumps(_perf_with_outcomes(
        ["t1_hit", "sl_hit", "sl_hit", "sl_hit"]  # only 3 consecutive losses
    )))
    with mock.patch.object(perf, "_PERF_FILE", perf_file):
        state = perf.loss_streak_state()
    assert state["in_cooldown"] is False
    assert state["streak"] == 3


def test_loss_streak_triggers_cooldown_at_5_consecutive(tmp_path):
    perf_file = tmp_path / "performance.json"
    perf_file.write_text(json.dumps(_perf_with_outcomes(
        ["t1_hit", "sl_hit", "sl_hit", "sl_hit", "sl_hit", "sl_hit"]  # 5 consecutive losses
    )))
    with mock.patch.object(perf, "_PERF_FILE", perf_file):
        state = perf.loss_streak_state()
    assert state["in_cooldown"] is True
    assert state["streak"] == 5
    assert state["cooldown_until"] is not None


def test_loss_streak_broken_by_a_win_resets_streak(tmp_path):
    perf_file = tmp_path / "performance.json"
    perf_file.write_text(json.dumps(_perf_with_outcomes(
        ["sl_hit", "sl_hit", "sl_hit", "sl_hit", "sl_hit", "t1_hit"]  # streak broken by the last win
    )))
    with mock.patch.object(perf, "_PERF_FILE", perf_file):
        state = perf.loss_streak_state()
    assert state["in_cooldown"] is False
    assert state["streak"] == 0


def test_loss_streak_cooldown_expires_after_7_days(tmp_path):
    perf_file = tmp_path / "performance.json"
    old_date = (date.today() - timedelta(days=30)).isoformat()
    perf_file.write_text(json.dumps({
        old_date: {f"T{i}.NS": {"direction": "buy", "outcome": "sl_hit", "outcome_date": old_date}
                   for i in range(5)}
    }))
    with mock.patch.object(perf, "_PERF_FILE", perf_file):
        state = perf.loss_streak_state(lookback_days=60)
    assert state["streak"] == 5
    assert state["in_cooldown"] is False  # cooldown_until is 7 days past a 30-day-old loss -- long expired


def test_loss_streak_open_picks_ignored(tmp_path):
    """An 'open' pick (no outcome_date yet) must not count as a decided
    outcome -- it should neither extend nor break the trailing streak."""
    perf_file = tmp_path / "performance.json"
    data = _perf_with_outcomes(["sl_hit"] * 5)
    data[date.today().isoformat()] = {"OPEN.NS": {"direction": "buy", "outcome": "open"}}
    perf_file.write_text(json.dumps(data))
    with mock.patch.object(perf, "_PERF_FILE", perf_file):
        state = perf.loss_streak_state()
    assert state["streak"] == 5
    assert state["in_cooldown"] is True


# ── size_position risk_multiplier (Phase 4 wiring) ───────────────────────────

def test_size_position_risk_multiplier_halves_shares():
    full = risk.size_position(250.0, 238.0, capital=100_000, risk_pct=0.01)
    half = risk.size_position(250.0, 238.0, capital=100_000, risk_pct=0.01, risk_multiplier=0.5)
    assert half["shares"] < full["shares"]
    # Integer share flooring means this is approximate, not exact, halving.
    assert half["risk_amount"] == pytest.approx(full["risk_amount"] / 2, rel=0.02)


def test_size_position_default_multiplier_is_unchanged_behavior():
    a = risk.size_position(250.0, 238.0, capital=100_000, risk_pct=0.01)
    b = risk.size_position(250.0, 238.0, capital=100_000, risk_pct=0.01, risk_multiplier=1.0)
    assert a == b
