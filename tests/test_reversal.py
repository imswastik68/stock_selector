"""
Tests for src/data/reversal.py -- the live reversal_oversold_v2 scanner
(SOTA Round Phase 2). The module constants (RET_3D_THRESHOLD, RSI2_THRESHOLD)
are the single source of truth, imported by scripts/backtest_events.py's
collect_reversal_diag_events -- these tests confirm the live _fires() /
_parse_batch() logic can't silently diverge from what was backtested.
"""

from __future__ import annotations

import pandas as pd

import backtest_events as be
import src.data.reversal as rv


def _make_df(closes: list[float], vol: int = 300000) -> pd.DataFrame:
    idx = pd.bdate_range(start="2025-01-01", periods=len(closes))
    return pd.DataFrame({
        "Open": closes, "High": [c * 1.01 for c in closes], "Low": [c * 0.99 for c in closes],
        "Close": closes, "Volume": [vol] * len(closes),
    }, index=idx)


def _dip_closes():
    """Same fixture math verified in the Alpha Round: 205 mildly-uptrending
    bars then a sharp final-day panic drop -- ret_3d=-8%, rsi2=1.7 (well
    under the 10 threshold)."""
    import numpy as np
    rng = np.random.default_rng(7)
    base = 100 + np.cumsum(rng.normal(0.15, 0.5, 205))
    base = np.clip(base, 50, None)
    base[-3] = base[-4]
    base[-2] = base[-4] * 0.95
    base[-1] = base[-4] * 0.92
    return list(base)


# ── _fires: the shared condition ─────────────────────────────────────────────

def test_fires_on_qualifying_dip():
    close = pd.Series(_dip_closes())
    fires, ret_3d, rsi2 = rv._fires(close)
    assert fires is True
    assert abs(ret_3d - (-0.08)) < 1e-6
    assert rsi2 < 10


def test_fires_false_on_flat_series():
    close = pd.Series([100.0] * 50)
    fires, ret_3d, rsi2 = rv._fires(close)
    assert fires is False
    assert ret_3d == 0.0


def test_fires_false_with_too_few_bars():
    close = pd.Series([100.0, 99.0])
    fires, ret_3d, rsi2 = rv._fires(close)
    assert fires is False
    assert ret_3d is None
    assert rsi2 is None


# ── _turnover_ok ──────────────────────────────────────────────────────────────

def test_turnover_ok_passes_liquid_stock():
    df = _make_df([100.0] * 35, vol=300000)  # 100*300000=3e7=3cr >= 2cr
    assert rv._turnover_ok(df) is True


def test_turnover_ok_fails_illiquid_stock():
    df = _make_df([100.0] * 35, vol=1000)  # 100*1000=1e5=0.01cr < 2cr
    assert rv._turnover_ok(df) is False


def test_turnover_ok_fails_on_halted_bar():
    df = _make_df([100.0] * 35, vol=300000)
    df.loc[df.index[-1], "High"] = df.loc[df.index[-1], "Low"]  # zero-range halt print
    assert rv._turnover_ok(df) is False


# ── _parse_batch: end-to-end single-ticker parse ─────────────────────────────

def test_parse_batch_fires_on_qualifying_dip():
    df = _make_df(_dip_closes())
    results = rv._parse_batch(df, ["DIP.NS"])
    assert len(results) == 1
    assert results[0]["ticker"] == "DIP.NS"
    assert results[0]["ret_3d_pct"] < -7.0
    assert results[0]["rsi2"] < 10


def test_parse_batch_skips_illiquid_qualifying_dip():
    df = _make_df(_dip_closes(), vol=1000)  # qualifies on price/RSI but fails turnover
    results = rv._parse_batch(df, ["ILLIQ.NS"])
    assert results == []


def test_parse_batch_skips_non_qualifying_ticker():
    df = _make_df([100.0] * 40)  # flat, never fires
    results = rv._parse_batch(df, ["FLAT.NS"])
    assert results == []


def test_parse_batch_multiindex_columns():
    df = _make_df(_dip_closes())
    df.columns = pd.MultiIndex.from_product([["DIP.NS"], df.columns])
    results = rv._parse_batch(df, ["DIP.NS"])
    assert len(results) == 1
    assert results[0]["ticker"] == "DIP.NS"


# ── backtest<->live parity (the critical regression) ─────────────────────────

def test_live_condition_matches_backtest_collector_on_shared_fixture():
    """The SAME OHLCV fed through both src.data.reversal._fires (live) and
    scripts/backtest_events.py's collect_reversal_diag_events (backtest) must
    agree on whether the condition fires -- they share module constants, but
    this pins the actual boolean outcome, not just the constants' equality."""
    closes = _dip_closes()
    df = _make_df(closes)

    live_fires, _, _ = rv._fires(pd.Series(closes))

    events, reason = be.collect_reversal_diag_events(weeks=9999, ohlcv={"DIP.NS": df})
    assert reason is None
    bt_fires = (df.index[-1].date(), "DIP.NS") in events

    assert live_fires is True
    assert bt_fires is True
    assert live_fires == bt_fires


def test_live_condition_matches_backtest_collector_on_non_firing_fixture():
    closes = [100.0] * 40
    df = _make_df(closes)

    live_fires, _, _ = rv._fires(pd.Series(closes))
    events, reason = be.collect_reversal_diag_events(weeks=9999, ohlcv={"FLAT.NS": df})
    assert reason is None
    bt_fires = any(t == "FLAT.NS" for _, t in events)

    assert live_fires is False
    assert bt_fires is False
    assert live_fires == bt_fires


def test_shared_constants_are_the_single_source_of_truth():
    """scripts/backtest_events.py must import (not hardcode a copy of) these
    two thresholds -- this pins the values so a future edit to one without
    the other fails loudly."""
    assert be._REV_RET_3D == rv.RET_3D_THRESHOLD == -0.07
    assert be._REV_RSI2 == rv.RSI2_THRESHOLD == 10.0


# ── score_candidates integration (weight=3, promoted SOTA Round Phase 2) ─────

import src.scorer as s


def test_reversal_signal_alone_qualifies_as_a_candidate():
    """reversal_oversold_v2 weight=3 clears MIN_SCORE=2 on its own -- unlike
    PEAD, this IS a standalone entry trigger, not just a booster."""
    candidates = s.score_candidates(
        [], [], [], [], [],
        reversal_signals=[{"ticker": "DIP.NS", "ret_3d_pct": -8.0, "rsi2": 1.7, "today_close": 92.0}],
    )
    match = next((c for c in candidates if c["ticker"] == "DIP.NS"), None)
    assert match is not None
    assert match["score"] == 3
    assert "reversal_oversold_v2" in match["active_signals"]
    assert match["today_close"] == 92.0


def test_reversal_signal_absent_ticker_not_scored():
    candidates = s.score_candidates([], [], [], [], [], reversal_signals=[])
    assert candidates == []


def test_reversal_signal_combines_with_delivery_surge():
    candidates = s.score_candidates(
        [], [], [], [], [],
        delivery_signals={"DIP.NS": {"delivery_surge": True}},
        reversal_signals=[{"ticker": "DIP.NS", "ret_3d_pct": -8.0, "rsi2": 1.7, "today_close": 92.0}],
    )
    match = next((c for c in candidates if c["ticker"] == "DIP.NS"), None)
    assert match is not None
    assert match["score"] == 4  # reversal_oversold_v2 (3) + delivery_surge (1)


def test_build_signal_map_reversal_data_presence_is_the_signal():
    signals = s._build_signal_map(
        "DIP.NS", [], None, [], [], None, reversal_data={"ticker": "DIP.NS"},
    )
    assert signals["reversal_oversold_v2"] is True


def test_build_signal_map_no_reversal_data():
    signals = s._build_signal_map(
        "DIP.NS", [], None, [], [], None, reversal_data=None,
    )
    assert signals["reversal_oversold_v2"] is False
