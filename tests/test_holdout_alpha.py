"""
Tests for scripts/holdout_alpha.py -- the fast holdout proxy for the live-proof
verdict. Focus is the ONE piece of new logic: reconstruct_events (pure,
network-free). The gate itself is already covered by tests/test_gates.py; here
we only assert the reconstruction feeds it the right synthetic picks with the
right cost-netted abnormal math, and that the wiring produces a live-currency
verdict.
"""

from __future__ import annotations

import pandas as pd
import pytest

import holdout_alpha as ha


def _bdays(n: int) -> pd.DatetimeIndex:
    # fixed anchor so the fixture is deterministic regardless of run date
    return pd.bdate_range(start="2024-01-01", periods=n)


def _ohlcv(closes: list[float], volume: float = 300_000.0) -> pd.DataFrame:
    idx = _bdays(len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({
        "Open": c,
        "High": c * 1.01,   # High != Low so _turnover_ok's halt check passes
        "Low": c * 0.99,
        "Close": c,
        "Volume": volume,
    }, index=idx)


# A series engineered so reversal_oversold_v2 fires on exactly one bar (T=idx 44):
#   flat 100 through idx41, then 100->98->95->92 into T (3d ret = -8% <= -7%,
#   two consecutive losses -> RSI(2) ~ 0 < 10), then recovers to 110 forward so
#   fwd_10d gross = +10%.
def _firing_stock() -> tuple[pd.DataFrame, pd.Timestamp]:
    closes = [100.0] * 42          # idx 0..41
    closes += [98.0, 95.0, 92.0]   # idx 42,43,44 (T)
    closes += [100.0]              # idx 45 = T+1 (entry close)
    closes += [110.0] * 21         # idx 46..66 = T+2..T+22
    df = _ohlcv(closes)
    T = df.index[44]
    return df, T


def _flat_nifty(n: int, level: float = 20_000.0) -> pd.DataFrame:
    idx = _bdays(n)
    c = pd.Series([level] * n, index=idx, dtype=float)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 0.0}, index=idx)


def test_fixture_actually_fires():
    """Guard: if the firing condition ever changes, this catches a stale fixture
    before the downstream assertions mislead."""
    df, T = _firing_stock()
    # _fires evaluates the LAST bar of the series it's given, so check the
    # point-in-time slice up to T (exactly what reconstruct_events does), not
    # the full recovered series.
    slice_ = df[df.index <= T].tail(ha._LOOKBACK_BARS)
    fires, ret3d, rsi2 = ha.reversal._fires(slice_["Close"])
    assert fires, f"fixture must fire at T; ret3d={ret3d} rsi2={rsi2}"


def test_reconstruct_emits_cost_netted_abnormal():
    df, T = _firing_stock()
    nifty = _flat_nifty(len(df))
    perf = ha.reconstruct_events({"TEST.NS": df}, nifty, df.index[0].date().isoformat(),
                                 df.index[-1].date().isoformat())

    key = T.date().isoformat()
    assert list(perf.keys()) == [key], "exactly one firing date expected"
    pick = perf[key]["TEST.NS"]

    # gross fwd_10d = 110/100-1 = +10%, cost = 0.3 -> net 9.7; NIFTY flat -> abnormal 9.7
    assert pick["fwd_10d"] == 9.7
    assert pick["nifty_fwd_10d"] == 0.0
    assert pick["abnormal_10d"] == 9.7
    # 2x-cost stress: 10 - 0.6 = 9.4, minus flat NIFTY = 9.4
    assert pick["abnormal_10d_stress"] == 9.4
    # shape the live gate requires
    assert pick["eval_method"] == "next_day_zone_v2"
    assert pick["direction"] == "buy"
    assert pick["active_signals"] == ["reversal_oversold_v2"]


def test_non_firing_stock_emits_nothing():
    flat = _ohlcv([100.0] * 67)  # never oversold
    nifty = _flat_nifty(67)
    perf = ha.reconstruct_events({"FLAT.NS": flat}, nifty, flat.index[0].date().isoformat(),
                                 flat.index[-1].date().isoformat())
    assert perf == {}


def test_insufficient_forward_bars_skipped():
    # same firing shape but truncated so the firing bar has <10 forward bars
    df, T = _firing_stock()
    truncated = df.iloc[:44 + 5]  # only 4 bars after T
    nifty = _flat_nifty(len(truncated))
    perf = ha.reconstruct_events({"TEST.NS": truncated}, nifty,
                                 truncated.index[0].date().isoformat(),
                                 truncated.index[-1].date().isoformat())
    assert perf == {}, "cannot measure fwd_10d without 10 forward bars -> skip"


def test_missing_nifty_yields_null_abnormal():
    df, T = _firing_stock()
    empty_nifty = pd.DataFrame()
    perf = ha.reconstruct_events({"TEST.NS": df}, empty_nifty, df.index[0].date().isoformat(),
                                 df.index[-1].date().isoformat())
    pick = perf[T.date().isoformat()]["TEST.NS"]
    assert pick["fwd_10d"] == 9.7            # stock side still computed
    assert pick["nifty_fwd_10d"] is None
    assert pick["abnormal_10d"] is None      # no benchmark -> no alpha, not a fake 0


def test_feeds_live_gate_in_live_currency():
    df, T = _firing_stock()
    nifty = _flat_nifty(len(df))
    perf = ha.reconstruct_events({"TEST.NS": df}, nifty, df.index[0].date().isoformat(),
                                 df.index[-1].date().isoformat())
    gate = ha.live_alpha_gate(perf=perf)
    rev = gate["per_signal"]["reversal_oversold_v2"]
    # one event -> below n=30 -> INSUFFICIENT, but wired correctly through the real gate
    assert rev["n"] == 1
    assert rev["verdict"] == "INSUFFICIENT"
    assert gate["aggregate"]["n"] == 1
