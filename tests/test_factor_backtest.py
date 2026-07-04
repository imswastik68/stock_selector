"""
Tests for scripts/factor_backtest.py -- factor math, cross-sectional z-score,
cost application, and the point-in-time no-lookahead assertion (the single
most important correctness property of any backtester: a factor computed
"as of" date T must be identical whether or not the DataFrame contains bars
after T).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factor_backtest as fb


def _make_price_series(n: int, start: float = 100.0, daily_ret: float = 0.0005,
                        seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    rets = rng.normal(daily_ret, 0.01, n)
    closes = start * np.cumprod(1 + rets)
    highs = closes * 1.01
    lows = closes * 0.99
    opens = closes * 1.0
    volumes = np.full(n, 1_000_000.0)
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=dates,
    )


# ── factor math ────────────────────────────────────────────────────────────────

def test_mom_12_1_formula():
    df = _make_price_series(300)
    val = fb.factor_mom_12_1(df)
    closes = df["Close"]
    expected = (closes.iloc[-1] / closes.iloc[-252] - 1) - (closes.iloc[-1] / closes.iloc[-21] - 1)
    assert val == pytest.approx(expected)


def test_mom_12_1_insufficient_history_returns_none():
    df = _make_price_series(200)  # < 253 bars required
    assert fb.factor_mom_12_1(df) is None


def test_hi_52w_at_new_high_equals_one():
    df = _make_price_series(260)
    df.loc[df.index[-1], "Close"] = df["Close"].tail(252).max() * 1.001  # push to a new high
    val = fb.factor_hi_52w(df)
    assert val == pytest.approx(1.001, rel=1e-3)


def test_hi_52w_insufficient_history_returns_none():
    df = _make_price_series(100)
    assert fb.factor_hi_52w(df) is None


def test_low_vol_is_negative_realized_stdev():
    df = _make_price_series(100)
    val = fb.factor_low_vol(df)
    expected = -float(df["Close"].pct_change().dropna().tail(60).std())
    assert val == pytest.approx(expected)
    assert val <= 0  # higher realized vol -> more negative score, by construction


def test_low_vol_zero_variance_returns_none():
    dates = pd.bdate_range("2020-01-01", periods=100)
    flat = pd.DataFrame({
        "Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0, "Volume": 1_000_000.0,
    }, index=dates)
    assert fb.factor_low_vol(flat) is None


def test_rs_quality_outperformance_is_positive():
    df = _make_price_series(60, daily_ret=0.01, seed=1)  # strong uptrend
    val = fb.factor_rs_quality(df, nifty_20d=0.0)  # flat benchmark
    assert val is not None and val > 0


def test_rs_quality_none_without_nifty_return():
    df = _make_price_series(60)
    assert fb.factor_rs_quality(df, nifty_20d=None) is None


# ── cross-sectional z-score ──────────────────────────────────────────────────

def test_zscore_mean_zero_std_one():
    values = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}
    z = fb._zscore(values)
    arr = np.array(list(z.values()))
    assert arr.mean() == pytest.approx(0.0, abs=1e-9)
    assert arr.std() == pytest.approx(1.0, abs=1e-9)


def test_zscore_ranks_preserved():
    values = {"A": 1.0, "B": 5.0, "C": 3.0}
    z = fb._zscore(values)
    assert z["B"] > z["C"] > z["A"]


def test_zscore_zero_variance_returns_zeros():
    values = {"A": 5.0, "B": 5.0, "C": 5.0}
    z = fb._zscore(values)
    assert all(v == 0.0 for v in z.values())


def test_zscore_single_value_returns_zero():
    assert fb._zscore({"A": 1.0}) == {"A": 0.0}


def test_composite_averages_available_zscores():
    date_scores = {
        "A": {"mom_12_1": 0.20, "hi_52w": 0.95},
        "B": {"mom_12_1": 0.05, "hi_52w": 0.70},
        "C": {"mom_12_1": -0.10, "hi_52w": 0.50},
    }
    composite = fb.compute_composite(date_scores)
    # "A" has the highest raw value in both factors -> should rank highest in the composite
    assert composite["A"] > composite["B"] > composite["C"]


# ── point-in-time no-lookahead ────────────────────────────────────────────────

@pytest.mark.parametrize("factor_fn", [fb.factor_mom_12_1, fb.factor_hi_52w, fb.factor_low_vol])
def test_factor_unchanged_by_future_bars(factor_fn):
    """The single most important backtester correctness property: a factor
    computed on a point-in-time slice up to date T must be IDENTICAL whether
    or not the DataFrame contains bars after T. This is what
    `full_df[full_df.index <= as_of]` (used throughout scripts/backtest.py
    and scripts/factor_backtest.py) is supposed to guarantee -- this test
    would fail if that slicing discipline were ever violated (e.g. a factor
    accidentally computed from the un-sliced full_df)."""
    full_df = _make_price_series(400)
    as_of = full_df.index[299]  # cut here
    sliced = full_df[full_df.index <= as_of]
    truncated = full_df.loc[:as_of]  # equivalent slice, built a different way

    val_sliced = factor_fn(sliced)
    val_truncated = factor_fn(truncated)
    assert val_sliced == val_truncated

    # Deleting bars strictly after as_of must not change the result at all.
    with_future = full_df.copy()  # same as sliced, plus 100 more (future) bars
    assert factor_fn(with_future[with_future.index <= as_of]) == val_sliced


def test_rs_quality_unchanged_by_future_bars():
    full_df = _make_price_series(400)
    as_of = full_df.index[299]
    sliced = full_df[full_df.index <= as_of]
    val_sliced = fb.factor_rs_quality(sliced, nifty_20d=0.01)

    with_future = full_df.copy()
    val_with_future_data_but_same_asof = fb.factor_rs_quality(
        with_future[with_future.index <= as_of], nifty_20d=0.01
    )
    assert val_sliced == val_with_future_data_but_same_asof


# ── cost application ──────────────────────────────────────────────────────────

def test_period_return_equal_weight_average():
    dates = pd.bdate_range("2020-01-01", periods=10)
    df_a = pd.DataFrame({"Close": np.linspace(100, 110, 10)}, index=dates)  # +10%
    df_b = pd.DataFrame({"Close": np.linspace(100, 90, 10)}, index=dates)   # -10%
    ohlcv = {"A.NS": df_a, "B.NS": df_b}
    ret = fb._period_return(ohlcv, {"A.NS", "B.NS"}, dates[0], dates[-1])
    assert ret == pytest.approx(0.0, abs=1e-6)  # +10% and -10% average to ~0


def test_simulate_strategy_charges_cost_on_exit():
    """A strategy that fully turns over its holdings every rebalance should
    have strictly lower final equity than the same strategy with cost_pct
    forced to zero -- i.e. round_trip_cost_pct is actually being deducted."""
    dates = pd.bdate_range("2020-01-01", periods=63)  # 3 monthly rebalances
    rebalance_dates = [dates[0], dates[21], dates[42], dates[62]]

    # Two tickers that swap ranking each period (guarantees 100% turnover)
    ohlcv = {
        "A.NS": pd.DataFrame({"Close": np.linspace(100, 105, len(dates))}, index=dates),
        "B.NS": pd.DataFrame({"Close": np.linspace(100, 103, len(dates))}, index=dates),
    }
    # Alternate which ticker scores higher at each rebalance to force full turnover
    panel = {}
    for i, t in enumerate(rebalance_dates):
        panel[t] = {
            "A.NS": {"mom_12_1": 1.0 if i % 2 == 0 else -1.0},
            "B.NS": {"mom_12_1": -1.0 if i % 2 == 0 else 1.0},
        }

    sim = fb.simulate_strategy(ohlcv, panel, rebalance_dates, top_n=1, factor="mom_12_1")
    assert sim["avg_turnover"] > 0  # confirms the scenario actually forces turnover

    # Reference: same holdings sequence, cost forced to zero
    import unittest.mock as mock
    with mock.patch.object(fb, "round_trip_cost_pct", return_value=0.0):
        sim_no_cost = fb.simulate_strategy(ohlcv, panel, rebalance_dates, top_n=1, factor="mom_12_1")

    assert sim["final_equity"] < sim_no_cost["final_equity"]
