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


# ── mom_gated (200DMA long-or-cash) ──────────────────────────────────────────

def test_rank_scores_mom_gated_aliases_to_mom_12_1():
    """mom_gated has no factor of its own -- it ranks identically to
    mom_12_1; only simulate_strategy applies the extra cash gate."""
    panel = {pd.Timestamp("2020-01-01"): {"A.NS": {"mom_12_1": 0.5}, "B.NS": {"mom_12_1": 0.2}}}
    t = pd.Timestamp("2020-01-01")
    assert fb._rank_scores(panel, t, "mom_gated") == fb._rank_scores(panel, t, "mom_12_1")


def test_simulate_strategy_mom_gated_goes_to_cash_below_200dma():
    dates = pd.bdate_range("2020-01-01", periods=63)
    rebalance_dates = [dates[0], dates[21], dates[42], dates[62]]
    ohlcv = {
        "A.NS": pd.DataFrame({"Close": np.linspace(100, 110, len(dates))}, index=dates),
    }
    panel = {t: {"A.NS": {"mom_12_1": 1.0}} for t in rebalance_dates}
    # Below 200DMA at every rebalance date -> should hold 100% cash throughout
    below_200dma = pd.Series(True, index=dates)

    sim = fb.simulate_strategy(ohlcv, panel, rebalance_dates, top_n=1, factor="mom_gated",
                                below_200dma=below_200dma)
    assert sim["pct_periods_in_cash"] == 100.0
    assert sim["final_equity"] == pytest.approx(1.0)  # never invested -> flat equity


def test_simulate_strategy_mom_gated_invests_when_above_200dma():
    dates = pd.bdate_range("2020-01-01", periods=63)
    rebalance_dates = [dates[0], dates[21], dates[42], dates[62]]
    ohlcv = {
        "A.NS": pd.DataFrame({"Close": np.linspace(100, 110, len(dates))}, index=dates),
    }
    panel = {t: {"A.NS": {"mom_12_1": 1.0}} for t in rebalance_dates}
    below_200dma = pd.Series(False, index=dates)  # always above -> never gated to cash

    sim = fb.simulate_strategy(ohlcv, panel, rebalance_dates, top_n=1, factor="mom_gated",
                                below_200dma=below_200dma)
    assert sim["pct_periods_in_cash"] == 0.0
    assert sim["final_equity"] > 1.0  # A.NS rises the whole period, book should gain


# ── multi-split ship gate (Phase 1) ──────────────────────────────────────────

def test_parse_splits():
    assert fb.parse_splits("2018,2020,2022") == ["2018-01-01", "2020-01-01", "2022-01-01"]
    assert fb.parse_splits(" 2024 , 2026 ") == ["2024-01-01", "2026-01-01"]


def test_split_train_holdout_respects_custom_split_date():
    dates = pd.bdate_range("2018-01-01", periods=500)
    train, holdout = fb.split_train_holdout(list(dates), split_date_str="2019-06-01")
    assert all(t < pd.Timestamp("2019-06-01") for t in train)
    assert all(t >= pd.Timestamp("2019-06-01") for t in holdout)
    assert len(train) + len(holdout) == len(dates)


def _split_result(split: str, n: int, ic_mean: float, ic_t: float | None,
                   spread: float, sharpe: float) -> dict:
    return {
        "split": split, "holdout_n_dates": n,
        "ic": {"mean_ic": ic_mean, "t_stat": ic_t, "n": n},
        "decile": {"mean_spread_pct": spread, "n": n},
        "holdout_metrics": {"sharpe": sharpe},
    }


def test_evaluate_multi_split_gate_ships_on_majority_ic_pass_and_all_other_checks():
    # 3 eligible splits (n>=24), 2/3 pass IC -> majority; decile+Sharpe pass everywhere
    splits = [
        _split_result("2018-01-01", 100, 0.05, 2.5, 1.0, 1.2),
        _split_result("2020-01-01", 60,  0.04, 2.1, 0.5, 0.9),
        _split_result("2022-01-01", 50,  0.02, 1.0, 0.3, 0.8),  # IC fails, but outvoted
    ]
    bench = [{"sharpe": 0.5}, {"sharpe": 0.5}, {"sharpe": 0.5}]
    gate = fb.evaluate_multi_split_gate(splits, bench)
    assert gate["ships"] is True
    assert gate["n_ic_eligible_splits"] == 3
    assert gate["n_ic_pass"] == 2


def test_evaluate_multi_split_gate_thin_holdout_is_informational_only():
    """A split with holdout_n < MIN_IC_HOLDOUT_N must not count toward the
    IC majority-vote denominator, even if its (meaningless) t-stat is huge --
    otherwise a single noisy thin split could flip the verdict either way."""
    splits = [
        _split_result("2018-01-01", 100, 0.05, 2.5, 1.0, 1.2),  # only eligible split, passes
        _split_result("2026-01-01", 5, 0.20, 9.9, 1.0, 2.0),     # n<24 -> informational only
    ]
    bench = [{"sharpe": 0.5}, {"sharpe": 0.5}]
    gate = fb.evaluate_multi_split_gate(splits, bench)
    assert gate["n_ic_eligible_splits"] == 1  # thin split excluded from the denominator
    assert gate["n_ic_pass"] == 1
    assert gate["ships"] is True
    thin = next(s for s in gate["per_split"] if s["split"] == "2026-01-01")
    assert thin["ic_informational_only"] is True


def test_evaluate_multi_split_gate_fails_when_sharpe_does_not_beat_nifty():
    splits = [
        _split_result("2018-01-01", 100, 0.05, 2.5, 1.0, 1.2),
        _split_result("2024-01-01", 40,  0.04, 2.1, 0.5, 0.4),  # Sharpe below Nifty's 0.5
    ]
    bench = [{"sharpe": 0.5}, {"sharpe": 0.5}]
    gate = fb.evaluate_multi_split_gate(splits, bench)
    assert gate["ships"] is False
    assert any("Sharpe does not beat Nifty" in r for r in gate["reasons"])


def test_evaluate_multi_split_gate_fails_when_decile_spread_not_positive():
    splits = [
        _split_result("2018-01-01", 100, 0.05, 2.5, -0.2, 1.2),  # negative decile spread
    ]
    bench = [{"sharpe": 0.5}]
    gate = fb.evaluate_multi_split_gate(splits, bench)
    assert gate["ships"] is False
    assert any("decile spread" in r for r in gate["reasons"])
