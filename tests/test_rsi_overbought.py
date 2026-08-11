"""
rsi_overbought (-1): RSI > 80 predicts mean reversion.

Validated 2026-08 on 202,905 buy trades from cache/backtest_ohlcv (278 days):
the RSI>80 cohort's 10-day forward return is +0.20% vs +1.12% for the rest
(Welch t=-5.13, n=7,716). As a -1 score penalty it is sign-consistent across a
70/30 chronological holdout (train diff +0.00133 t=4.51, holdout +0.00036
t=1.13) -- the repo's ship convention.

Deliberately conservative. Two more aggressive variants were tested and
REJECTED for failing the holdout:
  - penalising the 75-80 band (RSI>75): train t=5.16 but holdout t=-0.82,
  - a graduated -1/-2/-3 at 75/80/85: train t=5.59 but holdout t=-0.85.
Only the single >80 tier at -1 survives out-of-sample. These tests pin the
threshold and magnitude so a future well-meaning "make it graduated" change has
to re-clear the holdout.
"""

from __future__ import annotations

import pytest

import numpy as np
import pandas as pd

from src.scorer import DISQUALIFIER_WEIGHTS, _build_signal_map, _compute_score
from src.technicals import compute_rsi, enrich_with_technicals


def _signal_map(technical_data: dict) -> dict:
    return _build_signal_map(
        "T.NS", [], {"volume_ratio": 2.0}, [], [], None,
        technical_data=technical_data,
    )


def test_weight_is_minus_one():
    assert DISQUALIFIER_WEIGHTS["rsi_overbought"] == -1


def test_overbought_applies_a_one_point_penalty():
    base = _signal_map({"rsi_momentum": True})
    over = _signal_map({"rsi_momentum": True, "rsi_overbought": True})
    s_base, _ = _compute_score(base, "normal", "uptrend")
    s_over, _ = _compute_score(over, "normal", "uptrend")
    assert s_base - s_over == 1


def test_penalty_is_absent_when_not_overbought():
    m = _signal_map({"rsi_momentum": True, "rsi_overbought": False})
    assert m.get("rsi_overbought") is False


def _ohlcv_with_rsi(target_rsi: float) -> pd.DataFrame:
    """Construct a 120-bar OHLCV frame whose final RSI(14) is close to
    target_rsi, by mixing up/down days at the ratio Wilder's RSI implies. Verified
    against compute_rsi in the test below, so it exercises the real code path."""
    # RSI = 100 * avg_gain / (avg_gain + avg_loss). Pick a per-bar gain g and loss
    # l with the right ratio; alternate mostly-up runs to settle near target.
    up_frac = target_rsi / 100.0
    rng = np.random.default_rng(0)
    steps = rng.random(200) < up_frac
    price = [100.0]
    for up in steps:
        price.append(price[-1] * (1.008 if up else 0.992))
    close = pd.Series(price)
    idx = pd.date_range("2025-01-01", periods=len(close), freq="D")
    return pd.DataFrame(
        {"Open": close.values, "High": close.values * 1.005,
         "Low": close.values * 0.995, "Close": close.values,
         "Volume": 1_000_000},
        index=idx,
    )


def test_real_enrich_flags_overbought_only_above_80():
    """Drive the actual enrich_with_technicals: a strongly-rising series (high
    RSI) must set rsi_overbought; a gently-rising one (moderate RSI) must not."""
    hot = _ohlcv_with_rsi(92)
    warm = _ohlcv_with_rsi(65)

    hot_rsi = compute_rsi(hot["Close"].squeeze())
    warm_rsi = compute_rsi(warm["Close"].squeeze())
    assert hot_rsi > 80 and warm_rsi <= 80, (hot_rsi, warm_rsi)  # fixtures are valid

    hot_out = enrich_with_technicals(hot, close=float(hot["Close"].iloc[-1]), atr=1.0)
    warm_out = enrich_with_technicals(warm, close=float(warm["Close"].iloc[-1]), atr=1.0)

    assert hot_out["rsi_overbought"] is True
    assert warm_out["rsi_overbought"] is False


def test_the_75_to_80_band_is_not_penalised():
    """The band that failed holdout must carry no penalty: an RSI in 75-80 sets
    neither rsi_momentum (caps at 75) nor rsi_overbought (starts above 80)."""
    df = _ohlcv_with_rsi(82)  # seed 0 / this ratio lands RSI ~75.5, inside 75-80
    rsi = compute_rsi(df["Close"].squeeze())
    assert 75 < rsi <= 80, f"fixture RSI {rsi:.1f} drifted out of the 75-80 band"

    out = enrich_with_technicals(df, close=float(df["Close"].iloc[-1]), atr=1.0)
    assert out["rsi_overbought"] is False
    assert out["rsi_momentum"] is False
