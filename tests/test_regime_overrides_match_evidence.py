"""REGIME_WEIGHTS must not contradict the measured sign of a signal.

The override table was never validated when it was generated (2026-07, Phase 5) --
15 of its 37 entries had a sign the backtest contradicts. Audited and pruned
2026-09-17 by scripts/regime_override_audit.py; holdout per-day rank IC on
fwd_10d went 0.0241 -> 0.0356 (+48%, paired t=+1.70), positive in 12 of 12
split-point x horizon cells.

The regression that motivated it: "ranging" set rsi_momentum to -2, while
rsi_momentum is the only signal in scorer.py that is sign-consistently positive
across the 70/30 holdout at every horizon (fwd_5d train +0.91 t=11.63 / holdout
+0.82 t=4.13; fwd_10d +1.02 t=9.09 / +0.60 t=2.55; fwd_20d +0.81 t=4.82 /
+0.41 t=1.44). Ranging days are ~22% of the backtest, so the scorer spent a
fifth of its life penalising its best predictor.

These tests are structural (no network, no CSV) so they run anywhere. Re-run the
audit script to regenerate the evidence itself.
"""

from __future__ import annotations

import pytest

from src.scorer import (
    BEARISH_EVENT_WEIGHTS,
    DISQUALIFIER_WEIGHTS,
    REGIME_WEIGHTS,
    SHORT_TERM_WEIGHTS,
)

REGIMES = ("uptrend", "ranging", "downtrend")

# (regime, signal, the override value that the TRAIN data contradicted)
DROPPED = [
    ("uptrend", "rs_vs_nifty", +1),
    ("uptrend", "rsi_bearish_div", +2),
    ("uptrend", "macd_bearish_cross", -2),
    ("uptrend", "bb_squeeze_breakout", -2),
    ("uptrend", "bullish_candle", -2),
    ("uptrend", "actual_52w_breakout", +1),
    ("ranging", "rsi_momentum", -2),
    ("ranging", "rs_vs_nifty", -2),
    ("ranging", "rsi_bearish_div", -3),
    ("ranging", "macd_bearish_cross", +1),
    ("ranging", "bearish_candle", +3),
    ("ranging", "distribution_signal", -1),
    ("downtrend", "rs_vs_nifty", +5),
    ("downtrend", "macd_bearish_cross", -5),
    ("downtrend", "bullish_candle", +5),
]

# Entries that measured CORRECT and must survive -- the table's real value. An
# earlier pass wrongly concluded the whole table should go; that was an artifact
# of omitting exactly these. Deleting everything measures as a wash.
KEPT = [
    ("uptrend", "distribution_signal", -4),
    ("uptrend", "heavy_selling", -5),
    ("uptrend", "volume_5x", -2),
    ("ranging", "heavy_selling", -3),
    ("ranging", "volume_5x", -2),
    ("downtrend", "distribution_signal", -5),
    ("downtrend", "heavy_selling", -5),
    ("downtrend", "volume_5x", -3),
]


@pytest.mark.parametrize("regime", REGIMES)
def test_rsi_momentum_is_never_penalised(regime):
    """The one sign-consistently positive signal must never get a negative
    regime weight. This is the specific bug the 2026-09-17 audit found."""
    override = REGIME_WEIGHTS.get(regime) or {}
    weight = override.get("rsi_momentum", SHORT_TERM_WEIGHTS["rsi_momentum"])
    assert weight > 0, (
        f"{regime} gives rsi_momentum weight {weight}; it measures +0.60 to +0.82 "
        f"lift on holdout at every horizon and must not be penalised"
    )


@pytest.mark.parametrize("regime", REGIMES)
def test_rsi_momentum_effective_weight_is_positive(regime):
    """Guard the merged value _compute_score actually uses, not just the raw dict.

    Mirrors the merge at src/scorer.py:_compute_score --
    short_term = {**SHORT_TERM_WEIGHTS, **{k: v for k, v in override.items()
                                           if k in SHORT_TERM_WEIGHTS}}
    """
    override = REGIME_WEIGHTS.get(regime) or {}
    merged = {**SHORT_TERM_WEIGHTS,
              **{k: v for k, v in override.items() if k in SHORT_TERM_WEIGHTS}}
    assert merged["rsi_momentum"] > 0


@pytest.mark.parametrize("regime, signal, contradicted_value", DROPPED)
def test_contradicted_override_stays_removed(regime, signal, contradicted_value):
    override = REGIME_WEIGHTS.get(regime) or {}
    assert override.get(signal) != contradicted_value, (
        f"{regime}/{signal}={contradicted_value} was pruned 2026-09-17 because the "
        f"train data contradicts its sign; do not restore without a new positive "
        f"result from scripts/regime_override_audit.py"
    )


@pytest.mark.parametrize("regime, signal, value", KEPT)
def test_validated_bearish_penalty_is_retained(regime, signal, value):
    """Do not 'simplify' by deleting the whole table -- these entries are correct."""
    override = REGIME_WEIGHTS.get(regime) or {}
    assert override.get(signal) == value, (
        f"{regime}/{signal} should be {value}: it agrees with the measured sign and "
        f"carries most of REGIME_WEIGHTS' value"
    )


def test_overrides_only_reference_live_signal_keys():
    """An override keyed on a backtest-only column name is silently inert --
    the failure mode Phase 2 and Phase 5 both had to fix."""
    live = set(SHORT_TERM_WEIGHTS) | set(DISQUALIFIER_WEIGHTS) | set(BEARISH_EVENT_WEIGHTS)
    for regime in REGIMES:
        for key in (REGIME_WEIGHTS.get(regime) or {}):
            assert key in live, f"{regime}/{key} matches no live signal-map key -- inert"
