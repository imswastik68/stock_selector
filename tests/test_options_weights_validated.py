"""
Every options signal's live weight, pinned to the evidence that justifies it.

Backtested 2026-07-22 via scripts/backtest_events.py --source options over 156
weeks of F&O bhavcopy (617 cached sessions). Until then three of these carried
live non-zero weights that had NEVER been tested -- they were unreachable while
the option-chain fetch was broken, so nothing surfaced them. Rebuilding the
fetch on the bhavcopy made validating them a prerequisite.

Reading convention (same as pead_negative_surprise): every signal is evaluated
BUY-side, so a DISQUALIFIER is justified by a NEGATIVE ret_lift -- it marks
names that go on to underperform. The harness's stored holdout_consistent flag
checks positive-direction consistency only, so it reads False for a correct
disqualifier; the sign-aware reconciliation is done by hand, as in the scorer.

  options_pcr_greed       n=21,697  ret_lift -0.361  train -0.343 / holdout -0.401
                          -> both negative and strengthening: penalty JUSTIFIED,
                             magnitude round(3 x 0.361) = 1, already live at -1.
  options_short_buildup   n=40,215  ret_lift -0.356  train -0.469 / holdout -0.093
                          -> both negative: correct as a BEARISH (sell-side) signal.
  options_long_unwinding  n=17,827  ret_lift +0.193  train -0.019 / holdout +0.686
                          -> POSITIVE aggregate: the -1 penalty was BACKWARDS.
                             Sign-flipped across the split, so not a buy signal
                             either. Zeroed.
  options_pcr_fear        n=125     INSUFFICIENT_SAMPLE (PCR > 1.5 is rare)
  options_long_buildup    n=36,317  ret_lift -0.328  NO-SHIP
  options_short_covering  n=22,412  ret_lift -0.276  NO-SHIP
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scorer import (
    BEARISH_EVENT_WEIGHTS,
    DISQUALIFIER_WEIGHTS,
    SHORT_TERM_WEIGHTS,
)

_BACKTEST = Path(__file__).parent.parent / "outputs" / "event_backtest.json"


def _signals() -> dict:
    return json.loads(_BACKTEST.read_text())["signals"]


def test_long_unwinding_is_zero_not_a_backwards_penalty():
    """The regression that matters: a -1 here is contradicted by n=17,827."""
    assert DISQUALIFIER_WEIGHTS["options_long_unwinding"] == 0


def test_long_unwinding_evidence_still_contradicts_a_penalty():
    """If a future rerun ever makes the aggregate lift negative, this test should
    fail so the weight gets reconsidered on the new evidence rather than left at
    0 by inertia."""
    sig = _signals().get("options_long_unwinding")
    if sig is None:
        pytest.skip("options source not present in this backtest run")
    assert sig["ret_lift"] > 0, (
        "aggregate lift is no longer positive -- revisit the zeroed weight"
    )


def test_pcr_greed_penalty_matches_its_measured_magnitude():
    """round(3 x |ret_lift|) is the repo-wide convention for weight magnitude."""
    sig = _signals().get("options_pcr_greed")
    if sig is None:
        pytest.skip("options source not present in this backtest run")

    assert sig["ret_lift"] < 0, "a buy-side disqualifier needs a negative lift"
    expected = -round(3 * abs(sig["ret_lift"]))
    assert DISQUALIFIER_WEIGHTS["options_pcr_greed"] == expected


def test_pcr_greed_is_sign_consistent_across_the_holdout():
    sig = _signals().get("options_pcr_greed")
    if sig is None:
        pytest.skip("options source not present in this backtest run")
    assert sig["train_ret_lift"] < 0 and sig["holdout_ret_lift"] < 0


def test_short_buildup_is_sign_consistent_as_a_bearish_signal():
    sig = _signals().get("options_short_buildup")
    if sig is None:
        pytest.skip("options source not present in this backtest run")

    assert sig["train_ret_lift"] < 0 and sig["holdout_ret_lift"] < 0
    assert BEARISH_EVENT_WEIGHTS["options_short_buildup"] > 0


def test_no_ship_buy_side_options_signals_stay_at_zero():
    """pcr_fear (INSUFFICIENT), long_buildup and short_covering (both NO-SHIP with
    negative lift) must not carry positive weight."""
    for sig in ("options_pcr_fear", "options_long_buildup", "options_short_covering"):
        assert SHORT_TERM_WEIGHTS.get(sig, 0) == 0


def test_every_options_signal_has_a_recorded_verdict():
    """No options signal should carry a live weight without a backtest entry --
    that combination is what produced the backwards long_unwinding penalty."""
    sigs = _signals()
    if not any(k.startswith("options_") for k in sigs):
        pytest.skip("options source not present in this backtest run")

    live = {k: v for tbl in (SHORT_TERM_WEIGHTS, DISQUALIFIER_WEIGHTS, BEARISH_EVENT_WEIGHTS)
            for k, v in tbl.items() if k.startswith("options_") and v != 0}
    for name in live:
        assert name in sigs, f"{name} carries weight {live[name]} but was never backtested"
