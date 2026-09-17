"""The advertised holding period must match the exit the book actually enforces.

Before 2026-09-17 src/scorer.py hardcoded timeframe="1-2d" and 66 of 70 live
buy picks carried it, while WINNER_POLICY="time_10d" force-exited at 10 bars.
So the alert advised a hold the system never performed -- and 1-2 days is the
one horizon measured NET-NEGATIVE after the repo's 0.30% round-trip cost.

Holdout (2024-08..2026-06), buy rows at the live score>=4 emission bar,
net of cost:

    fwd_5d   -0.09%   47.0% up    <- what the alert was advising
    fwd_10d  +0.82%   50.6% up
    fwd_20d  +1.88%   52.0% up

timeframe is display-only (src/telegram_alert.py), so this was a pure
mis-instruction to the reader rather than a trading bug -- which is why it
survived so long. Deriving the label from WINNER_POLICY means it cannot drift
again.
"""

from __future__ import annotations

import pytest

from src.scorer import _compute_score
from src.trade_sim import _TIME_STOP_BARS, WINNER_POLICY, horizon_label


def test_label_matches_the_live_winner_policy():
    assert horizon_label() == horizon_label(WINNER_POLICY)


@pytest.mark.parametrize("policy, expected", [
    ("time_10d", "~10d"),
    ("time_15d", "~15d"),
    ("time_20d", "~20d"),
])
def test_time_stop_policies_advertise_their_own_bar_count(policy, expected):
    assert horizon_label(policy) == expected


def test_policy_without_a_time_stop_does_not_invent_a_deadline():
    assert horizon_label("static") == "until SL/T1"
    assert horizon_label("trail_atr3") == "until SL/T1"


def test_scorer_timeframe_is_the_policy_label_not_a_hardcoded_string():
    """The specific regression: a fixed "1-2d"/"5-7d" that ignores the exit."""
    _, timeframe = _compute_score({"rsi_momentum": True})
    assert timeframe == horizon_label(), "timeframe drifted from the exit policy"
    assert timeframe not in ("1-2d", "3-5d", "5-7d"), (
        "hardcoded short timeframe is back; 1-2 days is net-negative after costs"
    )


def test_swing_signals_no_longer_shorten_the_advertised_hold():
    """results_due/promoter_buying used to force "5-7d", which would now
    advertise a SHORTER hold than the book enforces."""
    base = _compute_score({"rsi_momentum": True})[1]
    for sig in ("results_due", "promoter_buying"):
        assert _compute_score({"rsi_momentum": True, sig: True})[1] == base


def test_every_time_stop_policy_has_a_label_and_an_exit_plan_hint():
    """A new policy added to _TIME_STOP_BARS without a hint would render as a
    blank exit plan in the Telegram alert."""
    from src.trade_sim import EXIT_PLAN_HINT

    for policy in _TIME_STOP_BARS:
        assert EXIT_PLAN_HINT.get(policy), f"{policy} has no EXIT_PLAN_HINT"
        assert horizon_label(policy).startswith("~")
