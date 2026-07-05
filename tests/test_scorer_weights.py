"""
Regression tests for the Phase 2 strict-validated-only scorer pass:
- every signal in PENDING_VALIDATION weighs 0 in whichever home dict it lives in
  (SHORT_TERM_WEIGHTS or SWING_WEIGHTS)
- rs_quality_strong contributes 0 under every nifty_trend regime (it FAILED its
  backtest, not pending -- and was removed from all 3 REGIME_WEIGHTS buckets,
  so the regime merge must not resurrect it)
- near_52w_high is scored (the best validated signal, previously computed but
  dropped) and stacks correctly with actual_52w_breakout
- the funnel doesn't crash when a candidate only carries untested signals
  (score collapses to 0, candidate is dropped -- this is intended behavior)
"""

from __future__ import annotations

import src.scorer as s


def test_every_pending_validation_signal_weighs_zero():
    for sig in s.PENDING_VALIDATION:
        weight = s.SHORT_TERM_WEIGHTS.get(sig, s.SWING_WEIGHTS.get(sig))
        assert weight is not None, f"{sig} not found in SHORT_TERM_WEIGHTS or SWING_WEIGHTS"
        assert weight == 0, f"{sig} should be zeroed pending validation, got {weight}"


def test_sast_and_promoter_alone_score_zero():
    signals = {k: False for k in
               list(s.SHORT_TERM_WEIGHTS) + list(s.SWING_WEIGHTS)
               + list(s.DISQUALIFIER_WEIGHTS) + list(s.BEARISH_EVENT_WEIGHTS)}
    signals["sast_insider_buying"] = True
    signals["promoter_buying"] = True
    score, _ = s._compute_score(signals)
    assert score == 0


def test_rs_quality_strong_contributes_zero_under_every_regime():
    signals = {k: False for k in
               list(s.SHORT_TERM_WEIGHTS) + list(s.SWING_WEIGHTS)
               + list(s.DISQUALIFIER_WEIGHTS) + list(s.BEARISH_EVENT_WEIGHTS)}
    signals["rs_quality_strong"] = True
    for trend in ("uptrend", "ranging", "downtrend"):
        score, _ = s._compute_score(signals, nifty_trend=trend)
        assert score == 0, f"rs_quality_strong leaked a nonzero score under {trend}"


def test_rs_quality_strong_removed_from_all_regime_weight_buckets():
    for trend, override in s.REGIME_WEIGHTS.items():
        assert "rs_quality_strong" not in (override or {}), \
            f"rs_quality_strong still present in REGIME_WEIGHTS[{trend}]"


def test_near_52w_high_scores_three():
    signals = {k: False for k in
               list(s.SHORT_TERM_WEIGHTS) + list(s.SWING_WEIGHTS)
               + list(s.DISQUALIFIER_WEIGHTS) + list(s.BEARISH_EVENT_WEIGHTS)}
    signals["near_52w_high"] = True
    score, _ = s._compute_score(signals)
    assert score == 3


def test_near_52w_high_stacks_with_actual_breakout():
    signals = {k: False for k in
               list(s.SHORT_TERM_WEIGHTS) + list(s.SWING_WEIGHTS)
               + list(s.DISQUALIFIER_WEIGHTS) + list(s.BEARISH_EVENT_WEIGHTS)}
    signals["near_52w_high"] = True
    signals["actual_52w_breakout"] = True
    score, _ = s._compute_score(signals)
    assert score == 6


def test_build_signal_map_sets_near_52w_high_from_breakout_data():
    signals = s._build_signal_map(
        "FOO.NS", bulk_deals=[], volume_data=None, fo_ban_removed=[],
        results_calendar=[], breakout_data={"actual_breakout": False, "today_close": 100},
    )
    assert signals["near_52w_high"] is True
    assert signals["actual_52w_breakout"] is False

    signals2 = s._build_signal_map(
        "FOO.NS", bulk_deals=[], volume_data=None, fo_ban_removed=[],
        results_calendar=[], breakout_data=None,
    )
    assert signals2["near_52w_high"] is False


def test_score_candidates_bulk_deals_only_returns_empty_without_raising():
    """A candidate that only carries an (untested, zeroed) bulk-deal signal must
    score 0 and drop before qualifying -- not raise. This is the funnel-collapse
    behavior Phase 2 intends: the scanner only acts on proven signals now."""
    bulk_deals = [{"ticker": "X.NS", "is_fii_dii": True, "side": ""}]
    candidates = s.score_candidates(bulk_deals, [], [], [], [])
    assert candidates == []


def test_delivery_surge_promoted_out_of_pending_validation():
    """Phase 7: delivery_surge SHIPPED (scripts/backtest_events.py, n=38279,
    ret_lift=0.316, 70/30 holdout sign-consistent) and was promoted to a
    real weight -- it must no longer sit in PENDING_VALIDATION (which would
    make test_every_pending_validation_signal_weighs_zero force it back to
    0), and must score its promoted weight when active."""
    assert "delivery_surge" not in s.PENDING_VALIDATION
    signals = {k: False for k in
               list(s.SHORT_TERM_WEIGHTS) + list(s.SWING_WEIGHTS)
               + list(s.DISQUALIFIER_WEIGHTS) + list(s.BEARISH_EVENT_WEIGHTS)}
    signals["delivery_surge"] = True
    score, _ = s._compute_score(signals)
    assert score == s.SHORT_TERM_WEIGHTS["delivery_surge"]
    assert score > 0


def test_score_candidates_near_52w_high_qualifies():
    breakouts = [{"ticker": "Y.NS", "today_close": 100.0, "volume_ratio": 1.5,
                  "actual_breakout": False}]
    candidates = s.score_candidates([], [], [], [], breakouts)
    assert len(candidates) == 1
    assert candidates[0]["ticker"] == "Y.NS"
    assert candidates[0]["score"] == 3


def test_results_beat_announced_is_now_a_disqualifier():
    """SOTA Round Phase 3: moved from SWING_WEIGHTS(0) to DISQUALIFIER_WEIGHTS(-3)
    -- the 260-week rerun found it sign-consistently negative (train -0.979,
    holdout -1.45, n=13915), not just an unproven buy signal."""
    assert "results_beat_announced" not in s.SWING_WEIGHTS
    assert "results_beat_announced" not in s.PENDING_VALIDATION
    assert s.DISQUALIFIER_WEIGHTS["results_beat_announced"] == -3
    signals = {k: False for k in
               list(s.SHORT_TERM_WEIGHTS) + list(s.SWING_WEIGHTS)
               + list(s.DISQUALIFIER_WEIGHTS) + list(s.BEARISH_EVENT_WEIGHTS)}
    signals["results_beat_announced"] = True
    score, _ = s._compute_score(signals)
    assert score == -3


def test_dividend_announced_is_now_a_disqualifier():
    """Same reclassification as results_beat_announced -- n=8291, train -0.708,
    holdout -0.158, both negative."""
    assert "dividend_announced" not in s.SWING_WEIGHTS
    assert "dividend_announced" not in s.PENDING_VALIDATION
    assert s.DISQUALIFIER_WEIGHTS["dividend_announced"] == -2
    signals = {k: False for k in
               list(s.SHORT_TERM_WEIGHTS) + list(s.SWING_WEIGHTS)
               + list(s.DISQUALIFIER_WEIGHTS) + list(s.BEARISH_EVENT_WEIGHTS)}
    signals["dividend_announced"] = True
    score, _ = s._compute_score(signals)
    assert score == -2


def test_buyback_and_contract_win_remain_unweighted_after_sign_flip():
    """Both crossed n>=500 on the 260-week rerun but showed a train/holdout
    sign-flip (inconclusive, not a stable effect) -- must stay at weight 0,
    not be promoted either direction."""
    assert s.SWING_WEIGHTS["buyback_announced"] == 0
    assert s.SWING_WEIGHTS["contract_win"] == 0
