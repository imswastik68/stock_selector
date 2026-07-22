"""
Regression test for the audit fix to src/agent.py's direction-aware signal
cleanup (_cleanup_weight). Phase 5 made src/scorer.py apply REGIME-MERGED
weights, but the cleanup originally subtracted STATIC base weights --
double-counting a signal that was already a heavy regime penalty, and
under-cancelling one boosted into a large regime reward. See the commit
"Post-overhaul audit: fix 5 bugs..." for the full writeup.

These 5 cases were verified by hand during that fix; codifying them here so
a future change to REGIME_WEIGHTS or _cleanup_weight can't silently
reintroduce the double-count / under-cancel bug.
"""

from __future__ import annotations

from src.agent import _cleanup_weight, BULLISH_ONLY_WEIGHTS, _BEARISH_EVENT_PRE_PHASE3_DISQUALIFIER
from src.scorer import BEARISH_EVENT_WEIGHTS, REGIME_WEIGHTS, _compute_score


def _net_score(sig: str, direction: str, nifty_trend: str) -> int:
    """What a lone-signal candidate's score nets to after scorer + cleanup,
    exactly as src/agent.py:_build_entries computes it."""
    override = REGIME_WEIGHTS.get(nifty_trend) or {}
    score, _ = _compute_score({sig: True}, nifty_trend=nifty_trend if nifty_trend in REGIME_WEIGHTS else "ranging")
    if nifty_trend not in REGIME_WEIGHTS:
        # unrecognized regime -> _compute_score also falls back to an empty override
        score, _ = _compute_score({sig: True}, nifty_trend=nifty_trend)
    if direction == "buy":
        cleanup = _cleanup_weight(sig, BEARISH_EVENT_WEIGHTS, override, _BEARISH_EVENT_PRE_PHASE3_DISQUALIFIER)
    else:
        cleanup = _cleanup_weight(sig, BULLISH_ONLY_WEIGHTS, override)
    return score - cleanup


def test_downtrend_buy_distribution_signal_nets_to_regime_penalty():
    """Scorer applies the downtrend-regime penalty for distribution_signal
    (-5); cleanup must NOT additionally subtract the pre-Phase-3 disqualifier
    on top of that (the old bug: -5 scorer, -1 cleanup = -6, double-counted)."""
    assert _net_score("distribution_signal", "buy", "downtrend") == -5


def test_unrecognized_regime_buy_distribution_signal_falls_back_to_pre_phase3():
    """With no regime override active, distribution_signal must net to -1 --
    exactly the pre-Phase-3 DISQUALIFIER_WEIGHTS behavior (scorer 0 + cleanup -1)."""
    assert _net_score("distribution_signal", "buy", "unknown_regime") == -1


def test_untouched_signal_options_short_buildup_nets_pre_phase3_value():
    """options_short_buildup has no REGIME_WEIGHTS override in any regime --
    must always net to -2 (its pre-Phase-3 DISQUALIFIER_WEIGHTS value)."""
    assert _net_score("options_short_buildup", "buy", "downtrend") == -2
    assert _net_score("options_short_buildup", "buy", "uptrend") == -2


def test_new_bearish_signal_with_no_pre_phase3_history_nets_zero():
    """consolidation_breakdown is new in Phase 3 (no prior disqualifier
    magnitude to restore) -- should net to a neutral 0 on a buy candidate."""
    assert _net_score("consolidation_breakdown", "buy", "downtrend") == 0


def test_downtrend_sell_actual_52w_breakout_nets_to_zero():
    """Scorer applies the downtrend-regime BUY reward for actual_52w_breakout
    (+5) universally (direction isn't known at scoring time); on a SELL
    candidate this is a pure contradiction and must cancel to exactly 0, not
    leave a positive bullish leftover (the old bug: +5 scorer, only -3
    static cleanup = +2 leftover reward on a short)."""
    assert _net_score("actual_52w_breakout", "sell", "downtrend") == 0


# ── Live-Proof Round Phase 2: active_signals flows onto the entry ───────────

def test_build_entries_carries_active_signals_onto_buy_entry():
    """The entry dict src.performance.record_picks reads from must carry the
    full active_signals list, not just the top-3 labeled subset (top_signals)."""
    from src.agent import _build_entries

    candidates = [{
        "ticker": "PROOF.NS", "score": 6, "active_signals": ["near_52w_high", "reversal_oversold_v2"],
        "volume_ratio": 1.5, "today_close": 100.0,
    }]
    market_context = {
        "technicals": {"PROOF.NS": {"wyckoff_phase": "MARKUP", "direction": "buy"}},
    }
    buy_list, sell_list, phase_b_list, _ = _build_entries(candidates, market_context, "ranging")

    assert len(buy_list) == 1
    assert buy_list[0]["active_signals"] == ["near_52w_high", "reversal_oversold_v2"]
