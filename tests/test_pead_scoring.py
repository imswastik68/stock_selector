"""
Tests for PEAD v2 as a first-class scoring entry (SOTA Round Phase 1).

Before this phase, pead_positive_surprise/pead_negative_surprise were only
set from `technical_data` in pass 3 (post top-20 OHLCV enrichment) -- but a
pure-PEAD ticker scores 1 (below MIN_SCORE=2), so it could never reach that
enrichment step in the first place. classify_pead_reaction is now computed
EARLY (src.data.bse_announcements.fetch_pead_signals) and threaded through
score_candidates via a dedicated `pead_signals` param, applying in every pass.
"""

from __future__ import annotations

import src.scorer as s
from main import _build_pead_watchlist


# ── _build_signal_map: pead_signal param (single source of truth) ───────────

def test_build_signal_map_positive_pead_signal():
    signals = s._build_signal_map(
        "FOO.NS", [], None, [], [], None, pead_signal="positive",
    )
    assert signals["pead_positive_surprise"] is True
    assert signals["pead_negative_surprise"] is False


def test_build_signal_map_negative_pead_signal():
    signals = s._build_signal_map(
        "FOO.NS", [], None, [], [], None, pead_signal="negative",
    )
    assert signals["pead_positive_surprise"] is False
    assert signals["pead_negative_surprise"] is True


def test_build_signal_map_no_pead_signal():
    signals = s._build_signal_map(
        "FOO.NS", [], None, [], [], None, pead_signal=None,
    )
    assert signals["pead_positive_surprise"] is False
    assert signals["pead_negative_surprise"] is False


# ── score_candidates: pead-only ticker enters the pool but doesn't qualify ───

def test_pead_only_ticker_is_evaluated_but_below_min_score():
    """pead_positive_surprise alone scores 1 (< MIN_SCORE=2) -- it must enter
    the scoring pool (no crash, no KeyError) but not qualify as a candidate.
    This is intended: PEAD alone is a watchlist signal, not an entry trigger."""
    candidates = s.score_candidates(
        [], [], [], [], [],
        pead_signals={"FOO.NS": "positive"},
    )
    assert all(c["ticker"] != "FOO.NS" for c in candidates)


def test_pead_positive_combined_with_delivery_surge_qualifies():
    """delivery_surge (weight 1) + pead_positive_surprise (weight 1) = score 2,
    meets MIN_SCORE -- the combined case IS a real, tradeable entry."""
    candidates = s.score_candidates(
        [], [], [], [], [],
        delivery_signals={"FOO.NS": {"delivery_surge": True}},
        pead_signals={"FOO.NS": "positive"},
    )
    match = next((c for c in candidates if c["ticker"] == "FOO.NS"), None)
    assert match is not None
    assert match["score"] == 2
    assert "pead_positive_surprise" in match["active_signals"]
    assert "delivery_surge" in match["active_signals"]


def test_pead_negative_is_a_disqualifier_not_a_qualifier():
    """pead_negative_surprise alone (weight -2) must not produce a candidate."""
    candidates = s.score_candidates(
        [], [], [], [], [],
        pead_signals={"BAR.NS": "negative"},
    )
    assert all(c["ticker"] != "BAR.NS" for c in candidates)


def test_pead_negative_demotes_an_otherwise_qualifying_ticker():
    """delivery_surge (1) + pead_negative_surprise (-2) = score -1 -> clamped
    out of qualification even though delivery_surge alone would not qualify
    either, but the point is the disqualifier correctly SUBTRACTS."""
    candidates_without_pead = s.score_candidates(
        [], [], [], [], [],
        delivery_signals={"BAZ.NS": {"delivery_surge": True}},
    )
    candidates_with_pead = s.score_candidates(
        [], [], [], [], [],
        delivery_signals={"BAZ.NS": {"delivery_surge": True}},
        pead_signals={"BAZ.NS": "negative"},
    )
    # neither qualifies (delivery_surge alone = 1 < MIN_SCORE), but confirms
    # no crash and the disqualifier doesn't accidentally inflate the score
    assert all(c["ticker"] != "BAZ.NS" for c in candidates_without_pead)
    assert all(c["ticker"] != "BAZ.NS" for c in candidates_with_pead)


def test_no_pead_signals_does_not_crash():
    candidates = s.score_candidates([], [], [], [], [])
    assert candidates == []


# ── main._build_pead_watchlist ───────────────────────────────────────────────

def test_build_pead_watchlist_empty_when_no_signals():
    assert _build_pead_watchlist({}, [], exclude=set()) == []


def test_build_pead_watchlist_excludes_already_listed_tickers():
    pead_signals = {"FOO.NS": "positive", "BAR.NS": "negative"}
    out = _build_pead_watchlist(pead_signals, [], exclude={"FOO.NS"})
    tickers = {e["ticker"] for e in out}
    assert tickers == {"BAR.NS"}


def test_build_pead_watchlist_attaches_headline_and_filed_at():
    pead_signals = {"FOO.NS": "positive"}
    announcements = [{"ticker": "FOO.NS", "signal_key": "results_beat_announced",
                       "headline": "Q3 results beat estimates", "filed_at": "2026-01-05T10:00:00+05:30"}]
    out = _build_pead_watchlist(pead_signals, announcements, exclude=set())
    assert len(out) == 1
    assert out[0]["ticker"] == "FOO.NS"
    assert out[0]["direction"] == "positive"
    assert out[0]["headline"] == "Q3 results beat estimates"
    assert out[0]["filed_at"] == "2026-01-05T10:00:00+05:30"


def test_build_pead_watchlist_no_crash_without_matching_announcement():
    """A pead_signals entry with no corresponding announcement item (edge
    case, e.g. cache mismatch) must still render with empty headline/filed_at."""
    out = _build_pead_watchlist({"FOO.NS": "negative"}, [], exclude=set())
    assert out == [{"ticker": "FOO.NS", "direction": "negative", "headline": "", "filed_at": ""}]
