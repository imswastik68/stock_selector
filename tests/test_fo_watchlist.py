"""
F&O SWING watchlist — derivatives-eligible setups the user can actually express
as an option or a future.

WHY IT EXISTS: only 11 of 71 picks (15%) across 2026-05/07 were F&O-listed, and
four scan days produced ZERO. The scanner hunts ~2,369 tickers, mostly SME and
microcaps with no derivatives, so a derivatives trader got nothing most days.

The subtle part, and what these tests pin: F&O-eligible names are NOT
concentrated at the top of the score ranking, so the ordinary top-20 candidate
slice usually excluded them entirely and the section came back empty -- the
exact failure it was built to fix. The pool must reach past the cut for them,
and those entries must still carry real technicals/levels or they aren't
tradeable.
"""

from __future__ import annotations

import pytest

from src.agent import _build_entries, MAX_FO_WATCHLIST, FO_ENRICH_EXTRA


def _ctx(n: int, fo: set[str]) -> dict:
    return {
        "technicals": {
            f"T{i}.NS": {"wyckoff_phase": "MARKUP", "direction": "buy",
                         "today_close": 100.0, "rsi": 60}
            for i in range(n)
        },
        "atr_pct": {f"T{i}.NS": 2.0 for i in range(n)},
        "beta":    {f"T{i}.NS": 1.0 for i in range(n)},
        "fo_eligible": fo,
    }


def _cands(n: int) -> list[dict]:
    # Descending score, so T0 is the strongest and T24 the weakest.
    # today_close mirrors src/scorer.py, which sets it on every candidate.
    return [{"ticker": f"T{i}.NS", "score": 30 - i, "active_signals": [],
             "options_pcr": 0.9, "today_close": 100.0} for i in range(n)]


def _ctx_watch_grade(n: int, fo: set[str], n_actionable: int = 5) -> dict:
    """The REAL shape of a scan: a few actionable MARKUP names at the top, and
    everything else watch-grade (direction='watch'). On 2026-07-22, 10 of the 11
    F&O-eligible qualifying candidates were watch-grade."""
    tech = {}
    for i in range(n):
        if i < n_actionable:
            tech[f"T{i}.NS"] = {"wyckoff_phase": "MARKUP", "direction": "buy",
                                "today_close": 100.0, "rsi": 60}
        else:
            tech[f"T{i}.NS"] = {"wyckoff_phase": "ACCUMULATION_B", "direction": "watch",
                                "today_close": 100.0, "rsi": 50}
    return {
        "technicals": tech,
        "atr_pct": {f"T{i}.NS": 2.0 for i in range(n)},
        "beta":    {f"T{i}.NS": 1.0 for i in range(n)},
        "fo_eligible": fo,
    }


def test_watch_grade_fo_names_still_populate_the_section():
    """Regression for the bug that shipped an empty F&O section on a day with 11
    tradeable candidates.

    Watch-grade candidates hit an early `continue` in _build_entries and used to
    never reach all_entries, which the F&O list draws from. Since F&O-eligible
    names are overwhelmingly sub-threshold, that made the section empty on
    essentially every real day -- while passing synthetic tests that gave every
    candidate a MARKUP phase.
    """
    fo = {"T6.NS", "T7.NS", "T8.NS", "T9.NS"}
    _, _, _, fo_list = _build_entries(_cands(25), _ctx_watch_grade(25, fo), "uptrend")

    assert fo_list, "F&O section empty despite watch-grade F&O candidates existing"
    assert all(e["ticker"] in fo for e in fo_list)


def test_actionable_fo_names_sort_above_watch_grade_ones():
    """An entry carrying real levels is more useful than one that doesn't."""
    # T5 is actionable AND F&O; T8/T9 are watch-grade F&O.
    fo = {"T5.NS", "T8.NS", "T9.NS"}
    _, _, _, fo_list = _build_entries(
        _cands(25), _ctx_watch_grade(25, fo, n_actionable=6), "uptrend")

    assert fo_list
    flags = [e.get("actionable", False) for e in fo_list]
    assert flags == sorted(flags, reverse=True)


def test_fo_names_below_the_top20_cut_still_surface():
    """The regression that matters: F&O names ranked 20+ used to be dropped
    before entries were even built, leaving the section permanently empty."""
    fo = {"T20.NS", "T21.NS", "T22.NS", "T23.NS"}
    _, _, _, fo_list = _build_entries(_cands(25), _ctx(25, fo), "uptrend")

    assert len(fo_list) == MAX_FO_WATCHLIST
    assert all(e["ticker"] in fo for e in fo_list)


def test_fo_entries_carry_tradeable_levels():
    """An F&O pick with no stop/target isn't actionable."""
    fo = {"T20.NS", "T21.NS", "T22.NS"}
    _, _, _, fo_list = _build_entries(_cands(25), _ctx(25, fo), "uptrend")

    assert fo_list
    for e in fo_list:
        assert e.get("stop_loss") and e.get("target_1")
        assert e.get("today_close")


def test_fo_list_does_not_repeat_names_already_shown():
    """T0/T1 are top-scoring AND F&O-eligible -- they belong in the buy list,
    tagged, not duplicated into a second section."""
    fo = {"T0.NS", "T1.NS", "T20.NS", "T21.NS", "T22.NS"}
    buy, sell, phase_b, fo_list = _build_entries(_cands(25), _ctx(25, fo), "uptrend")

    shown = {e["ticker"] for e in buy + sell + phase_b}
    assert not (shown & {e["ticker"] for e in fo_list})
    assert "T0.NS" in shown  # still surfaced, just not in the F&O section


def test_fo_list_is_score_ordered_and_capped():
    fo = {f"T{i}.NS" for i in range(20, 25)}
    _, _, _, fo_list = _build_entries(_cands(25), _ctx(25, fo), "uptrend")

    scores = [e["score"] for e in fo_list]
    assert scores == sorted(scores, reverse=True)
    assert len(fo_list) <= MAX_FO_WATCHLIST


def test_every_entry_is_tagged_with_fo_eligibility():
    """The main lists mark their own F&O names, so the tag must be on all
    entries -- not only the ones in the F&O section."""
    fo = {"T0.NS"}
    buy, _, _, _ = _build_entries(_cands(25), _ctx(25, fo), "uptrend")

    by_ticker = {e["ticker"]: e for e in buy}
    assert by_ticker["T0.NS"]["fo_eligible"] is True
    assert by_ticker["T1.NS"]["fo_eligible"] is False


def test_empty_fo_universe_yields_empty_section_not_a_crash():
    """fetch_fo_eligible failing must degrade to 'no F&O picks', not an error
    or a list of untradeable names."""
    _, _, _, fo_list = _build_entries(_cands(25), _ctx(25, set()), "uptrend")
    assert fo_list == []


def test_fo_pool_extension_is_bounded():
    """Reaching past the top-20 cut must not pull in an unbounded tail."""
    fo = {f"T{i}.NS" for i in range(20, 60)}
    _, _, _, fo_list = _build_entries(_cands(60), _ctx(60, fo), "uptrend")

    assert len(fo_list) <= MAX_FO_WATCHLIST
    assert FO_ENRICH_EXTRA < 60  # the pool extension is a bounded slice
