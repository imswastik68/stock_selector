"""
Tests for src/data/insider.py -- the live promoter_open_mkt_buy detector
(SOTA Round Phase 3). PIT_ACQ_MODE/PIT_PERSON_CATEGORIES/PIT_MIN_VALUE and
_matches_filter are the single source of truth, imported by
scripts/backtest_events.py's collect_pit_events -- these tests confirm the
live filter can't silently diverge from what was backtested.
"""

from __future__ import annotations

from datetime import date

import backtest_events as be
import src.cache as cache
import src.data.insider as ins


def _pit_row(personCategory="Promoters", tdpTransactionType="Buy",
             acqMode="Market Purchase", secVal="20000000"):
    return {"symbol": "FOO", "personCategory": personCategory,
            "tdpTransactionType": tdpTransactionType, "acqMode": acqMode, "secVal": secVal,
            "date": "10-Jan-2026 18:00"}


# ── _matches_filter ───────────────────────────────────────────────────────────

def test_matches_filter_fires_on_genuine_promoter_market_purchase():
    assert ins._matches_filter(_pit_row()) is True


def test_matches_filter_excludes_sell():
    assert ins._matches_filter(_pit_row(tdpTransactionType="Sell")) is False


def test_matches_filter_excludes_non_promoter():
    assert ins._matches_filter(_pit_row(personCategory="Employees/Designated Employees")) is False


def test_matches_filter_excludes_off_market_esop_pledge():
    """acqMode must be the EXACT string 'Market Purchase' -- 'Off Market'
    also contains the substring 'Market'."""
    for mode in ("Off Market", "ESOP", "Pledge Creation", "Preferential Offer", "-"):
        assert ins._matches_filter(_pit_row(acqMode=mode)) is False, f"acqMode={mode!r} must not fire"


def test_matches_filter_excludes_below_value_floor():
    assert ins._matches_filter(_pit_row(secVal="5000000")) is False  # Rs 0.5cr


# ── fetch_promoter_open_mkt_buys ─────────────────────────────────────────────
# fetch_promoter_open_mkt_buys uses src.cache's same-day cache (breakouts.py
# pattern) -- every test disables it (load_today -> None, save_today -> no-op)
# so tests hit the mocked network path deterministically instead of racing a
# real on-disk cache file shared across test runs within the same day.

def _disable_cache(monkeypatch):
    monkeypatch.setattr(cache, "load_today", lambda key: None)
    monkeypatch.setattr(cache, "load_latest", lambda key: None)
    monkeypatch.setattr(cache, "save_today", lambda key, data: None)


def test_fetch_promoter_open_mkt_buys_filters_and_dedupes(monkeypatch):
    _disable_cache(monkeypatch)
    rows = [_pit_row(), _pit_row()]  # exact duplicate -> should dedupe to 1
    monkeypatch.setattr(ins, "_make_www_session", lambda: object())
    monkeypatch.setattr(ins, "_www_get_json", lambda session, url, **k: rows)
    out = ins.fetch_promoter_open_mkt_buys()
    assert len(out) == 1
    assert out[0]["ticker"] == "FOO.NS"


def test_fetch_promoter_open_mkt_buys_api_unreachable_returns_empty(monkeypatch):
    _disable_cache(monkeypatch)
    monkeypatch.setattr(ins, "_make_www_session", lambda: object())
    monkeypatch.setattr(ins, "_www_get_json", lambda session, url, **k: None)
    assert ins.fetch_promoter_open_mkt_buys() == []


def test_fetch_promoter_open_mkt_buys_filters_out_non_matching_rows(monkeypatch):
    _disable_cache(monkeypatch)
    rows = [_pit_row(tdpTransactionType="Sell"), _pit_row(acqMode="Off Market")]
    monkeypatch.setattr(ins, "_make_www_session", lambda: object())
    monkeypatch.setattr(ins, "_www_get_json", lambda session, url, **k: rows)
    assert ins.fetch_promoter_open_mkt_buys() == []


def test_fetch_promoter_open_mkt_buys_uses_same_day_cache(monkeypatch):
    monkeypatch.setattr(cache, "load_today", lambda key: [{"ticker": "CACHED.NS"}])

    def _boom(*a, **k):
        raise AssertionError("should not hit the network when cache hits")
    monkeypatch.setattr(ins, "_make_www_session", _boom)
    out = ins.fetch_promoter_open_mkt_buys()
    assert out == [{"ticker": "CACHED.NS"}]


# ── backtest<->live parity: the shared filter function ───────────────────────

def test_backtest_imports_the_same_filter_function():
    """scripts/backtest_events.py must import (not reimplement) _matches_filter."""
    assert be._pit_matches_filter is ins._matches_filter


def test_backtest_collect_pit_events_uses_shared_filter(monkeypatch):
    """End-to-end: a row that fails the live filter must also be excluded by
    the backtest collector, via the SAME function."""
    rows = [_pit_row(), _pit_row(acqMode="Off Market", secVal="99999999")]
    monkeypatch.setattr(be, "_fetch_pit_items", lambda weeks: (rows, None))
    events, reason = be.collect_pit_events(weeks=156, universe=["FOO.NS"])
    assert reason is None
    assert events == [(date(2026, 1, 10), "FOO.NS")]


# ── score_candidates integration (weight=3, promoted SOTA Round Phase 3) ─────

import src.scorer as s


def test_promoter_open_mkt_signal_alone_qualifies_as_a_candidate():
    """weight=3 clears MIN_SCORE=2 on its own -- a standalone entry trigger."""
    candidates = s.score_candidates(
        [], [], [], [], [],
        promoter_open_mkt_signals=[{"ticker": "FOO.NS", "intimation_date": "2026-01-10", "value_rs": 2e7}],
    )
    match = next((c for c in candidates if c["ticker"] == "FOO.NS"), None)
    assert match is not None
    assert match["score"] == 3
    assert "promoter_open_mkt_buy" in match["active_signals"]


def test_promoter_open_mkt_signal_absent_ticker_not_scored():
    candidates = s.score_candidates([], [], [], [], [], promoter_open_mkt_signals=[])
    assert candidates == []


def test_build_signal_map_promoter_open_mkt_data_presence_is_the_signal():
    signals = s._build_signal_map(
        "FOO.NS", [], None, [], [], None, promoter_open_mkt_data={"ticker": "FOO.NS"},
    )
    assert signals["promoter_open_mkt_buy"] is True


def test_build_signal_map_no_promoter_open_mkt_data():
    signals = s._build_signal_map(
        "FOO.NS", [], None, [], [], None, promoter_open_mkt_data=None,
    )
    assert signals["promoter_open_mkt_buy"] is False
