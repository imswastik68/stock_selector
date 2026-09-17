"""record_picks must persist `score` alongside the other write-once proof inputs.

Measured 2026-09-17: all 285 picks in outputs/performance.json carried
score: None, because record_picks stored active_signals/regime/big_mover/
instrument but never score. Live "does a higher score actually predict a better
outcome" could not be answered from the audit trail at all -- the only reason
scripts/analyse_picks.py could speak to it is that it re-parses the score back
out of the Telegram message text, which is a fragile side channel, not a record.

score is the scorer's single output and the thing every ranking/truncation
decision (MAX_ACTIONABLE, portfolio_summary's sort, CONVICTION_SIZING tiers)
depends on, so not recording it left the system's central claim unfalsifiable.
"""

from __future__ import annotations

import json

import pytest

import src.performance as performance


@pytest.fixture
def perf_file(tmp_path, monkeypatch):
    f = tmp_path / "performance.json"
    monkeypatch.setattr(performance, "_PERF_FILE", f)
    return f


def _watchlist(**entry_overrides) -> dict:
    entry = {
        "ticker": "TEST.NS",
        "score": 7,
        "today_close": 100.0,
        "entry_zone": "₹99.0-₹101.0",
        "stop_loss": "₹95.0",
        "target_1": "₹110.0",
        "target_2": "₹120.0",
        "active_signals": ["rsi_momentum"],
        **entry_overrides,
    }
    return {"scan_date": "2026-09-17", "nifty_context": "ranging",
            "buy_watchlist": [entry], "sell_watchlist": []}


def test_score_is_recorded(perf_file):
    performance.record_picks(_watchlist(), nifty_at_emission=25000.0)
    pick = json.loads(perf_file.read_text())["2026-09-17"]["TEST.NS"]
    assert pick["score"] == 7, "score must be persisted, not dropped"


def test_missing_score_records_none_rather_than_raising(perf_file):
    """A candidate without a score must not break the EOD run."""
    wl = _watchlist()
    del wl["buy_watchlist"][0]["score"]
    performance.record_picks(wl, nifty_at_emission=25000.0)
    pick = json.loads(perf_file.read_text())["2026-09-17"]["TEST.NS"]
    assert pick["score"] is None


def test_score_zero_is_preserved_not_coerced(perf_file):
    """0 is a real score; `entry.get("score") or None` would silently lose it."""
    performance.record_picks(_watchlist(score=0), nifty_at_emission=25000.0)
    pick = json.loads(perf_file.read_text())["2026-09-17"]["TEST.NS"]
    assert pick["score"] == 0
