"""
Regression tests for the PEAD watchlist section in the EOD Telegram message
(src/telegram_alert._build_message). Must render on the has-picks path, on a
zero-qualifying-candidates-but-has-PEAD day (a real day type -- pead_positive_
surprise alone scores 1, below MIN_SCORE=2), and must not break the fully
empty zero-candidate path.
"""

from __future__ import annotations

import src.telegram_alert as t

_PEAD_LIST = [
    {"ticker": "FOO.NS", "direction": "positive", "headline": "Q3 results beat estimates", "filed_at": ""},
    {"ticker": "BAR.NS", "direction": "negative", "headline": "Q3 results miss estimates", "filed_at": ""},
]


def test_pead_section_renders_with_picks():
    data = {
        "scan_date": "2026-07-05", "total_screened": 300, "nifty_context": "ranging",
        "buy_watchlist": [{"ticker": "X.NS", "score": 5, "today_close": 100.0,
                            "stop_loss": "₹90", "target_1": "₹120", "target_2": "₹130"}],
        "sell_watchlist": [], "phase_b_watchlist": [], "pead_watchlist": _PEAD_LIST,
    }
    msg = t._build_message(data)
    assert "PEAD WATCH" in msg
    assert "FOO.NS" in msg
    assert "BAR.NS" in msg


def test_pead_only_day_renders_pead_section_not_terse_message():
    """Zero qualifying candidates but PEAD signals exist -- must show the
    PEAD section, not the generic 'No qualifying candidates' terse message
    that would otherwise hide real signal."""
    data = {
        "scan_date": "2026-07-05", "total_screened": 300, "nifty_context": "ranging",
        "buy_watchlist": [], "sell_watchlist": [], "phase_b_watchlist": [],
        "pead_watchlist": _PEAD_LIST,
    }
    msg = t._build_message(data)
    assert "No qualifying candidates" in msg  # still states the screening fact
    assert "PEAD WATCH" in msg
    assert "FOO.NS" in msg


def test_fully_empty_day_has_no_pead_section():
    data = {
        "scan_date": "2026-07-05", "total_screened": 300, "nifty_context": "ranging",
        "buy_watchlist": [], "sell_watchlist": [], "phase_b_watchlist": [], "pead_watchlist": [],
    }
    msg = t._build_message(data)
    assert "No qualifying candidates" in msg
    assert "PEAD WATCH" not in msg


def test_missing_pead_watchlist_key_does_not_crash():
    """Older-format watchlist_data without the key at all (pre-Phase-1) must
    still render safely."""
    data = {
        "scan_date": "2026-07-05", "total_screened": 300, "nifty_context": "ranging",
        "buy_watchlist": [], "sell_watchlist": [], "phase_b_watchlist": [],
    }
    msg = t._build_message(data)
    assert "No qualifying candidates" in msg
    assert "PEAD WATCH" not in msg


def test_negative_pead_entry_uses_miss_selloff_label():
    entry = {"ticker": "BAR.NS", "direction": "negative", "headline": "bad news"}
    formatted = t._format_pead_entry(entry, 1)
    assert "BAR.NS" in formatted
    assert "miss" in formatted.lower()


def test_positive_pead_entry_uses_beat_rally_label():
    entry = {"ticker": "FOO.NS", "direction": "positive", "headline": "good news"}
    formatted = t._format_pead_entry(entry, 1)
    assert "FOO.NS" in formatted
    assert "beat" in formatted.lower()
