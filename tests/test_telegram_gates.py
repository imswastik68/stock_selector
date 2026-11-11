"""
Regression tests for the gate-status line in the EOD Telegram message
(src/telegram_alert._build_message). The gate lines must render on BOTH the
has-picks path AND the zero-candidate early-return path -- the gate status is
arguably most useful on a no-pick day, and it was originally only wired into
the has-picks branch.
"""

from __future__ import annotations

import src.telegram_alert as t

_GATES = {
    "scanner": {"status_line": "SCANNER GATE: NOT CLEARED (12.5% over 16 decided)"},
    "momentum": {"status_line": "MOMENTUM GATE: PAPER-ONLY (nothing shipped)"},
}


def test_gates_render_with_picks():
    data = {
        "scan_date": "2026-07-05", "total_screened": 300, "nifty_context": "ranging",
        "gates": _GATES,
        "buy_watchlist": [{"ticker": "X.NS", "score": 5, "today_close": 100.0,
                            "stop_loss": "₹90", "target_1": "₹120", "target_2": "₹130"}],
        "sell_watchlist": [], "phase_b_watchlist": [],
    }
    msg = t._build_message(data)
    assert "🚦" in msg
    assert "SCANNER GATE" in msg
    assert "MOMENTUM GATE" in msg


def test_gates_render_on_zero_candidate_day():
    data = {
        "scan_date": "2026-07-05", "total_screened": 300, "nifty_context": "ranging",
        "gates": _GATES,
        "buy_watchlist": [], "sell_watchlist": [], "phase_b_watchlist": [],
    }
    msg = t._build_message(data)
    assert "No qualifying candidates" in msg
    assert "🚦" in msg
    assert "SCANNER GATE" in msg


def test_zero_candidate_day_without_gates_key_does_not_crash():
    data = {
        "scan_date": "2026-07-05", "total_screened": 300, "nifty_context": "ranging",
        "buy_watchlist": [], "sell_watchlist": [], "phase_b_watchlist": [],
    }
    msg = t._build_message(data)
    assert "No qualifying candidates" in msg
    assert "🚦" not in msg


def test_fo_entry_with_string_levels_renders():
    # technicals emits SL/targets as "₹90" strings; _fmt_price used :,.2f on
    # them and crashed every EOD send that had an F&O pick (CI 2026-11-06..11).
    data = {
        "scan_date": "2026-11-11", "total_screened": 300, "nifty_context": "ranging",
        "buy_watchlist": [{"ticker": "X.NS", "score": 5, "today_close": 100.0,
                            "stop_loss": "₹90", "target_1": "₹120", "target_2": "₹130"}],
        "sell_watchlist": [], "phase_b_watchlist": [],
        "fo_watchlist": [{"ticker": "Y.NS", "phase": "MARKUP", "score": 6,
                          "today_close": 200.0, "entry_zone": "₹195-₹205",
                          "stop_loss": "₹180", "target_1": "₹230"}],
    }
    msg = t._build_message(data)
    assert "SL" in msg and "₹180" in msg and "₹230" in msg
