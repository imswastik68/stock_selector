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
