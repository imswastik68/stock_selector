"""
Regression tests for the LIVE PROOF section in the EOD Telegram message
(src/telegram_alert._live_proof_section / _build_message). Must render on
all three message paths (has-picks, zero-candidates-with-PEAD,
zero-candidates-terse) when live_proof.lines is non-empty, and render
nothing (no crash, no empty section) when it's absent or empty.
"""

from __future__ import annotations

import src.telegram_alert as t

_LIVE_PROOF = {
    "lines": [
        "reversal_oversold_v2: +0.8% α over n=34 (win 58.8%), p=0.031 ✅ PROVEN",
        "AGGREGATE (all buys): INSUFFICIENT (n=42/100, 2/3 months)",
    ],
}


def test_live_proof_renders_on_has_picks_path():
    data = {
        "scan_date": "2026-07-06", "total_screened": 300, "nifty_context": "ranging",
        "buy_watchlist": [{"ticker": "X.NS", "score": 5, "today_close": 100.0,
                            "stop_loss": "₹90", "target_1": "₹120", "target_2": "₹130"}],
        "sell_watchlist": [], "phase_b_watchlist": [], "pead_watchlist": [],
        "live_proof": _LIVE_PROOF,
    }
    msg = t._build_message(data)
    assert "LIVE PROOF" in msg
    assert "reversal_oversold_v2" in msg
    assert "PROVEN" in msg


def test_live_proof_renders_on_zero_candidate_terse_path():
    data = {
        "scan_date": "2026-07-06", "total_screened": 300, "nifty_context": "ranging",
        "buy_watchlist": [], "sell_watchlist": [], "phase_b_watchlist": [], "pead_watchlist": [],
        "live_proof": _LIVE_PROOF,
    }
    msg = t._build_message(data)
    assert "No qualifying candidates" in msg
    assert "LIVE PROOF" in msg


def test_live_proof_renders_on_pead_only_path():
    data = {
        "scan_date": "2026-07-06", "total_screened": 300, "nifty_context": "ranging",
        "buy_watchlist": [], "sell_watchlist": [], "phase_b_watchlist": [],
        "pead_watchlist": [{"ticker": "FOO.NS", "direction": "positive", "headline": "beat"}],
        "live_proof": _LIVE_PROOF,
    }
    msg = t._build_message(data)
    assert "PEAD WATCH" in msg
    assert "LIVE PROOF" in msg


def test_live_proof_section_empty_when_no_lines():
    assert t._live_proof_section({"live_proof": {"lines": []}}) == ""
    assert t._live_proof_section({}) == ""


def test_missing_live_proof_key_does_not_crash():
    data = {
        "scan_date": "2026-07-06", "total_screened": 300, "nifty_context": "ranging",
        "buy_watchlist": [], "sell_watchlist": [], "phase_b_watchlist": [], "pead_watchlist": [],
    }
    msg = t._build_message(data)
    assert "No qualifying candidates" in msg
    assert "LIVE PROOF" not in msg
