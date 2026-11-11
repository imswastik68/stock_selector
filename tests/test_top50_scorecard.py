"""
Pins the success-bar rules adopted 2026-11-11 (scripts/top50_scorecard.py):

- Entry depends on when the alert was SENT. An after-hours alert must enter
  at the next open: scoring it at that day's close would credit a move the
  user could never have bought.
- "Touched" is a limit fill at the bar, and P&L books the bar on a touch,
  else the window's last close -- for shorts the sign flips.
- A horizon PASSES only with >= 30% hits AND >= 2x random AND P&L > 0 on
  enough alerts; any one failing is a FAIL, too few is INSUFFICIENT.

Synthetic data only, no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import top50_scorecard as sc

_CAL = pd.DatetimeIndex(["2026-11-06", "2026-11-09", "2026-11-11"])  # Fri, Mon, Wed (10th a holiday)
_P = {"cal": _CAL}


def _at(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="Asia/Kolkata")


def test_pre_open_alert_enters_same_day_open():
    assert sc.entry_index(_P, _at("2026-11-09 08:45")) == (1, "open")


def test_market_hours_alert_enters_same_day_close():
    assert sc.entry_index(_P, _at("2026-11-09 12:30")) == (1, "close")


def test_after_close_alert_enters_next_trading_day_open():
    assert sc.entry_index(_P, _at("2026-11-09 16:29")) == (2, "open")  # skips the 10th holiday


def test_weekend_and_holiday_alerts_enter_next_open():
    assert sc.entry_index(_P, _at("2026-11-07 10:00")) == (1, "open")   # Saturday
    assert sc.entry_index(_P, _at("2026-11-10 11:00")) == (2, "open")   # holiday


def test_alert_after_last_bar_has_no_entry():
    assert sc.entry_index(_P, _at("2026-11-11 16:00")) is None


def test_bullish_touch_books_the_bar_minus_cost():
    s = sc._score(np.array(100.0), np.array(98.0), np.array(113.0), "bullish", 5)
    assert bool(s["touch"]) and not bool(s["at_end"])
    assert float(s["pnl"]) == sc.BAR["bullish"][5] - sc.COST_PCT


def test_bullish_miss_books_the_window_close():
    s = sc._score(np.array(100.0), np.array(97.0), np.array(110.0), "bullish", 5)
    assert not bool(s["touch"])
    assert round(float(s["pnl"]), 6) == -3.0 - sc.COST_PCT


def test_bearish_profit_is_a_fall():
    hit = sc._score(np.array(100.0), np.array(95.0), np.array(90.0), "bearish", 5)
    assert bool(hit["touch"]) and float(hit["pnl"]) == abs(sc.BAR["bearish"][5]) - sc.COST_PCT
    miss = sc._score(np.array(100.0), np.array(104.0), np.array(97.0), "bearish", 5)
    assert not bool(miss["touch"]) and round(float(miss["pnl"]), 6) == -4.0 - sc.COST_PCT


def _rows(n, dates, hit, rand, pnl):
    k = int(round(n * hit))
    return pd.DataFrame({
        "date": [f"d{i % dates}" for i in range(n)],
        "touch": [True] * k + [False] * (n - k),
        "rand_touch": rand, "at_end": False, "rand_at_end": 0.0,
        "end": 0.0, "pnl": pnl, "rand_pnl": 0.0,
    })


def test_verdict_pass_needs_all_three():
    assert sc.verdict(_rows(120, 50, 0.35, 0.10, 0.5))["status"] == "PASS"


def test_verdict_fails_on_any_one():
    assert sc.verdict(_rows(120, 50, 0.25, 0.05, 0.5))["status"] == "FAIL"   # < 30%
    assert sc.verdict(_rows(120, 50, 0.35, 0.20, 0.5))["status"] == "FAIL"   # < 2x random
    assert sc.verdict(_rows(120, 50, 0.35, 0.10, -0.1))["status"] == "FAIL"  # loses money


def test_verdict_insufficient_sample():
    assert sc.verdict(_rows(99, 50, 0.5, 0.05, 1.0))["status"] == "INSUFFICIENT"
    assert sc.verdict(_rows(120, 39, 0.5, 0.05, 1.0))["status"] == "INSUFFICIENT"
