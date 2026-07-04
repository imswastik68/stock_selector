"""
Regression test for the audit fix to scripts/fetch_telegram_history.py:
_classify() tags an entire message with ONE direction (BUY checked first),
but a single alert can bundle a BUY section, a SELL section, and a WATCH
LIST section together. The old flat parse swept every numbered entry from
all three sections into that one direction. _split_sections() now parses
each section independently.

Uses a synthetic message matching the real 2026-06-14 STARHEALTH regression
(a BUY+SELL combined alert that mislabeled STARHEALTH.NS as "buy" instead of
"sell" before the fix) -- no network access, no live Telegram fetch.
"""

from __future__ import annotations

import fetch_telegram_history as fth

_COMBINED_MESSAGE = """**NSE/BSE Stock Scanner — 2026-06-14 | 1:37 PM IST**
__Nifty 50: DOWNTREND ↘ | 206 stocks screened__
────────────────────────────────


📈 **BUY WATCHLIST (1 picks)**


**1. SHREEJISPG.NS** 🟡  `₹483.10`
  🚀 **MARKUP — Strong uptrend**
  Score: `5` | Confidence: HIGH | RSI: `67.2`
  MACD: `below_signal`

  Entry: `₹474.69-₹491.51` | SL: `₹449.48`
  T1: `₹533.54` | T2: `₹567.16` | R:R `1:2`
  Short-term (1–2d)
  Signals: __actual 52w breakout, rsi momentum__


📉 **SELL / SHORT WATCHLIST (1 picks)**


**2. STARHEALTH.NS** 🟡  `₹519.60`
  ⚠️ **DISTRIBUTION_C — Bull trap**
  Score: `5` | Confidence: HIGH | RSI: `50.3`
  MACD: `below_signal`

  Short entry: `₹512.74-₹526.46` | SL: `₹547.06`
  Cover T1: `₹478.42` | T2: `₹450.96` | R:R `1:2`
  Short-term (1–2d)
  Signals: __rsi momentum, rs vs nifty__


👀 **WATCH LIST (1 stocks)**


**1. WATCHONLY.NS**  `₹100.00`
  📊 **ACCUMULATION_B — Basing**
  ⏳ __Phase building — no entry signal yet__
  Raw score: `3` | RSI: `55.0`
  Active signals: __volume 5x__


⚠️ __Not investment advice. Do your own due diligence.__"""


def test_split_sections_finds_all_three_sections():
    sections = fth._split_sections(_COMBINED_MESSAGE)
    directions = [d for d, _ in sections]
    assert directions == ["buy", "sell", "watch"]


def test_starhealth_parses_as_sell_not_buy():
    """The exact real-world regression: before the fix, _classify() tagged
    this whole message 'buy' (BUY WATCHLIST checked first) and STARHEALTH.NS
    was recorded with direction='buy' despite being in the SELL section."""
    picks = []
    for direction, text in fth._split_sections(_COMBINED_MESSAGE):
        if direction == "watch":
            continue
        picks.extend(fth._parse_picks(text, "2026-06-14", direction))

    by_ticker = {p["ticker"]: p for p in picks}
    assert by_ticker["SHREEJISPG.NS"]["direction"] == "buy"
    assert by_ticker["STARHEALTH.NS"]["direction"] == "sell"


def test_watch_list_entries_excluded_from_picks():
    """WATCH LIST entries have no entry/SL -- they are not trade
    recommendations and must never appear in the parsed picks list."""
    picks = []
    for direction, text in fth._split_sections(_COMBINED_MESSAGE):
        if direction == "watch":
            continue
        picks.extend(fth._parse_picks(text, "2026-06-14", direction))

    tickers = {p["ticker"] for p in picks}
    assert "WATCHONLY.NS" not in tickers
    assert len(picks) == 2


def test_starhealth_sl_geometry_matches_sell_direction():
    """Cross-check added alongside the section-split fix: for a SELL pick,
    sl > close > t1 (stop above entry, target below)."""
    for direction, text in fth._split_sections(_COMBINED_MESSAGE):
        if direction != "sell":
            continue
        picks = fth._parse_picks(text, "2026-06-14", direction)
        starhealth = next(p for p in picks if p["ticker"] == "STARHEALTH.NS")
        assert starhealth["sl"] > starhealth["close"] > starhealth["t1"]


def test_sell_only_message_still_classified_correctly():
    """A message with no BUY WATCHLIST section (only SELL) -- _classify()
    already handled this correctly pre-fix, confirm no regression."""
    sell_only = _COMBINED_MESSAGE.split("📉")[1]
    sell_only = "📉" + sell_only.split("👀")[0]
    assert fth._classify(sell_only) == "sell"
