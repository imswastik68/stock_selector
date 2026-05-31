"""
Deterministic technical indicators computed from OHLCV DataFrames.
No LLM involved. All outputs are auditable and reproducible.
"""

from __future__ import annotations

import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    last_loss = avg_loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    rs = avg_gain.iloc[-1] / last_loss
    return round(100 - 100 / (1 + rs), 1)


def compute_macd_signal(close: pd.Series) -> str:
    """Return 'bullish_cross', 'bearish_cross', or 'none' based on MACD(12,26,9)."""
    if len(close) < 27:
        return "none"
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    prev_diff = macd.iloc[-2] - signal.iloc[-2]
    curr_diff = macd.iloc[-1] - signal.iloc[-1]
    if prev_diff <= 0 < curr_diff:
        return "bullish_cross"
    if prev_diff >= 0 > curr_diff:
        return "bearish_cross"
    return "none"


def classify_wyckoff_phase(df: pd.DataFrame) -> str:
    """
    Rule-based Wyckoff phase from OHLCV. Returns one of the 8 exact phase strings.
    Logic: based on EMA trend, recent EMA crossover, and price vs range.
    """
    if len(df) < 30:
        return "ACCUMULATION_B"

    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    price_now = float(close.iloc[-1])
    ema20_now = float(ema20.iloc[-1])
    ema50_now = float(ema50.iloc[-1])

    # Recent EMA crossover (last 5 bars)
    ema_diff = ema20 - ema50
    recent_cross_up   = (ema_diff.iloc[-6:-1] <= 0).any() and ema_diff.iloc[-1] > 0
    recent_cross_down = (ema_diff.iloc[-6:-1] >= 0).any() and ema_diff.iloc[-1] < 0

    # Volume trend: is recent volume expanding or contracting?
    vol_30d = float(volume.iloc[-31:-1].mean()) if len(volume) >= 32 else float(volume.mean())
    vol_5d  = float(volume.iloc[-6:-1].mean())  if len(volume) >= 7  else vol_30d
    vol_expanding = vol_5d > vol_30d * 1.2

    # Price range over bars 10-45 (the "trading range")
    range_window = close.iloc[-45:-5] if len(close) >= 50 else close.iloc[:-5]
    range_high = float(range_window.max()) if not range_window.empty else price_now
    range_low  = float(range_window.min()) if not range_window.empty else price_now

    near_high = price_now >= range_high * 0.97
    near_low  = price_now <= range_low  * 1.03

    # Phase classification (order matters — most specific first)
    if recent_cross_up and near_low:
        return "ACCUMULATION_C"  # EMA just crossed up from low range = Spring/LPS signal

    if recent_cross_down and near_high:
        return "DISTRIBUTION_C"  # EMA just crossed down from high range = UTAD/LPSY signal

    if price_now > ema20_now > ema50_now and vol_expanding:
        return "MARKUP"

    if price_now < ema20_now < ema50_now and vol_expanding:
        return "MARKDOWN"

    if price_now > ema20_now > ema50_now:
        return "ACCUMULATION_D"  # Trending up but volume not yet expanding = late accum

    if price_now < ema20_now < ema50_now:
        return "DISTRIBUTION_D"  # Trending down but volume not expanding = late dist

    if near_high:
        return "DISTRIBUTION_B"  # Range-bound near highs

    return "ACCUMULATION_B"      # Range-bound near lows or midpoint


def compute_entry_levels(close: float, atr: float, direction: str) -> dict:
    """
    ATR-based entry zone, stop loss, and targets.
    Risk = 2×ATR, Reward = 4/6×ATR → R:R = 1:2 / 1:3.
    """
    if atr <= 0:
        return {}
    half_atr = atr * 0.5
    if direction == "buy":
        entry_lo = round(close - half_atr, 2)
        entry_hi = round(close + half_atr, 2)
        stop     = round(close - 2 * atr,  2)
        target1  = round(close + 4 * atr,  2)
        target2  = round(close + 6 * atr,  2)
    else:
        entry_lo = round(close - half_atr, 2)
        entry_hi = round(close + half_atr, 2)
        stop     = round(close + 2 * atr,  2)
        target1  = round(close - 4 * atr,  2)
        target2  = round(close - 6 * atr,  2)

    return {
        "entry_zone": f"₹{entry_lo}-₹{entry_hi}",
        "stop_loss":  f"₹{stop}",
        "target_1":   f"₹{target1}",
        "target_2":   f"₹{target2}",
        "risk_reward": "1:2",
    }


def enrich_with_technicals(df: pd.DataFrame, close: float, atr: float) -> dict:
    """Return RSI, MACD signal, Wyckoff phase, and direction for a candidate."""
    close_series = df["Close"].squeeze()
    rsi = compute_rsi(close_series)
    macd = compute_macd_signal(close_series)
    phase = classify_wyckoff_phase(df)

    _BUY_PHASES  = {"ACCUMULATION_C", "ACCUMULATION_D", "MARKUP"}
    _SELL_PHASES = {"DISTRIBUTION_C", "DISTRIBUTION_D", "MARKDOWN"}

    if phase in _BUY_PHASES:
        direction = "buy"
    elif phase in _SELL_PHASES:
        direction = "sell"
    else:
        direction = "watch"  # phase-B

    # RSI overrides: if RSI contradicts phase, downgrade confidence
    rsi_agrees = (
        (direction == "buy"  and rsi < 70) or
        (direction == "sell" and rsi > 30) or
        (direction == "watch")
    )

    levels = compute_entry_levels(close, atr, direction) if direction != "watch" else {}

    return {
        "rsi": rsi,
        "macd_signal": macd,
        "wyckoff_phase": phase,
        "direction": direction,
        "wyckoff_confidence": "HIGH" if rsi_agrees else "MEDIUM",
        **levels,
    }
