"""
Deterministic technical indicators computed from OHLCV DataFrames.
No LLM involved. All outputs are auditable and reproducible.
"""

from __future__ import annotations

import pandas as pd


def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """Full RSI series used by both compute_rsi() and divergence detection."""
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    avg_loss = (-delta).clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    # avg_loss == 0 means no losses at all → RSI = 100
    result = avg_gain.copy() * 0.0  # same index, start at 0
    nz = avg_loss != 0
    result[~nz] = 100.0
    result[nz] = 100 - 100 / (1 + avg_gain[nz] / avg_loss[nz])
    return result.round(1)


def compute_rsi(close: pd.Series, period: int = 14) -> float:
    return float(_rsi_series(close, period).iloc[-1])


def _find_pivots(series: pd.Series) -> tuple[list[int], list[int]]:
    """Return (peak_indices, trough_indices) as iloc positions within series."""
    vals = series.values
    peaks, troughs = [], []
    for i in range(1, len(vals) - 1):
        if vals[i] > vals[i - 1] and vals[i] > vals[i + 1]:
            peaks.append(i)
        elif vals[i] < vals[i - 1] and vals[i] < vals[i + 1]:
            troughs.append(i)
    return peaks, troughs


def compute_rsi_divergence(close: pd.Series, lookback: int = 20) -> str:
    """
    Detect regular RSI divergence. Compares TODAY's bar against the most
    recent prior pivot within `lookback` bars.

    Bearish: today's price > prior peak AND today's RSI < prior RSI at that
      peak by ≥2 pts. Strong trending stocks keep making higher highs WITH
      rising RSI — only exhausted moves show this split.
    Bullish: today's price < prior trough AND today's RSI > prior RSI at
      that trough by ≥2 pts. Sellers losing force even at new lows.

    Returns 'bearish_divergence', 'bullish_divergence', or 'none'.
    """
    if len(close) < max(lookback + 1, 20):
        return "none"

    full_rsi      = _rsi_series(close)
    current_price = float(close.iloc[-1])
    current_rsi   = float(full_rsi.iloc[-1])

    # History window excludes today (today is what we compare against)
    hist_c = close.iloc[-(lookback + 1):-1]
    hist_r = full_rsi.iloc[-(lookback + 1):-1]
    peaks, troughs = _find_pivots(hist_c)

    if peaks:
        p = peaks[-1]  # most recent prior peak
        if (current_price > float(hist_c.iloc[p]) and
                current_rsi < float(hist_r.iloc[p]) - 2):
            return "bearish_divergence"

    if troughs:
        t = troughs[-1]  # most recent prior trough
        if (current_price < float(hist_c.iloc[t]) and
                current_rsi > float(hist_r.iloc[t]) + 2):
            return "bullish_divergence"

    return "none"


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


def compute_pivot_points(df: pd.DataFrame) -> dict:
    """
    Standard pivot points from yesterday's OHLC, plus weekly and monthly pivots.
    Returns: daily_pivot, daily_r1/r2, daily_s1/s2, weekly_pivot, monthly_pivot.
    """
    result = {}

    def _pivots_from(high: float, low: float, close: float, prefix: str) -> None:
        p = (high + low + close) / 3
        result[f"{prefix}_pivot"] = round(p, 2)
        result[f"{prefix}_r1"]    = round(2 * p - low, 2)
        result[f"{prefix}_r2"]    = round(p + (high - low), 2)
        result[f"{prefix}_s1"]    = round(2 * p - high, 2)
        result[f"{prefix}_s2"]    = round(p - (high - low), 2)

    if df.empty or len(df) < 2:
        return result

    # Daily: previous session — skip if H=L=C (illiquid stock / data gap)
    prev = df.iloc[-2]
    ph, pl, pc = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
    if ph > pl:
        _pivots_from(ph, pl, pc, "daily")

    # Weekly: last full week (last 5 bars before today)
    week_data = df.iloc[-6:-1] if len(df) >= 6 else df.iloc[:-1]
    if not week_data.empty:
        _pivots_from(
            float(week_data["High"].max()),
            float(week_data["Low"].min()),
            float(week_data["Close"].iloc[-1]),
            "weekly",
        )

    # Monthly: last ~22 bars before today
    month_data = df.iloc[-23:-1] if len(df) >= 23 else df.iloc[:-1]
    if not month_data.empty:
        _pivots_from(
            float(month_data["High"].max()),
            float(month_data["Low"].min()),
            float(month_data["Close"].iloc[-1]),
            "monthly",
        )

    return result


def detect_candlestick_patterns(df: pd.DataFrame) -> list[str]:
    """
    Detect common candlestick patterns on the last 1-2 bars.
    Returns list of pattern names (empty if none detected).
    """
    if len(df) < 2:
        return []

    patterns = []
    o = df["Open"].squeeze()
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    c = df["Close"].squeeze()

    # ── last bar ──────────────────────────────────────────────────────────────
    o1, h1, l1, c1 = float(o.iloc[-1]), float(h.iloc[-1]), float(l.iloc[-1]), float(c.iloc[-1])
    o2, h2, l2, c2 = float(o.iloc[-2]), float(h.iloc[-2]), float(l.iloc[-2]), float(c.iloc[-2])

    body1 = abs(c1 - o1)
    range1 = h1 - l1 if h1 != l1 else 1e-9
    upper_shadow1 = h1 - max(o1, c1)
    lower_shadow1 = min(o1, c1) - l1

    body2 = abs(c2 - o2)

    # Doji: open ≈ close (body < 10% of range)
    if body1 < range1 * 0.1:
        patterns.append("doji")

    # Hammer: bullish reversal — small body in upper half, long lower shadow ≥ 2× body
    if (lower_shadow1 >= 2 * body1 and upper_shadow1 <= 0.3 * body1
            and min(o1, c1) > (h1 + l1) / 2):
        patterns.append("hammer")

    # Shooting star: bearish reversal — small body in lower half, long upper shadow ≥ 2× body
    if (upper_shadow1 >= 2 * body1 and lower_shadow1 <= 0.3 * body1
            and max(o1, c1) < (h1 + l1) / 2):
        patterns.append("shooting_star")

    # Bullish Engulfing: prev bar bearish, current bar bullish fully engulfs prev body
    if c2 < o2 and c1 > o1 and o1 <= c2 and c1 >= o2:
        patterns.append("bullish_engulfing")

    # Bearish Engulfing: prev bar bullish, current bar bearish fully engulfs prev body
    if c2 > o2 and c1 < o1 and o1 >= c2 and c1 <= o2:
        patterns.append("bearish_engulfing")

    # Bullish Marubozu: almost no shadows, full bullish body (≥ 90% of range)
    if c1 > o1 and body1 >= range1 * 0.9:
        patterns.append("bullish_marubozu")

    # Bearish Marubozu: almost no shadows, full bearish body (≥ 90% of range)
    if c1 < o1 and body1 >= range1 * 0.9:
        patterns.append("bearish_marubozu")

    return patterns


def compute_obv_signal(df: pd.DataFrame) -> bool:
    """OBV trending up faster than price over last 10 bars = stealth accumulation."""
    if len(df) < 20:
        return False
    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (direction * volume).cumsum()
    obv_10 = obv.iloc[-10:]
    price_10 = close.iloc[-10:]
    base_obv = float(obv_10.iloc[0]) or 1e-9
    base_price = float(price_10.iloc[0]) or 1e-9
    obv_slope   = (float(obv_10.iloc[-1])   - base_obv)   / abs(base_obv)
    price_slope = (float(price_10.iloc[-1]) - base_price) / abs(base_price)
    return bool(obv_slope > 0.01 and obv_slope > price_slope)


def compute_bb_squeeze(df: pd.DataFrame, period: int = 20) -> bool:
    """BB width currently below 50% of its 90-day average AND price broke outside the band."""
    if len(df) < period + 5:
        return False
    close = df["Close"].squeeze()
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    bb_width = ((upper - lower) / sma).dropna()
    if bb_width.empty:
        return False
    avg_width    = float(bb_width.mean())
    recent_width = float(bb_width.iloc[-5:].mean())
    price_now    = float(close.iloc[-1])
    upper_now    = float(upper.iloc[-1])
    lower_now    = float(lower.iloc[-1])
    squeezed  = recent_width < avg_width * 0.5
    broken_out = price_now > upper_now or price_now < lower_now
    return bool(squeezed and broken_out)


def enrich_with_technicals(
    df: pd.DataFrame,
    close: float,
    atr: float,
    nifty_20d_return: float | None = None,
) -> dict:
    """Return RSI, MACD, Wyckoff phase, pivot points, candlestick patterns, OBV, BB squeeze, RS."""
    close_series = df["Close"].squeeze()
    rsi  = compute_rsi(close_series)
    macd = compute_macd_signal(close_series)
    phase   = classify_wyckoff_phase(df)
    pivots  = compute_pivot_points(df)
    candles = detect_candlestick_patterns(df)
    obv_acc = compute_obv_signal(df)
    bb_sq   = compute_bb_squeeze(df)

    _BUY_PHASES  = {"ACCUMULATION_C", "ACCUMULATION_D", "MARKUP"}
    _SELL_PHASES = {"DISTRIBUTION_C", "DISTRIBUTION_D", "MARKDOWN"}

    if phase in _BUY_PHASES:
        direction = "buy"
    elif phase in _SELL_PHASES:
        direction = "sell"
    else:
        direction = "watch"

    rsi_agrees = (
        (direction == "buy"  and not rsi_bearish_div) or
        (direction == "sell" and not rsi_bullish_div) or
        (direction == "watch")
    )

    # RSI scoring signals
    rsi_momentum = bool(50 <= rsi <= 75)   # building momentum
    rsi_div      = compute_rsi_divergence(close_series)
    rsi_bearish_div = rsi_div == "bearish_divergence"  # price HH + RSI LH: momentum fading
    rsi_bullish_div = rsi_div == "bullish_divergence"  # price LL + RSI HL: sellers tiring

    # Relative strength vs Nifty over last 20 bars
    rs_vs_nifty = False
    if nifty_20d_return is not None and len(close_series) >= 20:
        stock_20d = float(close_series.iloc[-1] / close_series.iloc[-20] - 1)
        rs_vs_nifty = bool(stock_20d > nifty_20d_return + 0.02)

    levels = compute_entry_levels(close, atr, direction) if direction != "watch" else {}

    return {
        "rsi": rsi,
        "macd_signal": macd,
        "wyckoff_phase": phase,
        "direction": direction,
        "wyckoff_confidence": "HIGH" if rsi_agrees else "MEDIUM",
        "candlestick_patterns": candles,
        "rsi_momentum": rsi_momentum,
        "rsi_bearish_div": rsi_bearish_div,
        "rsi_bullish_div": rsi_bullish_div,
        "rs_vs_nifty": rs_vs_nifty,
        "obv_accumulation": obv_acc,
        "bb_squeeze_breakout": bb_sq,
        **pivots,
        **levels,
    }
