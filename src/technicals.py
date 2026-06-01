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


def _compute_macd(close: pd.Series) -> dict:
    """
    MACD(12,26,9) — all signals in one pass.

    Signal cross lookback 3 bars: catches crosses from the last 3 sessions
    (EOD screener may miss a same-day cross depending on run time).
    Zero cross lookback 5 bars: zero-line cross is a stronger, slower signal.

    Returns:
      signal (str)          — display string for Telegram / JSON
      bullish_cross (bool)  — histogram crossed from negative to positive in last 3 bars
      bearish_cross (bool)  — histogram crossed from positive to negative in last 3 bars
      above_signal (bool)   — MACD currently above signal line
      below_signal (bool)   — MACD currently below signal line
      zero_cross_up (bool)  — MACD crossed above zero in last 5 bars
      zero_cross_down (bool)— MACD crossed below zero in last 5 bars
    """
    _empty = {
        "signal": "none", "bullish_cross": False, "bearish_cross": False,
        "above_signal": False, "below_signal": False,
        "zero_cross_up": False, "zero_cross_down": False,
    }
    if len(close) < 35:  # 26 EMA + 9 signal warmup
        return _empty

    ema12       = close.ewm(span=12, adjust=False).mean()
    ema26       = close.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram   = macd_line - signal_line

    hist = histogram.values
    mac  = macd_line.values

    above_signal = bool(hist[-1] > 0)
    below_signal = bool(hist[-1] < 0)

    # Signal-line cross within last 3 bars
    sig_window = hist[-(3 + 1):]  # 4 values → 3 consecutive pairs
    bullish_cross = any(sig_window[i] < 0 <= sig_window[i + 1] for i in range(len(sig_window) - 1))
    bearish_cross = any(sig_window[i] > 0 >= sig_window[i + 1] for i in range(len(sig_window) - 1))

    # Zero-line cross within last 5 bars
    zero_window    = mac[-(5 + 1):]  # 6 values → 5 consecutive pairs
    zero_cross_up  = any(zero_window[i] <= 0 < zero_window[i + 1] for i in range(len(zero_window) - 1))
    zero_cross_down= any(zero_window[i] >= 0 > zero_window[i + 1] for i in range(len(zero_window) - 1))

    # Display string — priority: zero cross > signal cross > state
    if zero_cross_up:
        signal = "zero_cross_up"
    elif zero_cross_down:
        signal = "zero_cross_down"
    elif bullish_cross:
        signal = "bullish_cross"
    elif bearish_cross:
        signal = "bearish_cross"
    elif above_signal:
        signal = "above_signal"
    elif below_signal:
        signal = "below_signal"
    else:
        signal = "none"

    return {
        "signal":          signal,
        "bullish_cross":   bullish_cross,
        "bearish_cross":   bearish_cross,
        "above_signal":    above_signal,
        "below_signal":    below_signal,
        "zero_cross_up":   zero_cross_up,
        "zero_cross_down": zero_cross_down,
    }


def compute_macd_signal(close: pd.Series) -> str:
    """Public display string for MACD state. See _compute_macd for full signals."""
    return _compute_macd(close)["signal"]


def classify_wyckoff_phase(df: pd.DataFrame) -> str:
    """
    Rule-based Wyckoff phase classification.

    Four inputs (in order of reliability):
      1. Prior trend — bars -60 to -30 ago vs bars -30 to now
      2. EMA alignment — trending_up/down vs ranging
      3. Volume character — up-day vol vs down-day vol, expansion
      4. Price structure — range position, Spring wick, UTAD wick, LPS pullback

    Phase mapping:
      BUY:   ACCUMULATION_C (Spring), ACCUMULATION_D (LPS), MARKUP
      SELL:  DISTRIBUTION_C (UTAD),  DISTRIBUTION_D (LPSY), MARKDOWN
      WATCH: ACCUMULATION_B, DISTRIBUTION_B
    """
    if len(df) < 50:
        return "ACCUMULATION_B"

    close  = df["Close"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    volume = df["Volume"].squeeze()
    price_now = float(close.iloc[-1])

    # ── 1. Prior trend: compare price ~60 bars ago vs ~30 bars ago ───────────
    lookback = min(60, len(close) - 2)
    midback  = min(30, len(close) // 2)
    prior_up   = float(close.iloc[-midback]) > float(close.iloc[-lookback]) * 1.05
    prior_down = float(close.iloc[-midback]) < float(close.iloc[-lookback]) * 0.95

    # ── 2. EMA alignment ──────────────────────────────────────────────────────
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    e20, e50 = float(ema20.iloc[-1]), float(ema50.iloc[-1])
    trending_up   = price_now > e20 > e50
    trending_down = price_now < e20 < e50

    # ── 3. Volume character ───────────────────────────────────────────────────
    vol_avg30    = float(volume.iloc[-31:-1].mean()) if len(volume) >= 32 else float(volume.mean())
    vol_5d       = float(volume.iloc[-6:-1].mean())  if len(volume) >= 7  else vol_avg30
    vol_expanding = vol_5d > vol_avg30 * 1.2

    # Volume on up-days vs down-days over last 20 bars (accumulation / distribution character)
    diff20 = close.iloc[-21:].diff()
    vol20  = volume.iloc[-21:]
    up_vol   = float(vol20[diff20 > 0].mean()) if (diff20 > 0).any() else 0.0
    down_vol = float(vol20[diff20 < 0].mean()) if (diff20 < 0).any() else 0.0
    acc_vol  = up_vol   > down_vol * 1.1 if (up_vol > 0 and down_vol > 0) else False
    dist_vol = down_vol > up_vol   * 1.1 if (up_vol > 0 and down_vol > 0) else False

    # ── 4. Trading range (bars -35 to -5, so last 5 bars are "current action") ─
    rw      = close.iloc[-35:-5] if len(close) >= 40 else close.iloc[:-5]
    tr_high = float(rw.max()) if not rw.empty else price_now
    tr_low  = float(rw.min()) if not rw.empty else price_now
    tr_rng  = tr_high - tr_low

    range_pct = (price_now - tr_low) / tr_rng if tr_rng > 0 else 0.5
    near_high = range_pct >= 0.75
    near_low  = range_pct <= 0.25

    # Spring: LOW wick tested below range low within last 15 bars, CLOSE recovered
    # 15-bar window because the spring wick typically occurs 5-10 sessions before
    # the scanner run — 6 bars is too narrow to catch it reliably.
    low15  = float(low.iloc[-16:].min())  if len(low)  >= 16 else float(low.min())
    spring = low15 < tr_low * 0.99 and price_now > tr_low * 0.99

    # UTAD: HIGH wick tested above range high within last 15 bars, CLOSE failed back
    high15 = float(high.iloc[-16:].max()) if len(high) >= 16 else float(high.max())
    utad   = high15 > tr_high * 1.01 and price_now < tr_high * 1.01

    # LPS pullback: price pulled back 3%+ from its 20-bar high
    high20 = float(close.iloc[-21:-1].max()) if len(close) >= 21 else price_now
    low20  = float(close.iloc[-21:-1].min()) if len(close) >= 21 else price_now
    lps_pullback  = high20 > 0 and (high20 - price_now) / high20 > 0.03
    lpsy_bounce   = low20  > 0 and (price_now - low20)  / low20  > 0.03

    # EMA gap strength: wide gap (>5%) = strong trend, narrow = weakening / converging
    # Used to distinguish a genuine LPS (strong trend pausing) from a tired trend
    # that is drifting sideways before a reversal (which looks the same with EMAs alone).
    ema_bull_gap = (e20 - e50) / e50 if e50 > 0 else 0.0   # positive = bullish trend strength
    ema_bear_gap = (e50 - e20) / e50 if e50 > 0 else 0.0   # positive = bearish trend strength
    strong_bull  = ema_bull_gap > 0.05   # e20 is >5% above e50
    strong_bear  = ema_bear_gap > 0.05   # e20 is >5% below e50

    # ── Classification ────────────────────────────────────────────────────────

    # MARKUP / MARKDOWN: full EMA alignment — stock is actively trending
    if trending_up:
        return "MARKUP"
    if trending_down:
        return "MARKDOWN"

    # Ranging from here: price is between or around the two EMAs
    # ──────────────────────────────────────────────────────────────────────────

    # ACCUMULATION_C (Spring):
    #   Intraday wick tested below range low but CLOSE recovered above it.
    #   Requires prior downtrend or currently near range lows (no spring from uptrend).
    if spring and (prior_down or near_low):
        return "ACCUMULATION_C"

    # DISTRIBUTION_C (UTAD):
    #   Intraday wick tested above range high but CLOSE failed back below it.
    #   Requires prior uptrend or currently near range highs.
    if utad and (prior_up or near_high):
        return "DISTRIBUTION_C"

    # EMA alignment determines which regime (bullish vs bearish) we are in:
    #   e20 > e50 = medium-term trend still bullish  → accumulation family
    #   e20 < e50 = medium-term trend still bearish  → distribution family
    #   e20 ≈ e50 = neutral → use prior trend / position

    if e20 > e50:
        # LPS pullback: only valid when the EMA gap is wide (>5%) — that confirms the trend
        # was genuinely strong before the pullback, not just a convergence artifact.
        if strong_bull and lps_pullback and not vol_expanding and price_now > e50 * 0.97:
            return "ACCUMULATION_D"
        # Pre-breakout: still ranging but acc volume near range top after a downtrend
        if acc_vol and near_high and prior_down:
            return "ACCUMULATION_D"
        return "ACCUMULATION_B"

    if e20 < e50:
        # LPSY bounce: valid only when bear trend is genuinely strong (EMA gap > 5%)
        if strong_bear and lpsy_bounce and not vol_expanding and price_now < e50 * 1.03:
            return "DISTRIBUTION_D"
        # Pre-breakdown: ranging with dist volume near range bottom after an uptrend
        if dist_vol and near_low and prior_up:
            return "DISTRIBUTION_D"
        return "DISTRIBUTION_B"

    # EMAs nearly equal — use prior trend context
    if prior_up or near_high:
        return "DISTRIBUTION_B"
    return "ACCUMULATION_B"


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
    rsi     = compute_rsi(close_series)
    macd    = _compute_macd(close_series)
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

    # RSI divergence — must be computed before rsi_agrees
    rsi_momentum    = bool(50 <= rsi <= 75)
    rsi_div         = compute_rsi_divergence(close_series)
    rsi_bearish_div = rsi_div == "bearish_divergence"
    rsi_bullish_div = rsi_div == "bullish_divergence"

    # MACD momentum signals
    macd_bullish_cross = macd["bullish_cross"] or macd["zero_cross_up"]
    macd_bearish_cross = macd["bearish_cross"] or macd["zero_cross_down"]

    # Wyckoff confidence: HIGH when RSI and MACD both agree with phase direction
    rsi_agrees  = (
        (direction == "buy"  and not rsi_bearish_div) or
        (direction == "sell" and not rsi_bullish_div) or
        (direction == "watch")
    )
    macd_agrees = (
        (direction == "buy"  and not macd_bearish_cross) or
        (direction == "sell" and not macd_bullish_cross) or
        (direction == "watch")
    )
    wyckoff_confidence = "HIGH" if (rsi_agrees and macd_agrees) else "MEDIUM"

    # Relative strength vs Nifty over last 20 bars
    rs_vs_nifty = False
    if nifty_20d_return is not None and len(close_series) >= 20:
        stock_20d = float(close_series.iloc[-1] / close_series.iloc[-20] - 1)
        rs_vs_nifty = bool(stock_20d > nifty_20d_return + 0.02)

    levels = compute_entry_levels(close, atr, direction) if direction != "watch" else {}

    return {
        "rsi": rsi,
        "macd_signal": macd["signal"],
        "wyckoff_phase": phase,
        "direction": direction,
        "wyckoff_confidence": wyckoff_confidence,
        "candlestick_patterns": candles,
        "rsi_momentum":      rsi_momentum,
        "rsi_bearish_div":   rsi_bearish_div,
        "rsi_bullish_div":   rsi_bullish_div,
        "macd_bullish_cross": macd_bullish_cross,
        "macd_bearish_cross": macd_bearish_cross,
        "rs_vs_nifty":       rs_vs_nifty,
        "obv_accumulation":  obv_acc,
        "bb_squeeze_breakout": bb_sq,
        **pivots,
        **levels,
    }
