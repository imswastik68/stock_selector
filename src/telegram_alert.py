"""Telegram bot formatter and sender for the daily watchlist."""

from __future__ import annotations

import asyncio
import html
import math
import os

import telegram

from src.trade_sim import EXIT_PLAN_HINT, WINNER_POLICY

RISK_EMOJI = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠"}
TIMEFRAME_LABEL = {
    "1-2d": "Short-term (1–2d)", "5-7d": "Swing (5–7d)",
    "3-5d": "Swing (3–5d)", "5-10d": "Multi-day (5–10d)",
}
WYCKOFF_EMOJI = {
    "ACCUMULATION_C": "🌀", "ACCUMULATION_D": "📈", "MARKUP": "🚀",
    "DISTRIBUTION_C": "⚠️", "DISTRIBUTION_D": "📉", "MARKDOWN": "🔻",
    "ACCUMULATION_B": "📊", "DISTRIBUTION_B": "📊",
}

# Plain-English explanation shown next to each Wyckoff phase name
PHASE_DESCRIPTIONS = {
    "MARKUP":         "Strong uptrend — price above all short & long-term averages",
    "MARKDOWN":       "Strong downtrend — price below all averages",
    "ACCUMULATION_C": "Spring setup — wick dipped below support, snapped back above (bear trap, reversal likely)",
    "ACCUMULATION_D": "Last pullback before breakout — uptrend intact, dipping to support (LPS entry)",
    "DISTRIBUTION_C": "Bull trap — wick spiked above resistance, closed back below (buyers exhausted)",
    "DISTRIBUTION_D": "Dead-cat bounce — downtrend forming, bouncing to resistance (LPSY short entry)",
    "ACCUMULATION_B": "Basing — tight range after a fall, building a base (no entry yet)",
    "DISTRIBUTION_B": "Topping — tight range after a rise, stalling near highs (no entry yet)",
}

# Human-readable signal names for the Telegram message
SIGNAL_LABELS = {
    "volume_5x":             "volume surge",
    "bulk_deal_fii_dii":     "FII/DII bulk deal",
    "actual_52w_breakout":   "52-week breakout",
    "consolidation_breakout":"consolidation breakout",
    "delivery_surge":        "delivery surge",
    "rs_vs_nifty":           "outperforming Nifty",
    "rsi_momentum":          "RSI momentum",
    "rsi_bullish_div":       "RSI bullish divergence",
    "rsi_bearish_div":       "RSI bearish divergence",
    "macd_bearish_cross":    "MACD bearish cross",
    "options_pcr_fear":        "PCR extreme fear",
    "options_long_buildup":    "OI long buildup",
    "options_short_buildup":   "OI short buildup",
    "options_pcr_greed":       "PCR extreme greed",
    "options_short_covering":  "OI short covering",
    "options_long_unwinding":  "OI long unwinding",
    "promoter_buying":         "promoter buying",
    "results_due":             "results due",
    "fo_ban_lifted":           "F&O ban lifted",
    "distribution_signal":     "distribution (sell-on-rise)",
    "thin_market_extreme":     "very thin market (<₹1cr)",
    "thin_market_light":       "thin market (₹1–5cr)",
    "f_group":                 "on F&O ban list",
    "bb_squeeze_breakout":     "BB squeeze breakout",
    "bullish_candle":          "bullish candle",
    "bearish_candle":          "bearish candle",
    "weekly_trend_aligned":    "weekly trend aligned",
    # announcement signals
    "results_beat_announced":  "results filed",
    "buyback_announced":       "buyback announced",
    "contract_win":            "order/contract win",
    "dividend_announced":      "dividend declared",
    # sector rotation
    "sector_in_momentum":      "hot sector",
    "rs_quality_strong":       "quality RS",
    # insider / SAST
    "sast_insider_buying":     "SAST insider buying",
    # short pipeline (Phase 3)
    "actual_52w_breakdown":    "52-week breakdown",
    "consolidation_breakdown": "consolidation breakdown",
    "heavy_selling":           "heavy selling",
    "bulk_deal_fii_sell":      "FII/DII sell deal",
}

# Short readable label for MACD signal state
MACD_LABELS = {
    "zero_cross_up":   "crossed above zero ↑",
    "zero_cross_down": "crossed below zero ↓",
    "bullish_cross":   "signal-line cross up ↑",
    "bearish_cross":   "signal-line cross down ↓",
    "above_signal":    "above signal (bullish)",
    "below_signal":    "below signal (bearish)",
    "none":            "neutral",
}


def _e(text: str) -> str:
    """Escape text for Telegram HTML mode."""
    return html.escape(str(text))


def _code(text) -> str:
    return f"<code>{_e(text)}</code>"


def _b(text: str) -> str:
    return f"<b>{_e(text)}</b>"


def _i(text: str) -> str:
    return f"<i>{_e(text)}</i>"


def _fmt_tags(tags: list) -> str:
    return " ".join(_code(t) for t in tags) if tags else ""


def _fmt_signal(s: str) -> str:
    """Convert internal signal name to human-readable label."""
    unconfirmed = "[UNCONFIRMED]" in s
    key = s.replace(" [UNCONFIRMED]", "")
    label = SIGNAL_LABELS.get(key, key.replace("_", " "))
    return f"{label} [unconfirmed]" if unconfirmed else label


def _fmt_signals(signals: list, n: int = 4) -> str:
    if not signals:
        return "—"
    return ", ".join(_fmt_signal(s) for s in signals[:n])


def _fmt_candles(patterns: list) -> str:
    if not patterns:
        return ""
    return " · ".join(p.replace("_", " ") for p in patterns)


def _fmt_pivots(entry: dict) -> str:
    p  = entry.get("daily_pivot")
    r1 = entry.get("daily_r1")
    s1 = entry.get("daily_s1")
    wp = entry.get("weekly_pivot")
    if not any([p, r1, s1]):
        return ""
    parts = []
    if p:  parts.append(f"P={_code(p)}")
    if r1: parts.append(f"R1={_code(r1)}")
    if s1: parts.append(f"S1={_code(s1)}")
    if wp: parts.append(f"wP={_code(wp)}")
    return " | ".join(parts)


def _fmt_price(price) -> str:
    if price is None:
        return ""
    return f"₹{price:,.2f}"


def _fmt_phase(phase: str) -> str:
    """Return 'PHASE NAME — plain-English description'."""
    name = phase.replace("_", " ")
    desc = PHASE_DESCRIPTIONS.get(phase, "")
    return f"{name} — {desc}" if desc else name


# ── entry formatters ──────────────────────────────────────────────────────────

def _format_buy_entry(entry: dict, rank: int) -> str:
    ticker     = entry.get("ticker", "?")
    score      = entry.get("score", 0)
    risk       = entry.get("risk", "MEDIUM")
    phase      = entry.get("wyckoff_phase", "?")
    confidence = entry.get("wyckoff_confidence", "?")
    rsi        = entry.get("rsi", "?")
    macd_raw   = entry.get("macd_signal", "none")
    tags       = entry.get("volatility_tags", [])
    entry_zone = entry.get("entry_zone", "—")
    stop       = entry.get("stop_loss", "—")
    t1         = entry.get("target_1", "—")
    t2         = entry.get("target_2", "—")
    rr         = entry.get("risk_reward", "—")
    tf         = entry.get("timeframe", "?")
    signals    = entry.get("top_signals", [])
    narrative  = entry.get("narrative", "")
    candles       = entry.get("candlestick_patterns", [])
    pcr           = entry.get("options_pcr")
    prom_pct      = entry.get("promoter_pct")
    delivery_pct  = entry.get("delivery_pct")
    price         = entry.get("today_close")
    atr_pct_val   = entry.get("atr_pct")

    risk_icon  = RISK_EMOJI.get(risk, "⚪")
    phase_icon = WYCKOFF_EMOJI.get(phase, "📊")
    macd_label = MACD_LABELS.get(macd_raw, macd_raw.replace("_", " "))
    tf_label   = TIMEFRAME_LABEL.get(tf, tf)
    tags_str   = _fmt_tags(tags)
    pivot_str  = _fmt_pivots(entry)
    candle_str = _fmt_candles(candles)
    price_str  = f"  {_code(_fmt_price(price))}" if price is not None else ""
    fire       = "🔥 " if entry.get("big_mover") else ""

    lines = [
        f"{fire}{_b(f'{rank}. {ticker}')} {risk_icon}{price_str}",
        f"  {phase_icon} {_b(_fmt_phase(phase))}",
        f"  Score: {_code(score)} | Confidence: {_e(confidence)} | RSI: {_code(rsi)}",
        f"  MACD: {_code(macd_label)}" + (f"  |  {tags_str}" if tags_str else ""),
    ]
    if candle_str:
        lines.append(f"  Candle: {_i(candle_str)}")
    if pivot_str:
        lines.append(f"  Pivots: {pivot_str}")
    extra = []
    if pcr is not None:
        extra.append(f"PCR={_code(pcr)}")
    if prom_pct is not None:
        extra.append(f"Promoter={_code(f'{prom_pct}%')}")
    if delivery_pct is not None:
        extra.append(f"Delivery={_code(f'{delivery_pct}%')}")
    pos = entry.get("position") or {}
    if entry.get("watch_only"):
        extra.append(f"⏸ <b>WATCH-ONLY</b> ({_e(entry.get('watch_only_reason', 'circuit'))}) — no new position sized")
    elif pos.get("shares", 0) > 0:
        qty_str = (f"Qty={_code(str(pos['shares']))} | "
                   f"₹{pos['notional']:,.0f} | "
                   f"risk ₹{pos['risk_amount']:,.0f} ({pos['risk_pct_actual']:.1f}%)")
        if pos.get("capped"):
            qty_str += " [CAPPED]"
        extra.append(qty_str)
    elif (atr_pct_val is not None and math.isfinite(atr_pct_val)
            and price is not None and price > 0):
        atr_abs = price * atr_pct_val / 100
        if atr_abs > 0:
            shares = int(1000 / atr_abs)
            if shares > 0:
                extra.append(f"Qty={_code(str(shares))} <i>(ATR ₹{atr_abs:.0f})</i>")
    mom_6m = entry.get("momentum_6m")
    rs_q   = entry.get("rs_quality")
    if mom_6m is not None:
        extra.append(f"6m={_code(f'{mom_6m:+.1f}%')}")
    if rs_q is not None and math.isfinite(rs_q):
        extra.append(f"RS={_code(f'{rs_q:.2f}')}")
    fund = entry.get("fundamental") or {}
    if fund:
        roe = fund.get("roe")
        de  = fund.get("de_ratio")
        if roe is not None or de is not None:
            parts = []
            if roe is not None: parts.append(f"ROE {roe:.0%}")
            if de  is not None: parts.append(f"D/E {de:.1f}")
            extra.append(f"Fund={_code(' '.join(parts))}")
    if extra:
        lines.append("  " + " | ".join(extra))
    exit_hint = EXIT_PLAN_HINT.get(WINNER_POLICY, WINNER_POLICY)
    lines += [
        "",
        f"  Entry: {_code(entry_zone)} | SL: {_code(stop)}",
        f"  T1: {_code(t1)} | T2: {_code(t2)} | R:R {_code(rr)}",
        f"  Exit: {_i(_e(exit_hint))}",
        f"  {_i(_e(tf_label))}",
        f"  Signals: {_i(_fmt_signals(signals))}",
    ]
    if narrative:
        lines.append(f"  💬 {_i(_e(narrative))}")
    return "\n".join(lines)


def _format_sell_entry(entry: dict, rank: int) -> str:
    ticker     = entry.get("ticker", "?")
    score      = entry.get("score", 0)
    risk       = entry.get("risk", "MEDIUM")
    phase      = entry.get("wyckoff_phase", "?")
    confidence = entry.get("wyckoff_confidence", "?")
    rsi        = entry.get("rsi", "?")
    macd_raw   = entry.get("macd_signal", "none")
    tags       = entry.get("volatility_tags", [])
    entry_zone = entry.get("entry_zone", "—")
    stop       = entry.get("stop_loss", "—")
    t1         = entry.get("target_1", "—")
    t2         = entry.get("target_2", "—")
    rr         = entry.get("risk_reward", "—")
    tf         = entry.get("timeframe", "?")
    signals    = entry.get("top_signals", [])
    narrative  = entry.get("narrative", "")
    candles    = entry.get("candlestick_patterns", [])
    pcr        = entry.get("options_pcr")
    prom_pct   = entry.get("promoter_pct")
    price      = entry.get("today_close")
    atr_pct_val = entry.get("atr_pct")

    risk_icon  = RISK_EMOJI.get(risk, "⚪")
    phase_icon = WYCKOFF_EMOJI.get(phase, "📊")
    macd_label = MACD_LABELS.get(macd_raw, macd_raw.replace("_", " "))
    tf_label   = TIMEFRAME_LABEL.get(tf, tf)
    tags_str   = _fmt_tags(tags)
    pivot_str  = _fmt_pivots(entry)
    candle_str = _fmt_candles(candles)
    price_str  = f"  {_code(_fmt_price(price))}" if price is not None else ""
    fire       = "🔥 " if entry.get("big_mover") else ""

    lines = [
        f"{fire}{_b(f'{rank}. {ticker}')} {risk_icon}{price_str}",
        f"  {phase_icon} {_b(_fmt_phase(phase))}",
        f"  Score: {_code(score)} | Confidence: {_e(confidence)} | RSI: {_code(rsi)}",
        f"  MACD: {_code(macd_label)}" + (f"  |  {tags_str}" if tags_str else ""),
    ]
    if candle_str:
        lines.append(f"  Candle: {_i(candle_str)}")
    if pivot_str:
        lines.append(f"  Pivots: {pivot_str}")
    extra = []
    if pcr is not None:
        extra.append(f"PCR={_code(pcr)}")
    if prom_pct is not None:
        extra.append(f"Promoter={_code(f'{prom_pct}%')}")
    pos = entry.get("position") or {}
    if entry.get("watch_only"):
        extra.append(f"⏸ <b>WATCH-ONLY</b> ({_e(entry.get('watch_only_reason', 'circuit'))}) — no new position sized")
    elif pos.get("shares", 0) > 0:
        qty_str = (f"Qty={_code(str(pos['shares']))} | "
                   f"₹{pos['notional']:,.0f} | "
                   f"risk ₹{pos['risk_amount']:,.0f} ({pos['risk_pct_actual']:.1f}%)")
        if pos.get("capped"):
            qty_str += " [CAPPED]"
        extra.append(qty_str)
    elif (atr_pct_val is not None and math.isfinite(atr_pct_val)
            and price is not None and price > 0):
        atr_abs = price * atr_pct_val / 100
        if atr_abs > 0:
            shares = int(1000 / atr_abs)
            if shares > 0:
                extra.append(f"Qty={_code(str(shares))} <i>(ATR ₹{atr_abs:.0f})</i>")
    mom_6m = entry.get("momentum_6m")
    rs_q   = entry.get("rs_quality")
    if mom_6m is not None:
        extra.append(f"6m={_code(f'{mom_6m:+.1f}%')}")
    if rs_q is not None and math.isfinite(rs_q):
        extra.append(f"RS={_code(f'{rs_q:.2f}')}")
    fund = entry.get("fundamental") or {}
    if fund:
        roe = fund.get("roe")
        de  = fund.get("de_ratio")
        if roe is not None or de is not None:
            parts = []
            if roe is not None: parts.append(f"ROE {roe:.0%}")
            if de  is not None: parts.append(f"D/E {de:.1f}")
            extra.append(f"Fund={_code(' '.join(parts))}")
    if extra:
        lines.append("  " + " | ".join(extra))
    exit_hint = EXIT_PLAN_HINT.get(WINNER_POLICY, WINNER_POLICY)
    instrument_str = " (via stock futures — F&O)" if entry.get("instrument") == "stock_future" else ""
    lines += [
        "",
        f"  Short entry: {_code(entry_zone)} | SL: {_code(stop)}{_i(instrument_str)}",
        f"  Cover T1: {_code(t1)} | T2: {_code(t2)} | R:R {_code(rr)}",
        f"  Exit: {_i(_e(exit_hint))}",
        f"  {_i(_e(tf_label))}",
        f"  Signals: {_i(_fmt_signals(signals))}",
    ]
    if narrative:
        lines.append(f"  💬 {_i(_e(narrative))}")
    return "\n".join(lines)


def _format_phase_b_entry(entry: dict, rank: int) -> str:
    ticker       = entry.get("ticker", "?")
    phase        = entry.get("phase", "?")
    score        = entry.get("score")
    rsi          = entry.get("rsi", "")
    price        = entry.get("today_close")
    trigger      = entry.get("alert_trigger", "—")
    watch_reason = entry.get("watch_reason", "")

    phase_icon = WYCKOFF_EMOJI.get(phase, "📊")
    price_str  = f"  {_code(_fmt_price(price))}" if price is not None else ""
    score_str  = f"Raw score: {_code(score)}" if score is not None else ""
    rsi_str    = f"RSI: {_code(rsi)}" if rsi else ""

    lines = [
        f"{_b(f'{rank}. {ticker}')}{price_str}",
        f"  {phase_icon} {_b(_fmt_phase(phase))}",
    ]
    if watch_reason:
        lines.append(f"  ⏳ {_i(_e(watch_reason))}")
    meta = " | ".join(x for x in [score_str, rsi_str] if x)
    if meta:
        lines.append(f"  {meta}")
    if isinstance(trigger, list):
        trigger_str = _fmt_signals(trigger, n=3) or "—"
    else:
        trigger_str = _e(str(trigger)).replace("_", " ")
    lines.append(f"  Active signals: {_i(trigger_str)}")
    return "\n".join(lines)


# ── legacy formatter ──────────────────────────────────────────────────────────

def _format_entry_legacy(entry: dict, rank: int) -> str:
    ticker       = entry.get("ticker", "?")
    score        = entry.get("score", 0)
    tf           = entry.get("timeframe", "?")
    target       = entry.get("target_move_pct", 0)
    risk         = entry.get("risk", "MEDIUM")
    entry_zone   = entry.get("entry_zone", "—")
    invalidation = entry.get("invalidation", "—")
    signals      = entry.get("top_signals", [])

    risk_icon = RISK_EMOJI.get(risk, "⚪")
    tf_label  = TIMEFRAME_LABEL.get(tf, tf)
    sigs_str  = "\n    • ".join(_fmt_signal(s) for s in signals) if signals else "—"

    return (
        f"{_b(f'{rank}. {ticker}')} {risk_icon}\n"
        f"  Score: {_code(score)} | {_e(tf_label)} | Target: {_code(f'+{target}%')}\n"
        f"  Entry: {_code(entry_zone)} | Stop: {_code(invalidation)}\n"
        f"  Signals:\n    • {sigs_str}"
    )


# ── mid-day message builder ───────────────────────────────────────────────────

def _build_midday_message(data: dict) -> str:
    scan_date      = data.get("scan_date", "?")
    scan_time      = data.get("scan_time", "12:30 PM IST")
    confirmed      = data.get("intraday_confirmed", {})
    all_checks     = data.get("intraday_checked", {})
    sl_hits        = data.get("sl_hits", {})
    t2_hits        = data.get("t2_hits", {})
    time_stop_due  = data.get("time_stop_due", {})
    holding        = {t: v for t, v in all_checks.items()
                      if not v.get("intraday_surge") and t not in sl_hits}

    header = (
        f"<b>Mid-day Momentum Check — {_e(scan_date)}</b>\n"
        f"<i>Snapshot at {_e(scan_time)} | {len(all_checks)} candidates checked</i>\n"
        f"{'─' * 32}"
    )
    lines = [header]

    if sl_hits:
        lines.append(f"\n🚨 <b>STOP-LOSS HIT — EXIT NOW ({len(sl_hits)} stocks)</b>\n")
        for ticker, v in sl_hits.items():
            direction = v.get("direction", "buy")
            price = v.get("price_current", "?")
            stop = v.get("stop_loss", "?")
            pct = v.get("pct_vs_stop", 0)
            action = "SELL → exit long" if direction == "buy" else "BUY → cover short"
            lines.append(
                f"{_b(ticker)}\n"
                f"  Action: {_code(action)}\n"
                f"  Price: {_code(f'₹{price}')} | SL: {_code(f'₹{stop}')} ({pct:+.1f}% vs SL)"
            )

    if t2_hits:
        lines.append(f"\n🎯 <b>T2 REACHED — consider booking ({len(t2_hits)} stocks)</b>\n")
        for ticker, v in t2_hits.items():
            price = v.get("price_current", "?")
            t2    = v.get("target_2", "?")
            lines.append(
                f"{_b(ticker)}\n"
                f"  Price: {_code(f'₹{price}')} hit T2: {_code(f'₹{t2}')}"
            )

    if time_stop_due:
        lines.append(f"\n⏰ <b>TIME-STOP DUE — consider exiting ({len(time_stop_due)} stocks)</b>\n")
        for ticker, v in time_stop_due.items():
            price = v.get("price_current", "?")
            days  = v.get("days_held", "?")
            lines.append(
                f"{_b(ticker)}\n"
                f"  Price: {_code(f'₹{price}')} | Held {_code(str(days))} days"
            )

    if confirmed:
        lines.append(f"\n🔥 <b>INTRADAY CONFIRMED ({len(confirmed)} stocks)</b>\n")
        for i, (ticker, v) in enumerate(confirmed.items(), 1):
            direction = "▲" if v["price_above_prev_close"] else "▼"
            pct = v.get("pct_vs_prev_close", 0)
            vol_str   = f"{v['volume_ratio_projected']:.1f}x"
            price_str = f"₹{v['price_current']}"
            lines.append(
                f"{_b(f'{i}. {ticker}')}\n"
                f"  Vol: {_code(vol_str)} projected"
                f" | {_code(price_str)} "
                f"({direction}{abs(pct):.1f}% vs prev close)"
            )
    else:
        lines.append("\n<i>No intraday surges yet — watchlist unchanged from morning.</i>")

    if holding:
        names = ", ".join(holding.keys())
        lines.append(f"\n<i>Holding steady: {_e(names)}</i>")

    lines.append("\n⚠️ <i>Projected volume is approximate. Not investment advice.</i>")
    return "\n\n".join(lines)


# ── factor (monthly momentum book) message builder ────────────────────────────

def _build_factor_message(data: dict) -> str:
    month   = data.get("month", "?")
    as_of   = data.get("as_of", "?")
    gate    = data.get("gate_status", {})
    book    = data.get("book", {})
    diff    = data.get("diff", {})
    prior   = data.get("prior_month_realized", {})
    signal  = data.get("promotion_signal", "")
    positions = book.get("positions", [])
    status    = book.get("status", "?")
    is_live   = gate.get("is_live", False)

    mode_str = "🟢 LIVE" if is_live else "📝 PAPER"
    header = (
        f"<b>📆 MONTHLY MOMENTUM BOOK — {_e(month)}</b>\n"
        f"<i>As of {_e(as_of)} | {mode_str} — {_e(gate.get('reason', ''))}</i>\n"
        f"{'─' * 32}"
    )
    sections = [header]

    if status.startswith("CASH"):
        sections.append(f"\n💰 <b>{_e(status)}</b>")
    else:
        sections.append(f"\n📊 <b>Book ({len(positions)} names, {mode_str})</b>\n")
        for i, p in enumerate(positions, 1):
            ticker = p.get("ticker", "?")
            qty    = p.get("qty", 0)
            price  = p.get("price", 0)
            notional = p.get("notional", 0)
            score  = p.get("mom_12_1_score", 0)
            sections.append(
                f"{_b(f'{i}. {ticker}')}\n"
                f"  Qty {_code(qty)} @ {_code(_fmt_price(price))} = {_code(f'₹{notional:,.0f}')}"
                f" | mom_12_1={_code(f'{score:+.3f}')}"
            )

    enter, exit_, hold = diff.get("enter", []), diff.get("exit", []), diff.get("hold", [])
    if enter or exit_:
        order_lines = []
        if enter: order_lines.append(f"  Enter: {_e(', '.join(enter))}")
        if exit_: order_lines.append(f"  Exit: {_e(', '.join(exit_))}")
        if hold:  order_lines.append(f"  Hold: {_e(', '.join(hold))}")
        sections.append("\n🔄 <b>Orders vs last month</b>\n" + "\n".join(order_lines))

    if prior.get("month"):
        book_ret  = prior.get("book_return_pct")
        nifty_ret = prior.get("nifty_return_pct")
        if book_ret is not None and nifty_ret is not None:
            beat = "beat" if book_ret > nifty_ret else "lagged"
            sections.append(
                f"\n📈 <i>{_e(prior['month'])} realized: book {book_ret:+.2f}% "
                f"vs Nifty {nifty_ret:+.2f}% ({beat})</i>"
            )
    if signal:
        sections.append(f"\n🔔 <i>{_e(signal)}</i>")

    disclaimer = ("\n\n⚠️ <i>Not investment advice. Paper-only until the statistical ship "
                  "gate passes.</i>" if not is_live else
                  "\n\n⚠️ <i>Not investment advice.</i>")
    sections.append(disclaimer)
    return "\n\n".join(sections)


# ── monthly A/B/C scoreboard message builder (Phase 5) ────────────────────────

def _build_scoreboard_message(data: dict) -> str:
    month = data.get("month", "?")
    sb    = data.get("scoreboard", {})
    mom   = sb.get("momentum", {})
    daily = sb.get("daily", {})

    header = (
        f"<b>🏆 MONTHLY SCOREBOARD — {_e(month)}</b>\n"
        f"<i>Momentum book vs daily scanner vs buy-and-hold Nifty</i>\n"
        f"{'─' * 32}"
    )
    sections = [header]

    if mom.get("n_months", 0) > 0:
        cum_str   = _code(f"{mom['cum_return_pct']:+.2f}%")
        nifty_str = _code(f"{mom['nifty_cum_return_pct']:+.2f}%")
        sections.append(
            f"\n📆 <b>Momentum book</b> ({mom['n_months']}mo since inception)\n"
            f"  Cumulative: {cum_str} vs Nifty {nifty_str}\n"
            f"  Beat Nifty in {mom['months_beat_nifty']}/{mom['n_months']} months"
        )
    else:
        sections.append("\n📆 <b>Momentum book</b>\n  <i>No completed months yet.</i>")

    circuit_str = f" | circuit: {_e(daily.get('circuit_state', '?'))}"
    dd = daily.get("drawdown_pct")
    dd_str = f" ({dd:+.1f}% from peak)" if dd is not None else ""
    cagr = daily.get("cagr_to_date_pct")
    nifty_ret = daily.get("nifty_period_return_pct")
    if cagr is not None and nifty_ret is not None:
        cagr_str = f"CAGR-to-date {_code(f'{cagr:+.2f}%')} vs Nifty {_code(f'{nifty_ret:+.2f}%')} (same period)"
    else:
        cagr_str = "<i>Not enough history yet.</i>"
    win_rate = daily.get("win_rate_30d_pct")
    win_rate_str = _code(f"{win_rate}%") if win_rate is not None else _code("N/A")
    equity_str = _code(f"₹{daily.get('equity', 0):,.0f}")
    pnl_str    = _code(f"₹{daily.get('realized_pnl', 0):+,.0f}")
    sections.append(
        f"\n📈 <b>Daily scanner (live book)</b>\n"
        f"  Equity {equity_str} · realized {pnl_str}\n"
        f"  {cagr_str}\n"
        f"  30d win rate: {win_rate_str}{circuit_str}{dd_str}"
    )

    sections.append("\n\n⚠️ <i>Not investment advice. Informational track record only.</i>")
    return "\n\n".join(sections)


# ── message builder ───────────────────────────────────────────────────────────

def _build_message(watchlist_data: dict) -> str:
    scan_date  = watchlist_data.get("scan_date", "?")
    scan_time  = watchlist_data.get("scan_time", "")

    if "buy_watchlist" in watchlist_data:
        buy_list     = watchlist_data.get("buy_watchlist", [])
        sell_list    = watchlist_data.get("sell_watchlist", [])
        phase_b_list = watchlist_data.get("phase_b_watchlist", [])
        nifty_ctx    = watchlist_data.get("nifty_context", "ranging").upper()
        total        = watchlist_data.get("total_screened", 0)
        warnings     = watchlist_data.get("data_quality_warnings", [])

        time_str = f" | {_e(scan_time)}" if scan_time else ""

        if not buy_list and not sell_list and not phase_b_list:
            base = (
                f"<b>NSE/BSE Stock Scanner — {_e(scan_date)}{time_str}</b>\n\n"
                f"No qualifying candidates today ({total} screened). "
                f"Nifty: {_e(nifty_ctx)}"
            )
            # Gate status is arguably MOST useful on a zero-candidate day -- append
            # it here too, not just on the has-picks path below.
            gates = watchlist_data.get("gates") or {}
            if gates:
                gs = gates.get("scanner", {}); gm = gates.get("momentum", {})
                base += (f"\n\n🚦 <i>{_e(gs.get('status_line', ''))}\n"
                         f"{_e(gm.get('status_line', ''))}</i>")
            return base

        nifty_arrow = {"UPTREND": " ↗", "DOWNTREND": " ↘", "RANGING": " ↔"}.get(nifty_ctx, "")
        fii_dii     = watchlist_data.get("fii_dii", {})
        gift_nifty  = watchlist_data.get("gift_nifty", {})
        fii_net     = fii_dii.get("fii_net_cr")
        dii_net     = fii_dii.get("dii_net_cr")
        fii_str = ""
        if fii_net is not None:
            arrow = "↑" if fii_net > 0 else "↓"
            fii_str = f" | FII {arrow}₹{abs(fii_net):.0f}cr"
            if dii_net is not None:
                d_arrow = "↑" if dii_net > 0 else "↓"
                fii_str += f" DII {d_arrow}₹{abs(dii_net):.0f}cr"
        gift_str = ""
        gap_pct = gift_nifty.get("gap_pct")
        if gap_pct is not None:
            g_arrow = "↑" if gap_pct > 0 else "↓"
            gift_str = f" | Gift {g_arrow}{abs(gap_pct):.1f}%"
        breadth      = watchlist_data.get("breadth", {})
        breadth_lbl  = breadth.get("breadth_label", "")
        breadth_icon = {"strong": "🟢", "neutral": "🟡", "weak": "🔴"}.get(breadth_lbl, "")
        breadth_str  = f" | Breadth: {breadth_icon}{_e(breadth_lbl)}" if breadth_lbl else ""
        pct50        = breadth.get("pct_above_50dma")
        if pct50 is not None and breadth_lbl:
            breadth_str += f" ({pct50}% >50DMA)"
        header = (
            f"<b>NSE/BSE Stock Scanner — {_e(scan_date)}{time_str}</b>\n"
            f"<i>Nifty 50: {_e(nifty_ctx)}{nifty_arrow} | {total} stocks screened"
            f"{_e(fii_str)}{_e(gift_str)}{breadth_str}</i>\n"
            f"{'─' * 32}"
        )
        sections = [header]

        circuit = watchlist_data.get("circuit") or {}
        loss_streak = watchlist_data.get("loss_streak") or {}
        if circuit.get("state") in ("reduced", "halted"):
            icon = "🛑" if circuit["state"] == "halted" else "⚠️"
            sections.append(
                f"\n{icon} <b>CIRCUIT: {circuit['state'].upper()}</b> — book "
                f"{circuit.get('drawdown_pct', 0):+.1f}% from peak "
                f"({'no new positions' if circuit['state'] == 'halted' else 'risk halved on new entries'})"
            )
        if loss_streak.get("in_cooldown"):
            sections.append(
                f"\n🥶 <b>LOSS-STREAK COOLDOWN</b> — {loss_streak['streak']} consecutive stop-outs, "
                f"watch-only until {_e(loss_streak.get('cooldown_until', '?'))}"
            )

        gates = watchlist_data.get("gates") or {}
        if gates:
            gs = gates.get("scanner", {}); gm = gates.get("momentum", {})
            sections.append(
                f"\n🚦 <i>{_e(gs.get('status_line', ''))}\n{_e(gm.get('status_line', ''))}</i>"
            )

        if buy_list:
            sections.append(f"\n📈 <b>BUY WATCHLIST ({len(buy_list)} picks)</b>\n")
            sections.extend(_format_buy_entry(e, i + 1) for i, e in enumerate(buy_list))

        if sell_list:
            offset = len(buy_list)
            sections.append(f"\n📉 <b>SELL / SHORT WATCHLIST ({len(sell_list)} picks)</b>\n")
            sections.extend(_format_sell_entry(e, offset + i + 1) for i, e in enumerate(sell_list))

        if phase_b_list:
            sections.append(f"\n👀 <b>WATCH LIST ({len(phase_b_list)} stocks)</b>\n")
            sections.extend(_format_phase_b_entry(e, i + 1) for i, e in enumerate(phase_b_list))

        big_movers = [e.get("ticker", "?") for e in (buy_list + sell_list) if e.get("big_mover")]
        if big_movers:
            sections.append(
                f"\n🔥 <i>BIG MOVER (high ATR, informational only — not a validated "
                f"higher-probability signal): {_e(', '.join(big_movers))}</i>"
            )

        if warnings:
            sections.append("\n⚠ <b>Data warnings:</b>\n" + "\n".join(f"  • {_e(w)}" for w in warnings))

        perf = watchlist_data.get("performance", {})
        if perf and perf.get("total_picks", 0) > 0:
            wr = perf.get("win_rate_pct")
            wr_str = f"{wr}%" if wr is not None else "N/A"
            sections.append(
                f"\n📊 <i>30d signal track record: "
                f"{perf['t1_hit']}T1 / {perf['sl_hit']}SL / {perf['open']} open "
                f"({perf['total_picks']} picks) — win rate {wr_str}</i>"
            )

        pf = watchlist_data.get("portfolio_risk")
        if pf and pf.get("n_allocated", 0) > 0:
            extra_parts = []
            if pf.get("n_sector_capped", 0):
                extra_parts.append(f"{pf['n_sector_capped']} sector-capped")
            if pf.get("n_dropped", 0):
                budget_only = pf["n_dropped"] - pf.get("n_sector_capped", 0)
                if budget_only > 0:
                    extra_parts.append(f"{budget_only} watch-only (budget)")
            dropped_str = (" · " + " · ".join(extra_parts)) if extra_parts else ""
            sections.append(
                f"\n💼 <i>Portfolio risk: ₹{pf['total_risk']:,.0f} at risk "
                f"({pf['total_risk_pct']:.1f}% of ₹{pf['capital']:,.0f}) · "
                f"₹{pf['total_deployed']:,.0f} deployed ({pf['total_deployed_pct']:.0f}%) · "
                f"{pf['n_allocated']} picks{dropped_str} · "
                f"₹{pf['budget_left']:,.0f} budget left</i>"
            )

        pf_live = watchlist_data.get("portfolio_live")
        if pf_live and pf_live.get("n_holdings", 0) >= 0:
            real_pnl = pf_live.get("realized_pnl", 0.0)
            unreal   = pf_live.get("unrealized_pnl", 0.0)
            pnl_sign = "+" if real_pnl >= 0 else ""
            holdings_str = (f" · {pf_live['n_holdings']} open: "
                            + ", ".join(pf_live.get("holdings", [])[:5])
                            ) if pf_live.get("n_holdings") else ""
            sections.append(
                f"\n📊 <i>Book: equity ₹{pf_live['equity']:,.0f} · "
                f"cash ₹{pf_live['cash']:,.0f} · "
                f"realized {pnl_sign}₹{real_pnl:,.0f} · "
                f"unrealized {'+' if unreal>=0 else ''}₹{unreal:,.0f}"
                f"{holdings_str}</i>"
            )

        sections.append("\n\n⚠️ <i>Not investment advice. Do your own due diligence.</i>")
        return "\n\n".join(sections)

    # Legacy schema fallback
    watchlist = watchlist_data.get("watchlist", [])
    total = watchlist_data.get("total_candidates_scanned", 0)

    if not watchlist:
        return (
            f"<b>NSE/BSE Stock Scanner — {_e(scan_date)}</b>\n\n"
            f"No qualifying candidates today ({total} scanned)."
        )

    header = (
        f"<b>NSE/BSE Stock Scanner — {_e(scan_date)}</b>\n"
        f"<i>{len(watchlist)} picks from {total} candidates scanned</i>\n"
        f"{'─' * 32}"
    )
    entries = [_format_entry_legacy(e, i + 1) for i, e in enumerate(watchlist)]
    footer = "\n\n⚠️ <i>Not investment advice. Do your own due diligence.</i>"
    return header + "\n\n" + "\n\n".join(entries) + footer


def _safe_chunks(text: str, limit: int = 4000) -> list[str]:
    """Split message on newlines so no chunk ever cuts inside an HTML tag."""
    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        candidate = (current + "\n" + line) if current else line
        if len(candidate.encode("utf-8")) > limit:
            if current:
                chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


async def _send(token: str, chat_id: str, text: str) -> None:
    bot = telegram.Bot(token=token)
    for chunk in _safe_chunks(text):
        await bot.send_message(
            chat_id=chat_id,
            text=chunk,
            parse_mode="HTML",
        )


def send_telegram_alert(watchlist_data: dict, mode: str = "eod") -> None:
    token   = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("[telegram] TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not set — skipping alert")
        return

    if mode == "mid_day":
        message = _build_midday_message(watchlist_data)
    elif mode == "factor":
        message = _build_factor_message(watchlist_data)
    elif mode == "scoreboard":
        message = _build_scoreboard_message(watchlist_data)
    else:
        message = _build_message(watchlist_data)

    try:
        asyncio.run(_send(token, chat_id, message))
        print("[telegram] alert sent successfully")
    except Exception as exc:
        print(f"[telegram] send failed: {exc}")
