"""Telegram bot formatter and sender for the daily watchlist."""

from __future__ import annotations

import asyncio
import html
import math
import os

import telegram

RISK_EMOJI = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
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
    "macd_bullish_cross":    "MACD bullish cross",
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
    "thin_market":             "thin market",
    "f_group":                 "on F&O ban list",
    # technical signals added in alpha upgrade
    "obv_accumulation":        "OBV accumulation",
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

    lines = [
        f"{_b(f'{rank}. {ticker}')} {risk_icon}{price_str}",
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
    if delivery_pct is not None and "delivery_surge" in " ".join(str(s) for s in signals):
        extra.append(f"Delivery={_code(f'{delivery_pct}%')}")
    if (atr_pct_val is not None and math.isfinite(atr_pct_val)
            and price is not None and price > 0):
        atr_abs = price * atr_pct_val / 100
        if atr_abs > 0:
            shares = int(1000 / atr_abs)
            if shares > 0:
                extra.append(f"Size={_code(f'{shares}sh')} <i>(1ATR=₹{atr_abs:.0f})</i>")
    if extra:
        lines.append("  " + " | ".join(extra))
    lines += [
        "",
        f"  Entry: {_code(entry_zone)} | SL: {_code(stop)}",
        f"  T1: {_code(t1)} | T2: {_code(t2)} | R:R {_code(rr)}",
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

    lines = [
        f"{_b(f'{rank}. {ticker}')} {risk_icon}{price_str}",
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
    if (atr_pct_val is not None and math.isfinite(atr_pct_val)
            and price is not None and price > 0):
        atr_abs = price * atr_pct_val / 100
        if atr_abs > 0:
            shares = int(1000 / atr_abs)
            if shares > 0:
                extra.append(f"Size={_code(f'{shares}sh')} <i>(1ATR=₹{atr_abs:.0f})</i>")
    if extra:
        lines.append("  " + " | ".join(extra))
    lines += [
        "",
        f"  Short entry: {_code(entry_zone)} | SL: {_code(stop)}",
        f"  Cover T1: {_code(t1)} | T2: {_code(t2)} | R:R {_code(rr)}",
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
    scan_date  = data.get("scan_date", "?")
    scan_time  = data.get("scan_time", "12:30 PM IST")
    confirmed  = data.get("intraday_confirmed", {})
    all_checks = data.get("intraday_checked", {})
    sl_hits    = data.get("sl_hits", {})
    holding    = {t: v for t, v in all_checks.items() if not v.get("intraday_surge") and t not in sl_hits}

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
            action = "BUY → close long" if direction == "buy" else "SELL → close short"
            lines.append(
                f"{_b(ticker)}\n"
                f"  Action: {_code(action)}\n"
                f"  Price: {_code(f'₹{price}')} | SL: {_code(f'₹{stop}')} ({pct:+.1f}% vs SL)"
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
            return (
                f"<b>NSE/BSE Stock Scanner — {_e(scan_date)}{time_str}</b>\n\n"
                f"No qualifying candidates today ({total} screened). "
                f"Nifty: {_e(nifty_ctx)}"
            )

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
        header = (
            f"<b>NSE/BSE Stock Scanner — {_e(scan_date)}{time_str}</b>\n"
            f"<i>Nifty 50: {_e(nifty_ctx)}{nifty_arrow} | {total} stocks screened{_e(fii_str)}{_e(gift_str)}</i>\n"
            f"{'─' * 32}"
        )
        sections = [header]

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

    message = _build_midday_message(watchlist_data) if mode == "mid_day" else _build_message(watchlist_data)

    try:
        asyncio.run(_send(token, chat_id, message))
        print("[telegram] alert sent successfully")
    except Exception as exc:
        print(f"[telegram] send failed: {exc}")
