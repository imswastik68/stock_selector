"""Telegram bot formatter and sender for the daily watchlist."""

from __future__ import annotations

import asyncio
import html
import os

import telegram

RISK_EMOJI = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
TIMEFRAME_LABEL = {
    "1-2d": "Short-term (1-2d)", "5-7d": "Swing (5-7d)",
    "3-5d": "Swing (3-5d)", "5-10d": "Multi-day (5-10d)",
}
WYCKOFF_EMOJI = {
    "ACCUMULATION_C": "🌀", "ACCUMULATION_D": "📈", "MARKUP": "🚀",
    "DISTRIBUTION_C": "⚠️", "DISTRIBUTION_D": "📉", "MARKDOWN": "🔻",
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


def _fmt_signals(signals: list) -> str:
    if not signals:
        return "—"
    # replace underscores with spaces for readability
    return ", ".join(s.replace("_", " ") for s in signals[:3])


# ── new formatters ────────────────────────────────────────────────────────────

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


def _format_buy_entry(entry: dict, rank: int) -> str:
    ticker     = entry.get("ticker", "?")
    score      = entry.get("score", 0)
    risk       = entry.get("risk", "MEDIUM")
    phase      = entry.get("wyckoff_phase", "?")
    confidence = entry.get("wyckoff_confidence", "?")
    rsi        = entry.get("rsi", "?")
    macd       = entry.get("macd_signal", "none")
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

    risk_icon  = RISK_EMOJI.get(risk, "⚪")
    phase_icon = WYCKOFF_EMOJI.get(phase, "")
    tf_label   = TIMEFRAME_LABEL.get(tf, tf)
    tags_str   = _fmt_tags(tags)
    pivot_str  = _fmt_pivots(entry)
    candle_str = _fmt_candles(candles)

    lines = [
        f"{_b(f'{rank}. {ticker}')} {risk_icon}",
        f"  Score: {_code(score)} | {phase_icon} {_code(phase)} ({_e(confidence)})",
        f"  RSI: {_code(rsi)} | MACD: {_code(macd)}" + (f" | {tags_str}" if tags_str else ""),
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
    if extra:
        lines.append("  " + " | ".join(extra))
    lines += [
        f"  Entry: {_code(entry_zone)} | SL: {_code(stop)}",
        f"  T1: {_code(t1)} | T2: {_code(t2)} | R:R: {_code(rr)}",
        f"  {_e(tf_label)}",
        f"  Signals: {_i(_fmt_signals(signals))}",
    ]
    if narrative:
        lines.append(f"  💬 {_i(narrative)}")
    return "\n".join(lines)


def _format_sell_entry(entry: dict, rank: int) -> str:
    ticker     = entry.get("ticker", "?")
    score      = entry.get("score", 0)
    risk       = entry.get("risk", "MEDIUM")
    phase      = entry.get("wyckoff_phase", "?")
    confidence = entry.get("wyckoff_confidence", "?")
    rsi        = entry.get("rsi", "?")
    macd       = entry.get("macd_signal", "none")
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

    risk_icon  = RISK_EMOJI.get(risk, "⚪")
    phase_icon = WYCKOFF_EMOJI.get(phase, "")
    tf_label   = TIMEFRAME_LABEL.get(tf, tf)
    tags_str   = _fmt_tags(tags)
    pivot_str  = _fmt_pivots(entry)
    candle_str = _fmt_candles(candles)

    lines = [
        f"{_b(f'{rank}. {ticker}')} {risk_icon}",
        f"  Score: {_code(score)} | {phase_icon} {_code(phase)} ({_e(confidence)})",
        f"  RSI: {_code(rsi)} | MACD: {_code(macd)}" + (f" | {tags_str}" if tags_str else ""),
    ]
    if candle_str:
        lines.append(f"  Candle: {_i(candle_str)}")
    if pivot_str:
        lines.append(f"  Pivots: {pivot_str}")
    if pcr is not None:
        lines.append(f"  PCR={_code(pcr)}")
    lines += [
        f"  Short entry: {_code(entry_zone)} | SL: {_code(stop)}",
        f"  Cover T1: {_code(t1)} | Cover T2: {_code(t2)} | R:R: {_code(rr)}",
        f"  {_e(tf_label)}",
        f"  Signals: {_i(_fmt_signals(signals))}",
    ]
    if narrative:
        lines.append(f"  💬 {_i(narrative)}")
    return "\n".join(lines)


def _format_phase_b_entry(entry: dict, rank: int) -> str:
    ticker  = entry.get("ticker", "?")
    phase   = entry.get("phase", "?")
    trigger = entry.get("alert_trigger", "—")
    days    = entry.get("estimated_days_to_phase_c", "?")
    rsi     = entry.get("rsi", "")
    rsi_str = f" | RSI: {_code(rsi)}" if rsi else ""
    return (
        f"{_b(f'{rank}. {ticker}')} — {_code(phase)}{rsi_str}\n"
        f"  Trigger: {_i(_e(str(trigger)).replace('_', ' '))}\n"
        f"  Est. Phase C: ~{_e(str(days))}"
    )


# ── legacy formatter ──────────────────────────────────────────────────────────

def _format_entry_legacy(entry: dict, rank: int) -> str:
    ticker     = entry.get("ticker", "?")
    score      = entry.get("score", 0)
    tf         = entry.get("timeframe", "?")
    target     = entry.get("target_move_pct", 0)
    risk       = entry.get("risk", "MEDIUM")
    entry_zone = entry.get("entry_zone", "—")
    invalidation = entry.get("invalidation", "—")
    signals    = entry.get("top_signals", [])

    risk_icon = RISK_EMOJI.get(risk, "⚪")
    tf_label  = TIMEFRAME_LABEL.get(tf, tf)
    sigs_str  = "\n    • ".join(_fmt_signals([s]) for s in signals) if signals else "—"

    return (
        f"{_b(f'{rank}. {ticker}')} {risk_icon}\n"
        f"  Score: {_code(score)} | {_e(tf_label)} | Target: {_code(f'+{target}%')}\n"
        f"  Entry: {_code(entry_zone)} | Stop: {_code(invalidation)}\n"
        f"  Signals:\n    • {sigs_str}"
    )


# ── mid-day message builder ───────────────────────────────────────────────────

def _build_midday_message(data: dict) -> str:
    scan_date  = data.get("scan_date", "?")
    scan_time  = data.get("scan_time", "12:30")
    confirmed  = data.get("intraday_confirmed", {})
    all_checks = data.get("intraday_checked", {})
    holding    = {t: v for t, v in all_checks.items() if not v.get("intraday_surge")}

    header = (
        f"<b>Mid-day Momentum Check — {_e(scan_date)}</b>\n"
        f"<i>Snapshot at {_e(scan_time)} IST | {len(all_checks)} candidates checked</i>\n"
        f"{'─' * 32}"
    )
    lines = [header]

    if confirmed:
        lines.append(f"\n🔥 <b>INTRADAY CONFIRMED ({len(confirmed)} stocks)</b>\n")
        for i, (ticker, v) in enumerate(confirmed.items(), 1):
            direction = "▲" if v["price_above_prev_close"] else "▼"
            pct = v.get("pct_vs_prev_close", 0)
            vol_str   = f"{v['volume_ratio_projected']:.1f}x"
            price_str = f"Rs{v['price_current']}"
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
    scan_date = watchlist_data.get("scan_date", "?")

    if "buy_watchlist" in watchlist_data:
        buy_list    = watchlist_data.get("buy_watchlist", [])
        sell_list   = watchlist_data.get("sell_watchlist", [])
        phase_b_list = watchlist_data.get("phase_b_watchlist", [])
        nifty_ctx   = watchlist_data.get("nifty_context", "ranging").upper()
        total       = watchlist_data.get("total_screened", 0)
        warnings    = watchlist_data.get("data_quality_warnings", [])

        if not buy_list and not sell_list and not phase_b_list:
            return (
                f"<b>NSE/BSE Stock Scanner — {_e(scan_date)}</b>\n\n"
                f"No qualifying candidates today ({total} scanned). "
                f"Nifty: {_e(nifty_ctx)}"
            )

        header = (
            f"<b>NSE/BSE Stock Scanner — {_e(scan_date)}</b>\n"
            f"<i>Nifty 50: {_e(nifty_ctx)} | {total} stocks screened</i>\n"
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
            sections.append(f"\n👀 <b>WATCH — PHASE B ({len(phase_b_list)} forming)</b>\n")
            sections.extend(_format_phase_b_entry(e, i + 1) for i, e in enumerate(phase_b_list))

        if warnings:
            sections.append("\n⚠ <b>Data warnings:</b>\n" + "\n".join(f"  • {_e(w)}" for w in warnings))

        sections.append("\n\n⚠️ <i>This is not investment advice. Do your own due diligence.</i>")
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
    footer = "\n\n⚠️ <i>This is not investment advice. Do your own due diligence.</i>"
    return header + "\n\n" + "\n\n".join(entries) + footer


def _safe_chunks(text: str, limit: int = 4000) -> list[str]:
    """Split message on newlines so no chunk ever cuts inside an HTML tag."""
    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        # +1 for the newline we'll re-add
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
