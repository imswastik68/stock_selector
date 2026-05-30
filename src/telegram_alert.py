"""Telegram bot formatter and sender for the daily watchlist."""

from __future__ import annotations

import asyncio
import os

import telegram

RISK_EMOJI = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
TIMEFRAME_LABEL = {"1-2d": "Short-term (1-2d)", "5-7d": "Swing (5-7d)", "3-5d": "Swing (3-5d)", "5-10d": "Multi-day (5-10d)"}

WYCKOFF_EMOJI = {
    "ACCUMULATION_C": "🌀", "ACCUMULATION_D": "📈", "MARKUP": "🚀",
    "DISTRIBUTION_C": "⚠️", "DISTRIBUTION_D": "📉", "MARKDOWN": "🔻",
    "UNKNOWN": "❓",
}


# ── legacy formatter (old schema with 'watchlist' key) ───────────────────────

def _format_entry(entry: dict, rank: int) -> str:
    ticker = entry.get("ticker", "?")
    score = entry.get("score", 0)
    tf = entry.get("timeframe", "?")
    target = entry.get("target_move_pct", 0)
    risk = entry.get("risk", "MEDIUM")
    entry_zone = entry.get("entry_zone", "—")
    invalidation = entry.get("invalidation", "—")
    signals = entry.get("top_signals", [])

    risk_icon = RISK_EMOJI.get(risk, "⚪")
    tf_label = TIMEFRAME_LABEL.get(tf, tf)
    signals_str = "\n    • ".join(signals) if signals else "—"

    return (
        f"*{rank}. {ticker}* {risk_icon}\n"
        f"  Score: `{score}` | {tf_label} | Target: `+{target}%`\n"
        f"  Entry: `{entry_zone}` | Stop: `{invalidation}`\n"
        f"  Signals:\n"
        f"    • {signals_str}"
    )


# ── new formatters (Wyckoff schema) ──────────────────────────────────────────

def _fmt_tags(tags: list) -> str:
    return " ".join(f"`{t}`" for t in tags) if tags else ""


def _format_buy_entry(entry: dict, rank: int) -> str:
    ticker = entry.get("ticker", "?")
    score = entry.get("score", 0)
    risk = entry.get("risk", "MEDIUM")
    phase = entry.get("wyckoff_phase", "?")
    confidence = entry.get("wyckoff_confidence", "?")
    smc = entry.get("smc_structure", "none")
    vsa = entry.get("vsa_signal", "none")
    tags = entry.get("volatility_tags", [])
    entry_zone = entry.get("entry_zone", "—")
    stop = entry.get("stop_loss", "—")
    t1 = entry.get("target_1", "—")
    t2 = entry.get("target_2", "—")
    rr = entry.get("risk_reward", "—")
    tf = entry.get("timeframe", "?")
    catalyst = entry.get("catalyst", "none")
    signals = entry.get("top_signals", [])

    risk_icon = RISK_EMOJI.get(risk, "⚪")
    phase_icon = WYCKOFF_EMOJI.get(phase, "")
    tf_label = TIMEFRAME_LABEL.get(tf, tf)
    signals_str = ", ".join(signals[:3]) if signals else "—"
    tags_str = _fmt_tags(tags)

    return (
        f"*{rank}. {ticker}* {risk_icon}\n"
        f"  Score: `{score}` | {phase_icon} `{phase}` ({confidence})\n"
        f"  SMC: `{smc}` | VSA: `{vsa}`"
        + (f" | {tags_str}" if tags_str else "") + "\n"
        f"  Entry: `{entry_zone}` | SL: `{stop}`\n"
        f"  T1: `{t1}` | T2: `{t2}` | R:R: `{rr}`\n"
        f"  {tf_label}" + (f" | Catalyst: `{catalyst}`" if catalyst and catalyst != "none" else "") + "\n"
        f"  Signals: _{signals_str}_"
    )


def _format_sell_entry(entry: dict, rank: int) -> str:
    ticker = entry.get("ticker", "?")
    score = entry.get("score", 0)
    risk = entry.get("risk", "MEDIUM")
    phase = entry.get("wyckoff_phase", "?")
    confidence = entry.get("wyckoff_confidence", "?")
    smc = entry.get("smc_structure", "none")
    vsa = entry.get("vsa_signal", "none")
    tags = entry.get("volatility_tags", [])
    entry_zone = entry.get("short_entry_zone", "—")
    stop = entry.get("stop_loss", "—")
    t1 = entry.get("cover_target_1", "—")
    t2 = entry.get("cover_target_2", "—")
    rr = entry.get("risk_reward", "—")
    tf = entry.get("timeframe", "?")
    signals = entry.get("top_signals", [])

    risk_icon = RISK_EMOJI.get(risk, "⚪")
    phase_icon = WYCKOFF_EMOJI.get(phase, "")
    tf_label = TIMEFRAME_LABEL.get(tf, tf)
    signals_str = ", ".join(signals[:3]) if signals else "—"
    tags_str = _fmt_tags(tags)

    return (
        f"*{rank}. {ticker}* {risk_icon}\n"
        f"  Score: `{score}` | {phase_icon} `{phase}` ({confidence})\n"
        f"  SMC: `{smc}` | VSA: `{vsa}`"
        + (f" | {tags_str}" if tags_str else "") + "\n"
        f"  Short entry: `{entry_zone}` | SL: `{stop}`\n"
        f"  Cover T1: `{t1}` | Cover T2: `{t2}` | R:R: `{rr}`\n"
        f"  {tf_label}\n"
        f"  Signals: _{signals_str}_"
    )


def _format_phase_b_entry(entry: dict, rank: int) -> str:
    ticker = entry.get("ticker", "?")
    phase = entry.get("phase", "?")
    trigger = entry.get("alert_trigger", "—")
    days = entry.get("estimated_days_to_phase_c", "?")
    return (
        f"*{rank}. {ticker}* — `{phase}`\n"
        f"  Trigger: _{trigger}_\n"
        f"  Est. Phase C: ~{days}"
    )


# ── message builder ───────────────────────────────────────────────────────────

def _build_message(watchlist_data: dict) -> str:
    scan_date = watchlist_data.get("scan_date", "?")

    # New schema
    if "buy_watchlist" in watchlist_data:
        buy_list = watchlist_data.get("buy_watchlist", [])
        sell_list = watchlist_data.get("sell_watchlist", [])
        phase_b_list = watchlist_data.get("phase_b_watchlist", [])
        nifty_ctx = watchlist_data.get("nifty_context", "ranging").upper()
        total = watchlist_data.get("total_screened", 0)
        warnings = watchlist_data.get("data_quality_warnings", [])

        if not buy_list and not sell_list and not phase_b_list:
            return (
                f"*NSE/BSE Stock Scanner — {scan_date}*\n\n"
                f"No qualifying candidates today ({total} scanned). "
                f"Nifty: {nifty_ctx}"
            )

        header = (
            f"*NSE/BSE Stock Scanner — {scan_date}*\n"
            f"_Nifty 50: {nifty_ctx} | {total} stocks screened_\n"
            f"{'─' * 32}"
        )

        sections = [header]

        if buy_list:
            sections.append(f"\n📈 *BUY WATCHLIST ({len(buy_list)} picks)*\n")
            sections.extend(_format_buy_entry(e, i + 1) for i, e in enumerate(buy_list))

        if sell_list:
            offset = len(buy_list)
            sections.append(f"\n📉 *SELL / SHORT WATCHLIST ({len(sell_list)} picks)*\n")
            sections.extend(_format_sell_entry(e, offset + i + 1) for i, e in enumerate(sell_list))

        if phase_b_list:
            sections.append(f"\n👀 *WATCH — PHASE B ({len(phase_b_list)} stocks forming)*\n")
            sections.extend(_format_phase_b_entry(e, i + 1) for i, e in enumerate(phase_b_list))

        if warnings:
            sections.append("\n⚠ *Data warnings:*\n" + "\n".join(f"  • {w}" for w in warnings))

        sections.append("\n\n⚠️ _This is not investment advice. Do your own due diligence._")
        return "\n\n".join(sections)

    # Legacy schema fallback
    watchlist = watchlist_data.get("watchlist", [])
    total = watchlist_data.get("total_candidates_scanned", 0)

    if not watchlist:
        return (
            f"*NSE/BSE Stock Scanner — {scan_date}*\n\n"
            f"No qualifying candidates today ({total} scanned). "
            f"Market conditions don't meet signal thresholds."
        )

    header = (
        f"*NSE/BSE Stock Scanner — {scan_date}*\n"
        f"_{len(watchlist)} picks from {total} candidates scanned_\n"
        f"{'─' * 32}"
    )
    entries = [_format_entry(e, i + 1) for i, e in enumerate(watchlist)]
    footer = "\n\n⚠️ _This is not investment advice. Do your own due diligence._"
    return header + "\n\n" + "\n\n".join(entries) + footer


async def _send(token: str, chat_id: str, text: str) -> None:
    bot = telegram.Bot(token=token)
    chunk_size = 4000
    for i in range(0, len(text), chunk_size):
        await bot.send_message(
            chat_id=chat_id,
            text=text[i : i + chunk_size],
            parse_mode="Markdown",
        )


def send_telegram_alert(watchlist_data: dict) -> None:
    """
    Format the watchlist and send it to the configured Telegram chat.
    Reads TELEGRAM_TOKEN and TELEGRAM_CHAT_ID from env.
    Silently skips if env vars are missing.
    """
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("[telegram] TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not set — skipping alert")
        return

    message = _build_message(watchlist_data)

    try:
        asyncio.run(_send(token, chat_id, message))
        print("[telegram] alert sent successfully")
    except Exception as exc:
        print(f"[telegram] send failed: {exc}")
