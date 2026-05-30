"""Telegram bot formatter and sender for the daily watchlist."""

from __future__ import annotations

import asyncio
import os

import telegram

RISK_EMOJI = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
TIMEFRAME_LABEL = {"1-2d": "Short-term (1-2d)", "5-7d": "Swing (5-7d)"}


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


def _build_message(watchlist_data: dict) -> str:
    watchlist = watchlist_data.get("watchlist", [])
    scan_date = watchlist_data.get("scan_date", "?")
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
    # Telegram max message length is 4096 chars; split if needed
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
