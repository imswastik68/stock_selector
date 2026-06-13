"""
Fetch all messages from the Stock Scanner bot and parse them into structured picks.

Outputs:
  scripts/telegram_history.json  — raw messages (id, date, type, text)
  scripts/telegram_picks.json    — structured picks ready for analysis

Run: python scripts/fetch_telegram_history.py
Re-running is safe — fetches incrementally (only new messages since last run).
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import timezone
from pathlib import Path

try:
    from telethon import TelegramClient
except ImportError:
    sys.exit("Run: pip install telethon")

# ── config ────────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).parent
_ROOT        = _HERE.parent
_ENV         = _ROOT / ".env"
_OUTPUTS     = _ROOT / "outputs"
_SESSION     = _HERE / "tg_session"
_HISTORY_OUT = _OUTPUTS / "telegram_history.json"
_PICKS_OUT   = _OUTPUTS / "telegram_picks.json"

BOT_ENTITY_ID = 8987715797   # 'Stock Scanner' bot — discovered via iter_dialogs


def _load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if _ENV.exists():
        for line in _ENV.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


# ── message-type classifier ───────────────────────────────────────────────────

def _classify(text: str) -> str:
    if "BUY WATCHLIST" in text:
        return "buy"
    if "SELL" in text and "WATCHLIST" in text:
        return "sell"
    if "Mid-day" in text or "INTRADAY CONFIRMED" in text or "STOP-LOSS HIT" in text:
        return "midday"
    if "WATCH LIST" in text or "PHASE-B" in text:
        return "watch"
    return "other"


# ── pick parser ───────────────────────────────────────────────────────────────

def _p(s: str | None) -> float | None:
    """Parse a price string like '1,234.56' → float."""
    if not s:
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _parse_picks(text: str, date: str, direction: str) -> list[dict]:
    """
    Parse individual stock entries from a BUY or SELL alert.
    Returns a list of structured pick dicts.
    """
    picks: list[dict] = []

    # Split text into per-stock chunks at numbered bold headers
    chunks = re.split(r"\n(?=\*\*\d+\.)", text)

    for chunk in chunks:
        # Ticker: **1. SOTL.NS** or **1. SOTL.NS** 🟢
        t_m = re.search(r"\*\*\d+\.\s+([A-Z0-9&.]+)\*\*", chunk)
        if not t_m:
            continue
        ticker = t_m.group(1)
        if not ticker.endswith(".NS"):
            ticker = ticker + ".NS"

        # Close price: `₹514.60`
        price_m = re.search(r"`₹([\d,]+\.?\d*)`", chunk)

        # Score, RSI, Confidence
        score_m = re.search(r"Score:\s*`(\d+)`", chunk)
        rsi_m   = re.search(r"RSI:\s*`([\d.]+)`", chunk)
        conf_m  = re.search(r"Confidence:\s*([A-Z]+)", chunk)

        # Wyckoff phase
        phase_m = re.search(r"🚀|🌀|📈|⚠️|📉|🔻|📊", chunk)
        phase_text_m = re.search(
            r"(?:🚀|🌀|📈|⚠️|📉|🔻|📊)\s+\*\*([^*\n]+)\*\*", chunk
        )

        # MACD
        macd_m = re.search(r"MACD:\s*`([^`]+)`", chunk)

        # Entry zone: `₹550.31-₹578.29` or `₹550.31–₹578.29`
        entry_m = re.search(
            r"Entry:\s*`₹([\d,]+\.?\d*)[–\-]₹([\d,]+\.?\d*)`", chunk
        )
        # SL
        sl_m = re.search(r"SL:\s*`₹([\d,]+\.?\d*)`", chunk)
        # T1, T2
        t1_m = re.search(r"T1:\s*`₹([\d,]+\.?\d*)`", chunk)
        t2_m = re.search(r"T2:\s*`₹([\d,]+\.?\d*)`", chunk)
        # R:R
        rr_m = re.search(r"R:R\s*`([^`]+)`", chunk)
        # Timeframe
        tf_m = re.search(r"__([^_]+(?:term|Swing|Multi-day)[^_]*)__", chunk)
        # Signals
        sig_m = re.search(r"Signals:\s*__([^_\n]+)__", chunk)
        # Narrative
        nar_m = re.search(r"💬\s+__([^_]+)__", chunk)
        # Promoter %
        prom_m = re.search(r"Promoter=`([\d.]+)%`", chunk)
        # Delivery %
        deliv_m = re.search(r"Delivery=`([\d.]+)%`", chunk)
        # 6m momentum
        mom_m = re.search(r"6m=`([+-]?[\d.]+)%`", chunk)
        # RS quality
        rs_m = re.search(r"RS=`([\d.]+)`", chunk)
        # Risk emoji
        risk = "LOW" if "🟢" in chunk else ("HIGH" if "🟠" in chunk or "🔴" in chunk else "MEDIUM")

        entry_lo = _p(entry_m.group(1)) if entry_m else None
        entry_hi = _p(entry_m.group(2)) if entry_m else None
        entry_mid = round((entry_lo + entry_hi) / 2, 2) if entry_lo and entry_hi else None

        picks.append({
            "date":        date,
            "direction":   direction,
            "ticker":      ticker,
            "risk":        risk,
            "close":       _p(price_m.group(1)) if price_m else None,
            "score":       int(score_m.group(1)) if score_m else None,
            "rsi":         float(rsi_m.group(1)) if rsi_m else None,
            "confidence":  conf_m.group(1) if conf_m else None,
            "phase":       phase_text_m.group(1).strip() if phase_text_m else None,
            "macd":        macd_m.group(1) if macd_m else None,
            "entry_lo":    entry_lo,
            "entry_hi":    entry_hi,
            "entry_mid":   entry_mid,
            "sl":          _p(sl_m.group(1)) if sl_m else None,
            "t1":          _p(t1_m.group(1)) if t1_m else None,
            "t2":          _p(t2_m.group(1)) if t2_m else None,
            "rr":          rr_m.group(1) if rr_m else None,
            "timeframe":   tf_m.group(1).strip() if tf_m else None,
            "signals":     [s.strip() for s in sig_m.group(1).split(",")] if sig_m else [],
            "narrative":   nar_m.group(1).strip() if nar_m else None,
            "promoter_pct": float(prom_m.group(1)) if prom_m else None,
            "delivery_pct": float(deliv_m.group(1)) if deliv_m else None,
            "momentum_6m":  float(mom_m.group(1)) if mom_m else None,
            "rs_quality":   float(rs_m.group(1)) if rs_m else None,
            "outcome":     "open",   # filled in by analysis script
            "outcome_date": None,
        })

    return picks


def _parse_header(text: str) -> dict:
    """Extract scan-level context from the alert header."""
    date_m    = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    time_m    = re.search(r"(\d{1,2}:\d{2}\s*(?:AM|PM)\s*IST)", text)
    nifty_m   = re.search(r"Nifty 50:\s*([A-Z]+)", text)
    screened_m= re.search(r"(\d+)\s*stocks\s*screened", text)
    fii_m     = re.search(r"FII\s*[↑↓]₹([\d.]+)cr", text)
    dii_m     = re.search(r"DII\s*[↑↓]₹([\d.]+)cr", text)
    return {
        "scan_date":   date_m.group(1) if date_m else None,
        "scan_time":   time_m.group(1) if time_m else None,
        "nifty_trend": nifty_m.group(1) if nifty_m else None,
        "screened":    int(screened_m.group(1)) if screened_m else None,
        "fii_cr":      float(fii_m.group(1)) if fii_m else None,
        "dii_cr":      float(dii_m.group(1)) if dii_m else None,
    }


# ── incremental fetch ─────────────────────────────────────────────────────────

def _load_existing() -> tuple[list[dict], list[dict]]:
    history = json.loads(_HISTORY_OUT.read_text()) if _HISTORY_OUT.exists() else []
    picks   = json.loads(_PICKS_OUT.read_text())   if _PICKS_OUT.exists()   else []
    return history, picks


async def fetch():
    env = _load_env()
    api_id   = int(env.get("TG_API_ID", 0))
    api_hash = env.get("TG_API_HASH", "")
    if not api_id or not api_hash:
        sys.exit("TG_API_ID / TG_API_HASH missing from .env")

    history, existing_picks = _load_existing()
    seen_ids = {m["id"] for m in history}

    client = TelegramClient(str(_SESSION), api_id, api_hash)
    await client.start()

    new_msgs: list[dict] = []
    async for msg in client.iter_messages(BOT_ENTITY_ID, limit=None):
        if not msg.text or msg.id in seen_ids:
            continue
        new_msgs.append({
            "id":   msg.id,
            "date": msg.date.astimezone(timezone.utc).isoformat(),
            "type": _classify(msg.text),
            "text": msg.text,
        })

    await client.disconnect()

    if not new_msgs:
        print("No new messages.")
    else:
        new_msgs.sort(key=lambda m: m["date"])
        history.extend(new_msgs)
        history.sort(key=lambda m: m["date"])
        print(f"Fetched {len(new_msgs)} new messages ({len(history)} total)")

    # Parse picks from all BUY/SELL messages (re-parse all to be safe)
    all_picks: list[dict] = []
    for m in history:
        if "type" not in m:
            m["type"] = _classify(m["text"])
        if m["type"] not in ("buy", "sell"):
            continue
        date = m["date"][:10]
        hdr  = _parse_header(m["text"])
        for pick in _parse_picks(m["text"], date, m["type"]):
            pick["msg_id"]    = m["id"]
            pick["scan_time"] = hdr.get("scan_time")
            pick["nifty_trend"] = hdr.get("nifty_trend")
            pick["screened"]  = hdr.get("screened")
            all_picks.append(pick)

    # Deduplicate: keep latest message's parse for each (date, ticker, direction)
    seen: dict[tuple, dict] = {}
    for p in all_picks:
        k = (p["date"], p["ticker"], p["direction"])
        if k not in seen or p["msg_id"] > seen[k]["msg_id"]:
            seen[k] = p
    final_picks = sorted(seen.values(), key=lambda p: (p["date"], p["direction"], p.get("score") or 0), reverse=False)
    final_picks.sort(key=lambda p: p["date"])

    _HISTORY_OUT.write_text(json.dumps(history,      indent=2, ensure_ascii=False))
    _PICKS_OUT.write_text(  json.dumps(final_picks,  indent=2, ensure_ascii=False))
    print(f"Saved {len(history)} messages → {_HISTORY_OUT.name}")
    print(f"Saved {len(final_picks)} structured picks → {_PICKS_OUT.name}")


asyncio.run(fetch())
