"""
Watchlist synthesis: deterministic Python engine + LLM narrative only.

Architecture:
  - All prices (entry/stop/target) computed from ATR in src/technicals.py — never LLM
  - Wyckoff phase classified by rule-based Python — never LLM
  - LLM role: write one factual sentence of narrative per stock only

Backend selection via INFERENCE_BACKEND env var:
  groq   (default) — Groq cloud, free tier, llama-3.3-70b-versatile
  ollama            — local Ollama, qwen3:8b, no internet needed
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import date, datetime, timezone, timedelta

_GROQ_BASE  = "https://api.groq.com/openai/v1"
_GROQ_MODEL = "llama-3.3-70b-versatile"

_OLLAMA_BASE  = "http://localhost:11434/v1"
_OLLAMA_MODEL = "qwen3:8b"

_MAX_TOKENS = 600   # narrative only — ~5 stocks × 40 tokens each = ~200 tokens output

UNCONFIRMED_SIGNALS = {"results_due"}

_BUY_PHASES  = {"ACCUMULATION_C", "ACCUMULATION_D", "MARKUP"}
_SELL_PHASES = {"DISTRIBUTION_C", "DISTRIBUTION_D", "MARKDOWN"}


# ── risk rating ───────────────────────────────────────────────────────────────

def _risk(candidate: dict, tech: dict) -> str:
    is_penny  = candidate["ticker"].startswith("[PENNY]")
    has_disq  = bool(candidate.get("disqualifiers"))
    is_small  = (candidate.get("market_cap_cr") or 9999) < 500
    confidence = tech.get("wyckoff_confidence", "LOW")
    score = candidate.get("score", 0)

    if has_disq or is_penny:
        return "HIGH"
    if confidence == "HIGH" and score >= 7 and not is_small:
        return "LOW"
    if is_small or confidence == "LOW":
        return "HIGH"
    return "MEDIUM"


def _label_signals(signals: list[str]) -> list[str]:
    return [f"{s} [UNCONFIRMED]" if s in UNCONFIRMED_SIGNALS else s for s in signals]


def _volatility_tags(beta: float, atr_pct: float, volume_ratio: float | None) -> list[str]:
    tags = []
    if math.isfinite(beta) and beta > 1.5:
        tags.append("HIGH-BETA")
    if math.isfinite(atr_pct) and atr_pct > 3.0:
        tags.append("HIGH-ATR")
    if volume_ratio and volume_ratio > 2.0:
        tags.append("VOL-SURGE")
    return tags


def _watch_reason(direction: str, adj_score: int, raw_score: int, nifty_trend: str, penalty: int = 4) -> str:
    """Human-readable explanation of why this stock is in WATCH instead of BUY/SELL."""
    if direction == "watch":
        return "Phase building — no entry signal yet"
    if nifty_trend == "downtrend" and direction == "buy":
        return (
            f"Nifty DOWNTREND — cautious on buys "
            f"(score {raw_score} → {adj_score} after {penalty}pt headwind penalty)"
        )
    if nifty_trend == "uptrend" and direction == "sell":
        return (
            f"Nifty UPTREND — cautious on shorts "
            f"(score {raw_score} → {adj_score} after {penalty}pt headwind penalty)"
        )
    return f"Score {adj_score} below entry threshold"


# ── build deterministic watchlist entries ────────────────────────────────────

def _build_entries(candidates: list[dict], market_context: dict, nifty_trend: str) -> tuple[list, list, list]:
    """
    Build buy/sell/phase_b entries from deterministic signals only.
    Returns (buy_list, sell_list, phase_b_list) — no LLM involved.
    """
    technicals: dict = market_context.get("technicals", {})
    beta_data:  dict = market_context.get("beta", {})
    atr_data:   dict = market_context.get("atr_pct", {})

    buy_list, sell_list, phase_b_list = [], [], []

    for c in candidates[:20]:
        ticker = c["ticker"]
        tech   = technicals.get(ticker, {})
        beta   = beta_data.get(ticker, float("nan"))
        atr_pct = atr_data.get(ticker, float("nan"))
        vol_ratio = c.get("volume_ratio")
        score  = c.get("score", 0)
        phase  = tech.get("wyckoff_phase", "ACCUMULATION_B")
        direction = tech.get("direction", "watch")

        # Nifty trend adjustment to score.
        # A stock breaking its 52-week high WITH volume is demonstrating genuine strength
        # against the market trend — that's relative strength, not a reason to penalise more.
        # Flat -4 on breakout stocks was silencing the strongest setups (CPPLUS, RUBICON etc
        # going to WATCH despite score 8 with confirmed 52w breakout + volume surge).
        active_signals = c.get("active_signals", [])
        adjusted_score = score
        headwind_penalty = 0
        if nifty_trend == "downtrend" and direction == "buy":
            has_breakout = "actual_52w_breakout" in active_signals
            headwind_penalty = 2 if has_breakout else 4
            adjusted_score -= headwind_penalty
        elif nifty_trend == "uptrend" and direction == "sell":
            has_breakdown = "distribution_signal" in active_signals
            headwind_penalty = 2 if has_breakdown else 4
            adjusted_score -= headwind_penalty

        vtags = _volatility_tags(beta, atr_pct, vol_ratio)
        signals = _label_signals(active_signals)
        risk = _risk(c, tech)

        phase_b_threshold = 5 if nifty_trend == "downtrend" else 4
        if direction == "watch" or adjusted_score < phase_b_threshold:
            phase_b_list.append({
                "ticker": ticker,
                "phase": phase,
                "score": score,
                "rsi": tech.get("rsi", "N/A"),
                "today_close": c.get("today_close"),
                "alert_trigger": signals[:3],
                "watch_reason": _watch_reason(
                    direction, adjusted_score, score, nifty_trend, headwind_penalty
                ),
            })
            continue

        levels = {
            "entry_zone": tech.get("entry_zone", "N/A"),
            "stop_loss":  tech.get("stop_loss", "N/A"),
            "target_1":   tech.get("target_1", "N/A"),
            "target_2":   tech.get("target_2", "N/A"),
            "risk_reward": tech.get("risk_reward", "1:2"),
        }

        # Pivot points (daily/weekly/monthly)
        pivots = {k: tech[k] for k in tech if k.endswith(("_pivot", "_r1", "_r2", "_s1", "_s2"))}

        base = {
            "ticker": ticker,
            "score": adjusted_score,
            "today_close": c.get("today_close"),
            "volatility_tags": vtags,
            "wyckoff_phase": phase,
            "wyckoff_confidence": tech.get("wyckoff_confidence", "MEDIUM"),
            "rsi": tech.get("rsi", "N/A"),
            "macd_signal": tech.get("macd_signal", "none"),
            "candlestick_patterns": tech.get("candlestick_patterns", []),
            "top_signals": signals,
            "timeframe": c.get("timeframe", "1-2d"),
            "risk": risk,
            "promoter_pct": c.get("promoter_pct"),
            "options_pcr": c.get("options_pcr"),
            "narrative": "",  # filled in by LLM below
            **levels,
            **pivots,
        }

        if direction == "buy" and phase in _BUY_PHASES:
            base["expected_move_pct"] = round((atr_pct or 2) * 4, 1)
            buy_list.append(base)
        elif direction == "sell" and phase in _SELL_PHASES:
            base["expected_drop_pct"] = round((atr_pct or 2) * 4, 1)
            sell_list.append(base)

    # Sort by score descending, cap combined at 10
    combined = sorted(buy_list + sell_list, key=lambda e: e["score"], reverse=True)[:10]
    buy_list  = [e for e in combined if e.get("wyckoff_phase") in _BUY_PHASES]
    sell_list = [e for e in combined if e.get("wyckoff_phase") in _SELL_PHASES]

    # Sort watch list by raw score so the strongest setups appear first, cap at 8
    phase_b_list.sort(key=lambda e: e.get("score", 0), reverse=True)
    return buy_list, sell_list, phase_b_list[:8]


# ── LLM narrative request ─────────────────────────────────────────────────────

_NARRATIVE_SYSTEM = (
    "You are an Indian stock market analyst. "
    "For each stock given, write ONE factual sentence explaining why it is notable TODAY "
    "based only on the signals listed. Be specific — mention the signal and what it implies. "
    "Return ONLY a JSON array: "
    '[{"ticker":"TICKER.NS","narrative":"one sentence here"},...] '
    "No preamble, no markdown fences."
)


def _build_narrative_request(entries: list[dict], scan_date: str) -> str:
    lines = [f"scan_date={scan_date}"]
    for e in entries:
        lines.append(
            f"TICKER:{e['ticker']} PHASE:{e['wyckoff_phase']} "
            f"RSI:{e.get('rsi','?')} MACD:{e.get('macd_signal','?')} "
            f"SIGNALS:{','.join(e.get('top_signals',[]))}"
        )
    return "\n".join(lines)


def _extract_narratives(raw: str) -> dict[str, str]:
    """Parse LLM narrative response → {ticker: narrative}. Never raises."""
    raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
    # strip fences
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if m:
        raw = m.group(1)
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            return {item["ticker"]: item.get("narrative", "") for item in arr if "ticker" in item}
    except Exception:
        pass
    return {}


def _llm_call(user_msg: str, backend: str, in_ci: bool) -> str | None:
    try:
        from openai import OpenAI
    except ImportError:
        return None

    callers = []
    if backend == "groq":
        callers = [("groq", _GROQ_BASE, _GROQ_MODEL, os.environ.get("GROQ_API_KEY", ""))]
        if not in_ci:
            callers.append(("ollama", _OLLAMA_BASE, _OLLAMA_MODEL, "ollama"))
    else:
        callers = [("ollama", _OLLAMA_BASE, _OLLAMA_MODEL, "ollama")]
        if not in_ci:
            callers.append(("groq", _GROQ_BASE, _GROQ_MODEL, os.environ.get("GROQ_API_KEY", "")))

    for name, base, model, key in callers:
        if name == "groq" and not key:
            continue
        try:
            print(f"[agent] calling {name} ({model}) for narratives...")
            client = OpenAI(base_url=base, api_key=key)
            resp = client.chat.completions.create(
                model=model,
                max_tokens=_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": _NARRATIVE_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
            )
            raw = resp.choices[0].message.content
            print(f"[agent] narrative response ({len(raw)} chars): {raw[:300]!r}")
            return raw
        except Exception as exc:
            print(f"[agent] {name} failed: {exc} — trying next")

    return None


# ── main synthesis function ───────────────────────────────────────────────────

def synthesize_watchlist(
    candidates: list[dict],
    total_scanned: int,
    scan_date: date | None = None,
    market_context: dict | None = None,
) -> dict:
    if scan_date is None:
        scan_date = date.today()

    nifty_trend = "ranging"
    if market_context:
        nifty_trend = market_context.get("nifty_structure", {}).get("trend", "ranging")

    if not market_context or not candidates:
        print("[agent] no context or candidates — empty output")
        return {
            "scan_date": scan_date.isoformat(),
            "nifty_context": nifty_trend,
            "total_screened": total_scanned,
            "buy_watchlist": [],
            "sell_watchlist": [],
            "phase_b_watchlist": [],
            "data_quality_warnings": ["No market context available"],
        }

    # Step 1: Build all entries deterministically (no LLM)
    buy_list, sell_list, phase_b_list = _build_entries(candidates, market_context, nifty_trend)

    actionable = buy_list + sell_list
    print(f"[agent] deterministic: {len(buy_list)} buys, {len(sell_list)} sells, {len(phase_b_list)} phase-B")

    # Step 2: Ask LLM for one narrative sentence per actionable stock (optional enrichment)
    if actionable:
        narrative_req = _build_narrative_request(actionable, scan_date.isoformat())
        backend = os.environ.get("INFERENCE_BACKEND", "groq").lower()
        in_ci = os.environ.get("GITHUB_ACTIONS") == "true"
        raw = _llm_call(narrative_req, backend, in_ci)
        if raw:
            narratives = _extract_narratives(raw)
            for entry in actionable:
                t = entry["ticker"]
                if t in narratives:
                    entry["narrative"] = narratives[t]
                    print(f"[agent] narrative for {t}: {narratives[t][:80]}")

    IST = timezone(timedelta(hours=5, minutes=30))
    scan_time = datetime.now(IST).strftime("%I:%M %p IST").lstrip("0")

    return {
        "scan_date": scan_date.isoformat(),
        "scan_time": scan_time,
        "nifty_context": nifty_trend,
        "total_screened": total_scanned,
        "buy_watchlist": buy_list,
        "sell_watchlist": sell_list,
        "phase_b_watchlist": phase_b_list,
        "data_quality_warnings": [],
    }
