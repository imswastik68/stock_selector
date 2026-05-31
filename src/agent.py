"""
Watchlist synthesis via LLM (Groq → Ollama → pure-Python fallback).

Backend selection via INFERENCE_BACKEND env var:
  groq   (default) — Groq cloud, free tier, qwen-qwq-32b
  ollama            — local Ollama, qwen3:8b, no internet needed
"""

from __future__ import annotations

import os
from datetime import date

from src.prompt_builder import SYSTEM_PROMPT, build_user_message
from src.response_parser import ParseError, parse_claude_response

_GROQ_BASE  = "https://api.groq.com/openai/v1"
_GROQ_MODEL = "qwen/qwen3-32b"

_OLLAMA_BASE  = "http://localhost:11434/v1"
_OLLAMA_MODEL = "qwen3:8b"

_MAX_TOKENS = 2000   # Groq TPM = input + max_tokens; ~2550 input + 2000 = 4550 < 6K free limit

MAX_WATCHLIST = 10
TARGET_MOVE = {"1-2d": 7, "5-7d": 28}
STOP_PCT    = {"1-2d": 0.04, "5-7d": 0.06}
ENTRY_BAND  = 0.015
UNCONFIRMED_SIGNALS = {"eps_surprise_15pct"}


# ── fallback helpers (pure-Python path) ──────────────────────────────────────

def _risk(candidate: dict) -> str:
    score = candidate["score"]
    is_penny = candidate["ticker"].startswith("[PENNY]")
    has_disq  = bool(candidate.get("disqualifiers"))
    is_small  = candidate.get("market_cap_cr") is not None and candidate["market_cap_cr"] < 500
    if has_disq or is_penny:
        return "HIGH"
    if score >= 7 and not is_small:
        return "LOW"
    if is_small:
        return "HIGH"
    return "MEDIUM"


def _fmt_price(p: float) -> str:
    return f"₹{p:.0f}" if p >= 100 else f"₹{p:.2f}"


def _entry_and_stop(candidate: dict) -> tuple[str, str]:
    close = candidate.get("today_close")
    if close is None or close <= 0:
        return "data unavailable", "data unavailable"
    tf   = candidate.get("timeframe", "1-2d")
    stop = close * (1 - STOP_PCT.get(tf, 0.04))
    return (
        f"{_fmt_price(close * (1 - ENTRY_BAND))}–{_fmt_price(close * (1 + ENTRY_BAND))}",
        f"below {_fmt_price(stop)}",
    )


def _label_signals(signals: list[str]) -> list[str]:
    return [f"{s} [UNCONFIRMED]" if s in UNCONFIRMED_SIGNALS else s for s in signals]


def _fallback_synthesize(candidates: list[dict], total_scanned: int, scan_date: date) -> dict:
    top = sorted(candidates, key=lambda c: c["score"], reverse=True)[:5]
    buy_watchlist = []
    for c in top:
        tf = c.get("timeframe", "1-2d")
        entry_zone, invalidation = _entry_and_stop(c)
        close = c.get("today_close") or 0
        target = _fmt_price(close * (1 + TARGET_MOVE.get(tf, 7) / 100)) if close > 0 else "N/A"
        buy_watchlist.append({
            "ticker": c["ticker"],
            "score": c["score"],
            "volatility_tags": [],
            "wyckoff_phase": "UNKNOWN",
            "wyckoff_confidence": "LOW",
            "smc_structure": "none",
            "vsa_signal": "none",
            "top_signals": _label_signals(c.get("active_signals", [])),
            "expected_move_pct": TARGET_MOVE.get(tf, 7),
            "timeframe": tf,
            "entry_zone": entry_zone,
            "target_1": target,
            "target_2": "N/A",
            "stop_loss": invalidation,
            "risk_reward": "1:2",
            "invalidation": invalidation,
            "risk": _risk(c),
            "catalyst": "none",
        })
    return {
        "scan_date": scan_date.isoformat(),
        "nifty_context": "ranging",
        "total_screened": total_scanned,
        "buy_watchlist": buy_watchlist,
        "sell_watchlist": [],
        "phase_b_watchlist": [],
        "data_quality_warnings": ["LLM unavailable — Wyckoff/SMC/VSA analysis not performed"],
    }


# ── LLM client builders ───────────────────────────────────────────────────────

def _groq_call(user_msg: str) -> str:
    from openai import OpenAI
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    client = OpenAI(base_url=_GROQ_BASE, api_key=api_key)
    print(f"[agent] calling Groq ({_GROQ_MODEL})...")
    resp = client.chat.completions.create(
        model=_GROQ_MODEL,
        max_tokens=_MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
    )
    return resp.choices[0].message.content


def _ollama_call(user_msg: str) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=_OLLAMA_BASE, api_key="ollama")
    print(f"[agent] calling Ollama ({_OLLAMA_MODEL})...")
    resp = client.chat.completions.create(
        model=_OLLAMA_MODEL,
        max_tokens=_MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
    )
    return resp.choices[0].message.content


# ── main synthesis function ───────────────────────────────────────────────────

def synthesize_watchlist(
    candidates: list[dict],
    total_scanned: int,
    scan_date: date | None = None,
    market_context: dict | None = None,
) -> dict:
    if scan_date is None:
        scan_date = date.today()

    if market_context is None:
        print("[agent] market_context not available — using pure-Python fallback")
        return _fallback_synthesize(candidates, total_scanned, scan_date)

    try:
        from openai import OpenAI  # noqa: F401 — verify package installed
    except ImportError:
        print("[agent] openai package not installed — using fallback")
        return _fallback_synthesize(candidates, total_scanned, scan_date)

    user_msg = build_user_message(
        candidates=candidates,
        market_context=market_context,
        bulk_deals=market_context.get("bulk_deals", []),
        fo_ban_delta=market_context.get("fo_ban_delta", []),
        results_calendar=market_context.get("results_calendar", []),
        scan_date=scan_date.isoformat(),
    )

    backend = os.environ.get("INFERENCE_BACKEND", "groq").lower()
    in_ci = os.environ.get("GITHUB_ACTIONS") == "true"

    if backend == "groq":
        # In CI Ollama is not running — don't waste time trying to connect
        callers = [_groq_call] if in_ci else [_groq_call, _ollama_call]
    else:
        callers = [_ollama_call] if in_ci else [_ollama_call, _groq_call]

    last_exc = None
    for caller in callers:
        try:
            raw = caller(user_msg)
            print(f"[agent] raw response ({len(raw)} chars): {raw[:800]!r}")
            print("[agent] response received, parsing...")
            result = parse_claude_response(raw, scan_date.isoformat(), total_scanned)
            b = len(result.get("buy_watchlist", []))
            s = len(result.get("sell_watchlist", []))
            p = len(result.get("phase_b_watchlist", []))
            print(f"[agent] parsed: {b} buys, {s} sells, {p} phase-B")
            return result
        except ParseError as exc:
            print(f"[agent] parse error from {caller.__name__}: {exc}")
            last_exc = exc
            break  # bad JSON from one backend → don't retry same prompt on next
        except Exception as exc:
            print(f"[agent] {caller.__name__} failed: {type(exc).__name__}: {exc} — trying next")
            last_exc = exc

    print(f"[agent] all backends failed ({last_exc}) — using pure-Python fallback")
    return _fallback_synthesize(candidates, total_scanned, scan_date)
