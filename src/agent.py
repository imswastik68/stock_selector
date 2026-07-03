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

import calendar
import json
import math
import os
import re
from datetime import date, datetime, timezone, timedelta

from src.scorer import BEARISH_EVENT_WEIGHTS, REGIME_WEIGHTS

_GROQ_BASE  = "https://api.groq.com/openai/v1"
_GROQ_MODEL = "llama-3.3-70b-versatile"

_OLLAMA_BASE  = "http://localhost:11434/v1"
_OLLAMA_MODEL = "qwen3:8b"

_MAX_TOKENS = 600   # narrative only — ~5 stocks × 40 tokens each = ~200 tokens output

UNCONFIRMED_SIGNALS = {"results_due"}

_BUY_PHASES  = {"ACCUMULATION_C", "ACCUMULATION_D", "MARKUP"}
_SELL_PHASES = {"DISTRIBUTION_C", "DISTRIBUTION_D", "MARKDOWN"}

# A bullish-only signal firing on a SELL candidate (or a bearish-only signal firing
# on a BUY candidate) is a contradiction, not an edge — strip its weight back out
# before the trend-headwind penalty. delivery_surge deliberately excluded from both:
# institutional conviction is direction-ambiguous.
BULLISH_ONLY_WEIGHTS = {
    "actual_52w_breakout": 3, "bulk_deal_fii_dii": 3, "promoter_buying": 3,
    "sast_insider_buying": 3, "consolidation_breakout": 3, "buyback_announced": 2,
    "contract_win": 2, "options_long_buildup": 1, "options_short_covering": 1,
    "options_pcr_fear": 1,
}
# distribution_signal (-1) and options_short_buildup (-2) were DISQUALIFIER_WEIGHTS
# (universal penalty) before Phase 3; they're now positive in BEARISH_EVENT_WEIGHTS
# so sells get properly rewarded for them, applied universally by _compute_score
# since direction isn't known yet at scoring time. Simply cancelling that addition
# for buys (subtracting the same weight) nets to 0, not the old -1/-2 penalty —
# add the original disqualifier magnitude back on top so buy-side behavior for
# these two signals is unchanged from before Phase 3. The other 4 bearish-event
# signals (actual_52w_breakdown, heavy_selling, consolidation_breakdown,
# bulk_deal_fii_sell) are new in Phase 3 and had no prior buy-side penalty, so
# they net to a neutral 0 on buys — cancel only, no extra penalty.
_BEARISH_EVENT_PRE_PHASE3_DISQUALIFIER = {"distribution_signal": 1, "options_short_buildup": 2}


def _cleanup_weight(sig: str, base_weights: dict, regime_override: dict, pre3: dict | None = None) -> int:
    """
    Amount to subtract from a candidate's score for a signal that contradicts its
    direction (a bearish-event signal on a buy, or a bullish-only signal on a sell).

    Uses the regime-effective weight (override REPLACES the base weight, same
    semantics as src/scorer.py:_compute_score's merge — not additive) so this stays
    correct as REGIME_WEIGHTS gets recalibrated: a signal that's already a heavy
    regime penalty (effective weight < 0) isn't double-penalized here, and a signal
    boosted into a large regime reward gets fully cancelled, not just its static
    base amount.

    pre3 (buy-side only): the pre-Phase-3 disqualifier magnitude for the 2 signals
    that were DISQUALIFIER_WEIGHTS before Phase 3 moved them into BEARISH_EVENT_WEIGHTS.
    Only added when the effective weight is still >=0 (not already a penalty) —
    restores the original buy-side penalty instead of just cancelling scorer's add.
    """
    eff = regime_override.get(sig, base_weights.get(sig, 0))
    cleanup = max(0, eff)
    if pre3 and eff >= 0:
        cleanup += pre3.get(sig, 0)
    return cleanup

# Sell entry threshold — set by Phase 0 mining, RE-CONFIRMED by Phase 5 on the full
# 156-week/185,673-trade backtest (outputs/big_mover_analysis.json:
# downtrend_short_edge.verdict == "short_edge_negative" both passes — Phase 0 (52w):
# n=582, -3.52% vs longs +0.33%; Phase 5 (156w): n=892, -5.41% vs longs +0.45%, i.e.
# WORSE on the larger sample, not better). Downtrend sells need a materially higher
# bar than buys. Non-downtrend regimes are unmeasured so far — kept at the existing
# buy-side value. See SHORT_PIPELINE_LIVE below — this threshold is currently
# secondary to that harder gate.
SELL_ENTRY_THRESHOLD_DOWNTREND = 6
SELL_ENTRY_THRESHOLD_ELSE = 4

# Short pipeline (Phase 3) is built but NOT LIVE — Phase 5's 156-week backtest found
# the short thesis has no measured edge yet, even in its textbook case. Isolated by
# direction, sign-stable NEGATIVE in both train and holdout 70/30 split for every
# OHLCV-derivable bearish-event signal: actual_52w_breakdown -2.50%/-2.48%,
# heavy_selling -2.44%/-2.67%, distribution_signal -1.76%/-0.43% (return_pct lift on
# SELL trades only). Matches the direct downtrend_short_edge comparison above. All
# sells route to phase_b (informational only) until this changes. Flip to True and
# re-run scripts/mine_big_movers.py's downtrend_short_edge check once more data
# accumulates or the market regime shifts out of the current correction.
SHORT_PIPELINE_LIVE = False

# BIG MOVER tier — Phase 0 mining found no atr/score/signal combination that clears
# the 1.5x-holdout-lift ship bar; RE-CONFIRMED by Phase 5 on the 156-week backtest
# (best candidate atr>=5+score>=6: holdout P(big)=40.2%, decisively above the 30.4%
# needed, but holdout n=199 — one short of the n>=200 floor, and picks/day=3.75
# exceeds the <=3.0 cap). Ships as an ATR-only INFORMATIONAL flag with no
# target/exit override: "more volatile," not a validated "more likely to hit
# +-10%." Worth re-running once more data closes that n=199-vs-200 near-miss.
BIG_MOVER_GATE = {"min_atr_pct": 5.0, "status": "no_gate_found"}
BIG_MOVER_MAX_PER_DAY = 3


def _is_big_mover(atr_pct: float | None) -> bool:
    return atr_pct is not None and math.isfinite(atr_pct) and atr_pct >= BIG_MOVER_GATE["min_atr_pct"]


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


def _is_expiry_proximity(today: date) -> bool:
    """
    True on days near F&O expiry where artificial OI unwinding inflates SELL signals.
      - Nifty weekly: every Tuesday (effective Sep 2025). Monday included as pre-expiry.
      - Monthly stock F&O: last Thursday of the month (unchanged SEBI schedule).
    Only suppresses SELL entries by -1; BUY entries are unaffected.
    """
    wd = today.weekday()  # Mon=0, Tue=1, Wed=2, Thu=3, Fri=4
    if wd in (0, 1):  # Monday / Tuesday — Nifty weekly expiry proximity
        return True
    if wd == 3:       # Thursday — check if it's the last Thursday of the month
        days_left = calendar.monthrange(today.year, today.month)[1] - today.day
        if days_left < 7:
            return True
    return False


# ── build deterministic watchlist entries ────────────────────────────────────

def _build_entries(candidates: list[dict], market_context: dict, nifty_trend: str) -> tuple[list, list, list]:
    """
    Build buy/sell/phase_b entries from deterministic signals only.
    Returns (buy_list, sell_list, phase_b_list) — no LLM involved.
    """
    technicals: dict = market_context.get("technicals", {})
    beta_data:  dict = market_context.get("beta", {})
    atr_data:   dict = market_context.get("atr_pct", {})
    fii_dii:    dict = market_context.get("fii_dii", {})
    gift_nifty: dict = market_context.get("gift_nifty", {})
    fo_set:     set  = market_context.get("fo_eligible") or set()

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

        # Direction-aware cleanup: a bullish-only signal contradicting a SELL
        # candidate (or vice versa) isn't an edge for that trade — strip it back out.
        # Uses the SAME regime-merge _compute_score applies (override replaces the
        # base weight, doesn't stack with it) — a static-weight cleanup would either
        # double-count a signal that's already a heavy regime penalty, or under-cancel
        # one boosted into a large regime reward. See _cleanup_weight docstring.
        regime_override = REGIME_WEIGHTS.get(nifty_trend) or {}
        if direction == "buy":
            adjusted_score -= sum(
                _cleanup_weight(s, BEARISH_EVENT_WEIGHTS, regime_override, _BEARISH_EVENT_PRE_PHASE3_DISQUALIFIER)
                for s in active_signals if s in BEARISH_EVENT_WEIGHTS
            )
        elif direction == "sell":
            adjusted_score -= sum(
                _cleanup_weight(s, BULLISH_ONLY_WEIGHTS, regime_override)
                for s in active_signals if s in BULLISH_ONLY_WEIGHTS
            )

        # FII strong buying in downtrend = institutional dip-buying; ease penalty by 1
        fii_easing  = fii_dii.get("fii_buying_strong", False)
        # FII strong selling in uptrend = distribution; ease short penalty by 1
        fii_selling = fii_dii.get("fii_selling_strong", False)
        # Gift Nifty gap: strong gap-up in downtrend = global tailwind → ease by 1 more
        gift_gap_up   = gift_nifty.get("gap_up_strong", False)
        gift_gap_down = gift_nifty.get("gap_down_strong", False)
        if nifty_trend == "downtrend" and direction == "buy":
            has_breakout    = "actual_52w_breakout"  in active_signals
            has_6m_momentum   = (tech.get("momentum_6m") or 0) > 15.0
            # A stock at 52w high with 6m momentum WHILE Nifty is down/flat = genuine RS.
            # These are exactly the picks we want — don't penalise them at all.
            # Penalise only stocks riding index beta with no independent RS evidence.
            if has_breakout and has_6m_momentum:
                headwind_penalty = 0      # confirmed RS leader: passes on its own merit
            elif has_breakout:
                headwind_penalty = 3      # 52w breakout but no 6m confirmation
            else:
                headwind_penalty = 12     # no RS: needs raw score 17+ to pass (rare)
            if fii_easing:
                headwind_penalty = max(0, headwind_penalty - 1)
            if gift_gap_up:
                headwind_penalty = max(0, headwind_penalty - 1)
            adjusted_score -= headwind_penalty
        elif nifty_trend == "uptrend" and direction == "sell":
            has_breakdown   = "distribution_signal"  in active_signals
            lacks_6m_momentum = (tech.get("momentum_6m") or 0) <= 15.0
            # Stock breaking down AND losing 6m momentum = confirmed weakness
            if has_breakdown and lacks_6m_momentum:
                headwind_penalty = 1
            elif has_breakdown:
                headwind_penalty = 2
            else:
                headwind_penalty = 4
            if fii_selling:
                headwind_penalty = max(0, headwind_penalty - 1)
            if gift_gap_down:
                headwind_penalty = max(0, headwind_penalty - 1)
            adjusted_score -= headwind_penalty

        # F&O expiry proximity: artificial OI unwinding suppresses SELL reliability
        expiry_suppressed = False
        if direction == "sell" and _is_expiry_proximity(date.today()):
            adjusted_score -= 1
            expiry_suppressed = True

        vtags = _volatility_tags(beta, atr_pct, vol_ratio)
        signals = _label_signals(active_signals)
        risk = _risk(c, tech)

        if direction == "sell":
            phase_b_threshold = SELL_ENTRY_THRESHOLD_DOWNTREND if nifty_trend == "downtrend" else SELL_ENTRY_THRESHOLD_ELSE
        else:
            phase_b_threshold = 5 if nifty_trend == "downtrend" else 4

        # F&O executability gate: Indian cash-market shorts are intraday-only, so a
        # multi-day sell needs stock futures, which only exist for F&O-listed names.
        # Fail-closed — an empty fo_set (fetch failure) routes every sell to watch.
        fo_gated = direction == "sell" and phase in _SELL_PHASES and ticker not in fo_set
        # Short pipeline built but not live — see SHORT_PIPELINE_LIVE comment above.
        short_disabled = direction == "sell" and phase in _SELL_PHASES and not SHORT_PIPELINE_LIVE

        if direction == "watch" or adjusted_score < phase_b_threshold or fo_gated or short_disabled:
            watch_reason = (
                "Short pipeline not yet validated — informational only (no measured edge, see Phase 5)"
                if short_disabled else
                "Short signal — not F&O-tradeable" if fo_gated else
                _watch_reason(direction, adjusted_score, score, nifty_trend, headwind_penalty)
            )
            phase_b_list.append({
                "ticker": ticker,
                "phase": phase,
                "score": score,
                "rsi": tech.get("rsi", "N/A"),
                "today_close": c.get("today_close"),
                "alert_trigger": signals[:3],
                "watch_reason": watch_reason,
            })
            continue

        levels = {
            "entry_zone": tech.get("entry_zone", "N/A"),
            "stop_loss":  tech.get("stop_loss", "N/A"),
            "target_1":   tech.get("target_1", "N/A"),
            "target_2":   tech.get("target_2", "N/A"),
            "risk_reward": tech.get("risk_reward", "—"),
        }

        # Pivot points (daily/weekly/monthly)
        pivots = {k: tech[k] for k in tech if k.endswith(("_pivot", "_r1", "_r2", "_s1", "_s2"))}

        base = {
            "ticker": ticker,
            "score": adjusted_score,
            "today_close": c.get("today_close"),
            "delivery_pct": c.get("delivery_pct"),
            "delivery_spike_pp": c.get("delivery_spike_pp"),
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
            "atr_pct": atr_pct,
            "momentum_6m": tech.get("momentum_6m"),
            "rs_quality":  tech.get("rs_quality"),
            "expiry_suppressed": expiry_suppressed,
            "instrument": "stock_future" if direction == "sell" else "cash_equity",
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

    # BIG MOVER flag: ATR-only informational tag, up to BIG_MOVER_MAX_PER_DAY/day,
    # in score order. See BIG_MOVER_GATE comment — no target/exit override.
    for e in combined:
        e["big_mover"] = False
    flagged = 0
    for e in combined:
        if flagged >= BIG_MOVER_MAX_PER_DAY:
            break
        if _is_big_mover(e.get("atr_pct")):
            e["big_mover"] = True
            flagged += 1

    buy_list  = [e for e in combined if e.get("wyckoff_phase") in _BUY_PHASES]
    sell_list = [e for e in combined if e.get("wyckoff_phase") in _SELL_PHASES]
    # Big movers surface first within each list; score order otherwise.
    buy_list.sort(key=lambda e: (not e["big_mover"], -e["score"]))
    sell_list.sort(key=lambda e: (not e["big_mover"], -e["score"]))

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

    fii_dii    = market_context.get("fii_dii", {}) if market_context else {}
    gift_nifty = market_context.get("gift_nifty", {}) if market_context else {}

    return {
        "scan_date": scan_date.isoformat(),
        "scan_time": scan_time,
        "nifty_context": nifty_trend,
        "total_screened": total_scanned,
        "buy_watchlist": buy_list,
        "sell_watchlist": sell_list,
        "phase_b_watchlist": phase_b_list,
        "fii_dii": fii_dii,
        "gift_nifty": gift_nifty,
        "data_quality_warnings": [],
    }
