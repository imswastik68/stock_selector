"""
Parses and validates Claude's JSON response into the final watchlist schema.
"""

from __future__ import annotations

import json
import re


class ParseError(Exception):
    pass


_BUY_REQUIRED = {"ticker", "score", "wyckoff_phase", "wyckoff_confidence", "entry_zone", "stop_loss", "risk_reward", "risk"}
_SELL_REQUIRED = {"ticker", "score", "wyckoff_phase", "wyckoff_confidence", "stop_loss", "risk_reward", "risk"}
_PHASE_B_REQUIRED = {"ticker", "phase", "alert_trigger", "estimated_days_to_phase_c"}


def _strip_thinking(raw: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (QwQ, R1, etc.)."""
    return re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()


def _extract_json(raw: str) -> str:
    raw = _strip_thinking(raw).strip()
    # Try direct parse first
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass
    # Look for ```json ... ``` fences
    fence_match = re.search(r"```json\s*([\s\S]+?)\s*```", raw)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    # Find outermost {} span
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    raise ParseError(f"No valid JSON found in response (first 200 chars): {raw[:200]}")


def _rr_float(rr_str: str) -> float:
    """Parse '1:2.5' → 2.5. Returns 0.0 on failure."""
    try:
        parts = str(rr_str).split(":")
        if len(parts) == 2:
            return float(parts[1])
    except (ValueError, IndexError):
        pass
    return 0.0


def _validate_buy_entry(entry: dict) -> dict | None:
    missing = _BUY_REQUIRED - entry.keys()
    for field in missing:
        if field == "score":
            entry[field] = 0
        elif field == "risk":
            entry[field] = "HIGH"
        else:
            entry[field] = "unknown"

    # Enforce minimum score and R:R
    if entry.get("score", 0) < 6:
        return None
    if _rr_float(entry.get("risk_reward", "1:1")) < 2.0:
        return None

    # Fill optional fields with safe defaults
    entry.setdefault("volatility_tags", [])
    entry.setdefault("smc_structure", "none")
    entry.setdefault("vsa_signal", "none")
    entry.setdefault("top_signals", [])
    entry.setdefault("expected_move_pct", 0)
    entry.setdefault("timeframe", "3-5d")
    entry.setdefault("target_1", "N/A")
    entry.setdefault("target_2", "N/A")
    entry.setdefault("invalidation", "N/A")
    entry.setdefault("catalyst", "none")

    # LOW-confidence phase → penalise (already done by Claude per constraints,
    # but re-enforce here as a guard)
    if entry.get("wyckoff_confidence") == "LOW" and entry.get("score", 0) < 4:
        return None

    return entry


def _validate_sell_entry(entry: dict) -> dict | None:
    missing = _SELL_REQUIRED - entry.keys()
    for field in missing:
        if field == "score":
            entry[field] = 0
        elif field == "risk":
            entry[field] = "HIGH"
        else:
            entry[field] = "unknown"

    if entry.get("score", 0) < 6:
        return None
    if _rr_float(entry.get("risk_reward", "1:1")) < 2.0:
        return None

    entry.setdefault("volatility_tags", [])
    entry.setdefault("smc_structure", "none")
    entry.setdefault("vsa_signal", "none")
    entry.setdefault("top_signals", [])
    entry.setdefault("expected_drop_pct", 0)
    entry.setdefault("timeframe", "3-5d")
    entry.setdefault("short_entry_zone", entry.get("stop_loss", "N/A"))
    entry.setdefault("cover_target_1", "N/A")
    entry.setdefault("cover_target_2", "N/A")
    entry.setdefault("invalidation", "N/A")

    return entry


def _validate_phase_b_entry(entry: dict) -> dict | None:
    missing = _PHASE_B_REQUIRED - entry.keys()
    for field in missing:
        entry[field] = "unknown"
    return entry


def parse_claude_response(raw: str, scan_date: str, total_screened: int) -> dict:
    """
    Extract, validate, and normalise Claude's JSON output.
    Raises ParseError if no valid JSON is recoverable.
    """
    json_str = _extract_json(raw)
    data = json.loads(json_str)

    if not isinstance(data, dict):
        raise ParseError("Top-level JSON is not an object")

    buy_raw = data.get("buy_watchlist", [])
    sell_raw = data.get("sell_watchlist", [])
    phase_b_raw = data.get("phase_b_watchlist", [])
    warnings_raw = data.get("data_quality_warnings", [])

    if not isinstance(buy_raw, list):
        buy_raw = []
    if not isinstance(sell_raw, list):
        sell_raw = []
    if not isinstance(phase_b_raw, list):
        phase_b_raw = []

    buy_list = [e for e in (_validate_buy_entry(e) for e in buy_raw) if e is not None]
    sell_list = [e for e in (_validate_sell_entry(e) for e in sell_raw) if e is not None]
    phase_b_list = [e for e in (_validate_phase_b_entry(e) for e in phase_b_raw) if e is not None]

    # Cap combined buy+sell at 10, sorted by score
    # Re-split by phase only to correct clear LLM mistakes; unrecognised phases are dropped.
    _BUY_PHASES  = {"ACCUMULATION_C", "ACCUMULATION_D", "MARKUP"}
    _SELL_PHASES = {"DISTRIBUTION_C", "DISTRIBUTION_D", "MARKDOWN"}
    all_actionable = sorted(buy_list + sell_list, key=lambda e: e.get("score", 0), reverse=True)[:10]
    buy_list  = [e for e in all_actionable if e.get("wyckoff_phase", "") in _BUY_PHASES]
    sell_list = [e for e in all_actionable if e.get("wyckoff_phase", "") in _SELL_PHASES]

    return {
        "scan_date": scan_date,
        "nifty_context": data.get("nifty_context", "ranging"),
        "total_screened": total_screened,
        "buy_watchlist": buy_list,
        "sell_watchlist": sell_list,
        "phase_b_watchlist": phase_b_list[:5],
        "data_quality_warnings": warnings_raw if isinstance(warnings_raw, list) else [],
    }
