"""
Automated ship/no-ship gate status -- the single source of truth for "has
anything in this system actually earned real money yet," so a human doesn't
have to eyeball outputs/performance.json or outputs/factor_backtest.json
every day and judge for themselves. Two gates:

  scanner_gate()  -- daily TA scanner's live win rate, bucketed by month.
  momentum_gate() -- moved here from scripts/factor_scan.py:_gate_status
                     (src/ must not import from scripts/; factor_scan.py
                     delegates to this module instead).

Neither gate auto-applies anything -- main.py and scripts/factor_scan.py
print/render the status; a human still decides whether to act on it.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.performance import _load_perf

OUTPUTS = Path(__file__).parent.parent / "outputs"

GATE_MIN_WR_PCT    = 45.0
GATE_MIN_DECIDED   = 40
GATE_CONSEC_MONTHS = 3


def scanner_gate(perf: dict | None = None) -> dict:
    """
    Buckets decided picks (t1_hit/t2_hit = win, sl_hit = loss; timeout and
    open excluded, matching src.performance.performance_summary's own
    win/loss semantics) by the scan_date's calendar month.

    cleared = the GATE_CONSEC_MONTHS most recent months (by scan_date month,
    including the current partial month if that's what's most recent) each
    have >= GATE_MIN_DECIDED decided picks AND >= GATE_MIN_WR_PCT win rate.
    Fewer than GATE_CONSEC_MONTHS distinct months of data at all -> not
    cleared (insufficient history, not a special case).

    Returns {"cleared": bool, "months": [{"month","decided","wins","wr_pct","passes"}, ...],
             "overall_wr_pct", "overall_decided", "status_line"}.
    """
    perf = perf if perf is not None else _load_perf()

    by_month: dict[str, dict[str, int]] = {}
    for scan_date, picks in perf.items():
        month = scan_date[:7]  # "YYYY-MM"
        bucket = by_month.setdefault(month, {"wins": 0, "losses": 0})
        for pick in picks.values():
            outcome = pick.get("outcome", "open")
            if outcome in ("t1_hit", "t2_hit"):
                bucket["wins"] += 1
            elif outcome == "sl_hit":
                bucket["losses"] += 1
            # timeout and open are excluded from decided counts

    months_sorted = sorted(by_month.keys())
    month_rows = []
    for month in months_sorted:
        wins, losses = by_month[month]["wins"], by_month[month]["losses"]
        decided = wins + losses
        wr_pct = round(wins / decided * 100, 1) if decided > 0 else None
        passes = decided >= GATE_MIN_DECIDED and wr_pct is not None and wr_pct >= GATE_MIN_WR_PCT
        month_rows.append({
            "month": month, "decided": decided, "wins": wins, "wr_pct": wr_pct, "passes": passes,
        })

    recent = month_rows[-GATE_CONSEC_MONTHS:]
    cleared = len(recent) == GATE_CONSEC_MONTHS and all(m["passes"] for m in recent)

    total_wins = sum(m["wins"] for m in month_rows)
    total_decided = sum(m["decided"] for m in month_rows)
    overall_wr_pct = round(total_wins / total_decided * 100, 1) if total_decided > 0 else None

    if cleared:
        status_line = (
            f"SCANNER GATE: CLEARED (win rate {overall_wr_pct}% over {total_decided} decided, "
            f"last {GATE_CONSEC_MONTHS} months each >= {GATE_MIN_WR_PCT}% with >= {GATE_MIN_DECIDED} decided)"
        )
    else:
        status_line = (
            f"SCANNER GATE: NOT CLEARED (win rate {overall_wr_pct}% over {total_decided} decided, "
            f"need >= {GATE_MIN_WR_PCT}% x {GATE_CONSEC_MONTHS} consecutive months "
            f"with >= {GATE_MIN_DECIDED} decided)"
        )

    return {
        "cleared": cleared,
        "months": month_rows,
        "overall_wr_pct": overall_wr_pct,
        "overall_decided": total_decided,
        "status_line": status_line,
    }


def momentum_gate() -> dict:
    """
    True (live) only if mom_12_1 or mom_gated actually cleared the
    multi-split ship gate in the most recent scripts/factor_backtest.py
    --validate run. Everything else, including no backtest output existing
    yet, defaults to paper-only -- the safe default, not the exception.

    Returns {"live": bool, "reason": str, "status_line": str}.
    """
    bt_file = OUTPUTS / "factor_backtest.json"
    if not bt_file.exists():
        live, reason = False, "no factor_backtest.json found — defaulting to PAPER-ONLY"
    else:
        try:
            bt = json.loads(bt_file.read_text())
        except Exception:
            live, reason = False, "factor_backtest.json unreadable — defaulting to PAPER-ONLY"
        else:
            live, reason = False, (
                "no momentum strategy has passed the multi-split ship gate "
                "— PAPER-ONLY (outputs/factor_backtest.json)"
            )
            for name in ("mom_12_1", "mom_gated"):
                gate = bt.get("strategies", {}).get(name, {}).get("ship_gate_multi_split", {})
                if gate.get("ships"):
                    live, reason = True, f"{name} passed the multi-split ship gate (outputs/factor_backtest.json)"
                    break

    status_line = f"MOMENTUM GATE: {'LIVE' if live else 'PAPER-ONLY'} ({reason})"
    return {"live": live, "reason": reason, "status_line": status_line}


def gates_report() -> str:
    """Two-line combined status for stdout/Telegram."""
    return scanner_gate()["status_line"] + "\n" + momentum_gate()["status_line"]
