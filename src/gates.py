"""
Automated ship/no-ship gate status -- the single source of truth for "has
anything in this system actually earned real money yet," so a human doesn't
have to eyeball outputs/performance.json or outputs/factor_backtest.json
every day and judge for themselves. Three gates:

  scanner_gate()    -- daily TA scanner's live win rate, bucketed by month.
                       Simple and intuitive, but a 45% win rate proves
                       nothing without a benchmark -- see live_alpha_gate.
  momentum_gate()   -- moved here from scripts/factor_scan.py:_gate_status
                       (src/ must not import from scripts/; factor_scan.py
                       delegates to this module instead).
  live_alpha_gate() -- Live-Proof Round (Phase 4): per-signal AND aggregate
                       forward alpha vs NIFTY, with a pre-declared
                       significance test. THIS is what actually proves an
                       EDGE rather than a possibly-lucky win rate.

Neither win-rate gate auto-applies anything -- main.py and scripts/factor_scan.py
print/render the status; a human still decides whether to act on it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from src.performance import _load_perf

OUTPUTS = Path(__file__).parent.parent / "outputs"

GATE_MIN_WR_PCT    = 45.0
GATE_MIN_DECIDED   = 40
GATE_CONSEC_MONTHS = 3

# Live-Proof Round Phase 4 -- FROZEN NOW, before data accumulates. Never
# loosened after seeing results (that would be exactly the p-hacking this
# exists to prevent). Metric = mean abnormal_10d (fwd_10d - nifty_fwd_10d,
# already cost-netted -- see src/performance.py:evaluate_live_alpha).
LIVE_ALPHA_MIN_N_PER_SIGNAL = 30     # decided-for-alpha picks carrying that signal
LIVE_ALPHA_MIN_N_AGGREGATE  = 100
LIVE_ALPHA_MIN_MONTHS       = 3      # distinct calendar months spanned
LIVE_ALPHA_SIGNIFICANCE     = 0.05   # one-sample t-test p-value, H0: mean abnormal_10d = 0


def scanner_gate(perf: dict | None = None) -> dict:
    """
    Buckets decided picks (t1_hit/t2_hit = win, sl_hit = loss; timeout and
    open excluded, matching src.performance.performance_summary's own
    win/loss semantics) by the scan_date's calendar month -- V2-METHODOLOGY
    ONLY (pick.get("eval_method") == "next_day_zone_v2"). Picks recorded
    before that stamp existed were evaluated with a scan-day point-entry bug
    (same-day stop-outs on a bar that had already happened by scan time --
    see src/performance.py's evaluate_prior_picks as_of comment) and must not
    poison the verdict that answers "has this actually made money." Their
    counts are still surfaced (legacy_decided/legacy_wr_pct) as context, never
    as gate input.

    cleared = the GATE_CONSEC_MONTHS most recent months (by scan_date month,
    including the current partial month if that's what's most recent) each
    have >= GATE_MIN_DECIDED v2-decided picks AND >= GATE_MIN_WR_PCT v2 win
    rate. Fewer than GATE_CONSEC_MONTHS distinct months of data at all -> not
    cleared (insufficient history, not a special case).

    Returns {"cleared": bool, "months": [{"month","decided","wins","wr_pct","passes"}, ...],
             "overall_wr_pct", "overall_decided" (both v2-only),
             "legacy_decided", "legacy_wr_pct", "status_line"}.
    """
    perf = perf if perf is not None else _load_perf()

    by_month: dict[str, dict[str, int]] = {}
    legacy_wins = 0
    legacy_losses = 0
    for scan_date, picks in perf.items():
        month = scan_date[:7]  # "YYYY-MM"
        bucket = by_month.setdefault(month, {"wins": 0, "losses": 0})
        for pick in picks.values():
            outcome = pick.get("outcome", "open")
            is_v2 = pick.get("eval_method") == "next_day_zone_v2"
            if outcome in ("t1_hit", "t2_hit"):
                if is_v2:
                    bucket["wins"] += 1
                else:
                    legacy_wins += 1
            elif outcome == "sl_hit":
                if is_v2:
                    bucket["losses"] += 1
                else:
                    legacy_losses += 1
            # timeout and open are excluded from decided counts (v2 or legacy)

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

    legacy_decided = legacy_wins + legacy_losses
    legacy_wr_pct = round(legacy_wins / legacy_decided * 100, 1) if legacy_decided > 0 else None

    if cleared:
        status_line = (
            f"SCANNER GATE: CLEARED (v2 win rate {overall_wr_pct}% over {total_decided} decided, "
            f"last {GATE_CONSEC_MONTHS} months each >= {GATE_MIN_WR_PCT}% with >= {GATE_MIN_DECIDED} decided)"
        )
    else:
        v2_part = (f"v2: {overall_wr_pct}% over {total_decided} decided" if total_decided > 0
                   else "v2: no decided picks yet")
        legacy_part = (
            f"; legacy {legacy_wr_pct}% over {legacy_decided}, excluded — pre-fix methodology"
            if legacy_decided > 0 else ""
        )
        status_line = (
            f"SCANNER GATE: NOT CLEARED ({v2_part}{legacy_part}, "
            f"need >= {GATE_MIN_WR_PCT}% x {GATE_CONSEC_MONTHS} consecutive months "
            f"with >= {GATE_MIN_DECIDED} decided)"
        )

    return {
        "cleared": cleared,
        "months": month_rows,
        "overall_wr_pct": overall_wr_pct,
        "overall_decided": total_decided,
        "legacy_decided": legacy_decided,
        "legacy_wr_pct": legacy_wr_pct,
        "status_line": status_line,
    }


def _two_tailed_p_value(t_stat: float, n: int) -> float:
    """Two-tailed p-value for a one-sample t-test, via the normal
    approximation (math.erf) -- no scipy dependency (matches this codebase's
    existing convention of computing t-stats inline, e.g.
    scripts/backtest_events.py:_rebalance_verdict). At n >=
    LIVE_ALPHA_MIN_N_PER_SIGNAL (30) the t- and normal distributions are close
    enough for a binary ship/no-ship decision. Known direction of the
    approximation error: the t-distribution has fatter tails at low df, so
    this normal approximation is slightly ANTI-conservative (a shade easier
    to clear than an exact t-test) -- if it biases the gate at all, it biases
    toward false "PROVEN," never toward hiding a real edge. Worth swapping
    for an exact incomplete-beta t-CDF (or adding scipy) if this ever becomes
    a real go/no-go for capital, not just an internal instrument."""
    z = abs(t_stat)
    return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))


def _alpha_verdict(rows: list[tuple[str, float]], min_n: int) -> dict:
    """One-sample t-test (H0: mean=0) on the abnormal-return values in `rows`
    (each a (month, value) pair). Returns {n, months, mean, win_pct, t_stat,
    p_value, verdict}, verdict in {"PROVEN", "NO-EDGE", "INSUFFICIENT"}.
    win_pct = % of rows that beat NIFTY at all (value > 0) -- reported for
    human context only, NOT part of the gate decision (a signal can win most
    of the time with a small mean, or lose most of the time with a few huge
    winners; the t-test on the mean is what actually decides PROVEN/NO-EDGE).

    PROVEN requires ALL of: n >= min_n, mean > 0, p_value < SIGNIFICANCE, AND
    the rows span >= LIVE_ALPHA_MIN_MONTHS distinct calendar months (enough n
    crammed into one wild week is not the same as a proven edge over time)."""
    n = len(rows)
    months = len({m for m, _ in rows})
    win_pct = round(sum(1 for _, v in rows if v > 0) / n * 100, 1) if n else None
    if n < min_n:
        return {
            "n": n, "months": months,
            "mean": round(float(np.mean([v for _, v in rows])), 3) if n else None,
            "win_pct": win_pct, "t_stat": None, "p_value": None, "verdict": "INSUFFICIENT",
        }

    values = np.array([v for _, v in rows])
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    t_stat = mean / (std / math.sqrt(n)) if std > 0 else 0.0
    p_value = _two_tailed_p_value(t_stat, n)

    if months < LIVE_ALPHA_MIN_MONTHS:
        verdict = "INSUFFICIENT"  # enough n, not enough calendar spread yet
    elif mean > 0 and p_value < LIVE_ALPHA_SIGNIFICANCE:
        verdict = "PROVEN"
    else:
        verdict = "NO-EDGE"

    return {
        "n": n, "months": months, "mean": round(mean, 3), "win_pct": win_pct,
        "t_stat": round(t_stat, 2), "p_value": round(p_value, 4), "verdict": verdict,
    }


def live_alpha_gate(perf: dict | None = None) -> dict:
    """
    Live-Proof Round Phase 4 -- the gate that actually proves an EDGE, not
    just a win rate. Buckets every v2-methodology, cost-netted abnormal_10d
    value (src.performance.evaluate_live_alpha) by each signal in the pick's
    active_signals (a pick with 3 signals counts toward all 3 -- this is
    ATTRIBUTION, not a partition of the sample) plus one aggregate-over-all-
    buys bucket. "Decided" here means "abnormal_10d has been computed" (10+
    forward trading bars exist) -- NOT that the SL/T1 exit outcome is
    decided; those are deliberately independent (see evaluate_live_alpha).

    BUY-ONLY: the benchmark math (fwd_10d vs nifty_fwd_10d) assumes a long
    position: a short's alpha vs "did nothing" isn't "beat long-NIFTY," so
    sell-direction picks are excluded here rather than silently mismeasured.
    The short pipeline isn't live today anyway (see src/agent.py's
    SHORT_PIPELINE_LIVE), so this excludes ~nothing in practice.

    Also reports reversal_oversold_v2's 2x-cost STRESS verdict alongside its
    normal one -- the explicit, honest slippage check on the single most
    survivorship-inflated backtest in this codebase (+2.10 lift, see
    src/scorer.py's reversal_oversold_v2 comment).

    Returns {"per_signal": {signal: {n, months, mean, t_stat, p_value, verdict}, ...},
             "aggregate": {...}, "reversal_oversold_v2_stress": {...} | None}.
    """
    perf = perf if perf is not None else _load_perf()

    by_signal: dict[str, list[tuple[str, float]]] = {}
    aggregate: list[tuple[str, float]] = []
    reversal_stress: list[tuple[str, float]] = []

    for scan_date, picks in perf.items():
        month = scan_date[:7]
        for pick in picks.values():
            if pick.get("eval_method") != "next_day_zone_v2":
                continue  # v2-only, same discipline as scanner_gate
            if pick.get("direction") != "buy":
                continue  # see BUY-ONLY note above
            abnormal = pick.get("abnormal_10d")
            if abnormal is None:
                continue  # not yet computed, or entry never filled

            aggregate.append((month, abnormal))
            active_signals = pick.get("active_signals") or []
            for sig in active_signals:
                by_signal.setdefault(sig, []).append((month, abnormal))

            if "reversal_oversold_v2" in active_signals:
                stress = pick.get("abnormal_10d_stress")
                if stress is not None:
                    reversal_stress.append((month, stress))

    per_signal = {
        sig: _alpha_verdict(rows, LIVE_ALPHA_MIN_N_PER_SIGNAL)
        for sig, rows in sorted(by_signal.items())
    }
    agg = _alpha_verdict(aggregate, LIVE_ALPHA_MIN_N_AGGREGATE)
    reversal_stress_verdict = (
        _alpha_verdict(reversal_stress, LIVE_ALPHA_MIN_N_PER_SIGNAL) if reversal_stress else None
    )

    return {
        "per_signal": per_signal,
        "aggregate": agg,
        "reversal_oversold_v2_stress": reversal_stress_verdict,
    }


def _format_alpha_line(name: str, result: dict, min_n: int) -> str:
    if result["verdict"] == "INSUFFICIENT":
        return (f"{name}: INSUFFICIENT (n={result['n']}/{min_n}, "
                f"{result['months']}/{LIVE_ALPHA_MIN_MONTHS} months)")
    icon = "✅" if result["verdict"] == "PROVEN" else "❌"
    sign = "+" if result["mean"] >= 0 else ""
    return (f"{name}: {sign}{result['mean']}% α over n={result['n']} "
            f"(win {result['win_pct']}%), p={result['p_value']} {icon} {result['verdict']}")


def live_proof_report(perf: dict | None = None) -> dict:
    """
    Live-Proof Round Phase 5 -- consolidates live_alpha_gate() into a
    printable report and the outputs/live_proof.json artifact (git-committed,
    Phase 1), so "which signals have actually PROVEN a live edge" never
    requires a human to cross-reference performance.json by hand.

    Reconciliation note (read this before citing either number as "the"
    proof): outputs/performance.json (unconstrained, every pick regardless of
    capital) is what THIS report reads -- it proves the SIGNAL edge exists.
    outputs/portfolio.json (capital-constrained, cost-netted, drawdown-
    tracked) proves TRADEABLE P&L -- a signal can have a real live alpha here
    and still be un-deployable at scale if positions can't be sized without
    moving the price (see reversal_oversold_v2's stress row: this exact
    concern is why it gets one). Neither file alone is "proof" -- report both.

    Returns {**live_alpha_gate()'s dict, "lines": [str, ...]}, and writes the
    same structure to outputs/live_proof.json.
    """
    gate = live_alpha_gate(perf)

    lines = [_format_alpha_line(sig, result, LIVE_ALPHA_MIN_N_PER_SIGNAL)
             for sig, result in gate["per_signal"].items()]
    lines.append(_format_alpha_line("AGGREGATE (all buys)", gate["aggregate"], LIVE_ALPHA_MIN_N_AGGREGATE))
    if gate["reversal_oversold_v2_stress"] is not None:
        lines.append(_format_alpha_line(
            "reversal_oversold_v2 [2x-cost stress]", gate["reversal_oversold_v2_stress"],
            LIVE_ALPHA_MIN_N_PER_SIGNAL,
        ))

    report = {**gate, "lines": lines}
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "live_proof.json").write_text(json.dumps(report, indent=2, default=str))
    return report


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
