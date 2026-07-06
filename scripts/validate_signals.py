"""
Unified signal-validation harness (Move 3) -- one command that answers "what
is the backtested state of every signal the scanner uses, and does the live
weight match the evidence?"

It does NOT re-derive backtests; it consolidates the verdicts two existing
backtests already produce and reconciles them against the live scorer weights:

  outputs/backtest_signal_stats.json  <- scripts/backtest.py     (OHLCV signals)
  outputs/event_backtest.json         <- scripts/backtest_events.py (event signals)
  src/scorer.py weight dicts          <- what the scanner actually scores today

For each signal it prints: source, n, ret_lift, SHIP/NO-SHIP/UNTESTABLE, the
live weight, and an ACTION -- PROMOTE (ships but unweighted), DEMOTE (weighted
but fails), OK, or WAIT (no data yet). That single table is the "is everything
validated" view, so a weight decision never needs a human to cross-reference
three files by hand.

Ship gate (same bar as scripts/backtest_events.py, applied uniformly):
  SHIP = n >= MIN_N_SHIP AND wr_lift_pp > 0 AND ret_lift > 0.
Event signals additionally carry their own 70/30 holdout verdict (already in
event_backtest.json) -- when present, that verdict wins, since it's stricter.

Usage:
  python scripts/validate_signals.py            # consolidate + reconcile (fast, no network)
  python scripts/validate_signals.py --run      # re-run both backtests first, then reconcile
  python scripts/validate_signals.py --json      # machine-readable output to stdout
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import src.scorer as scorer  # noqa: E402

OUTPUTS = ROOT / "outputs"
SIGNAL_STATS = OUTPUTS / "backtest_signal_stats.json"
EVENT_BT     = OUTPUTS / "event_backtest.json"
OUT_FILE     = OUTPUTS / "signal_validation.json"

MIN_N_SHIP = 500

# backtest_signal_stats uses backtest-only column names for a couple of signals;
# map them to the live scorer key so reconciliation lines up.
_BT_TO_LIVE = {"volume_surge": "volume_5x", "distribution": "distribution_signal"}


def _live_weight(signal: str) -> tuple[int | None, str]:
    """Return (weight, table) for a signal across all scorer weight dicts.
    weight None = signal isn't in any live weight table."""
    for name, table in [
        ("SHORT_TERM", scorer.SHORT_TERM_WEIGHTS), ("SWING", scorer.SWING_WEIGHTS),
        ("DISQUALIFIER", scorer.DISQUALIFIER_WEIGHTS), ("BEARISH", scorer.BEARISH_EVENT_WEIGHTS),
    ]:
        if signal in table:
            return table[signal], name
    return None, "-"


def _gate(n, wr_lift, ret_lift) -> str:
    if n is None or n < MIN_N_SHIP:
        return "INSUFFICIENT_SAMPLE"
    if wr_lift is not None and wr_lift > 0 and ret_lift is not None and ret_lift > 0:
        return "SHIP"
    return "NO-SHIP"


def _collect() -> list[dict]:
    rows: list[dict] = []

    # OHLCV signals (backtest_signal_stats.json)
    if SIGNAL_STATS.exists():
        stats = json.loads(SIGNAL_STATS.read_text()).get("signals", {})
        for sig, st in stats.items():
            live_key = _BT_TO_LIVE.get(sig, sig)
            n, wr, ret = st.get("n"), st.get("wr_lift_pp"), st.get("ret_lift")
            rows.append({
                "signal": live_key, "source": "OHLCV", "n": n,
                "wr_lift_pp": wr, "ret_lift": ret, "verdict": _gate(n, wr, ret),
            })

    # Event signals (event_backtest.json) -- carry their own holdout verdict
    if EVENT_BT.exists():
        sigs = json.loads(EVENT_BT.read_text()).get("signals", {})
        for sig, st in sigs.items():
            v = st.get("verdict") or _gate(st.get("n"), st.get("wr_lift_pp"), st.get("ret_lift"))
            rows.append({
                "signal": sig, "source": "event", "n": st.get("n"),
                "wr_lift_pp": st.get("wr_lift_pp"), "ret_lift": st.get("ret_lift"), "verdict": v,
            })

    # reconcile against live weights + decide action -- MUST be weight-sign-aware:
    # a disqualifier (negative weight) is VALIDATED by negative lift (it correctly
    # predicts underperformance), not positive. Treating "negative lift + weighted"
    # as a demote is wrong for the whole disqualifier table.
    DISQ_FLIP_DEADBAND = 0.2  # only flag a disqualifier if lift is meaningfully positive
    for r in rows:
        w, table = _live_weight(r["signal"])
        r["live_weight"], r["weight_table"] = w, table
        r["pending"] = r["signal"] in scorer.PENDING_VALIDATION
        v = r["verdict"]
        n, ret = r["n"], r["ret_lift"]
        n_ok = n is not None and n >= MIN_N_SHIP

        if "_diag" in r["signal"]:
            r["action"] = "OK"  # diagnostic-only variant (see backtest_events.py), never a promotion candidate
        elif v in ("UNTESTABLE",):
            r["action"] = "WAIT (no data)"
        elif v == "INSUFFICIENT_SAMPLE":
            r["action"] = f"WAIT (n<{MIN_N_SHIP})"
        elif w is not None and w > 0:                      # live BUY signal
            r["action"] = "OK" if v == "SHIP" else ("DEMOTE" if n_ok else "WAIT")
        elif w is not None and w < 0:                      # live DISQUALIFIER
            # correct iff it predicts underperformance (negative lift). Only flag
            # if it's clearly HELPING despite being penalized.
            if n_ok and ret is not None and ret > DISQ_FLIP_DEADBAND:
                r["action"] = "DEMOTE"   # penalizing a signal that actually helps
            else:
                r["action"] = "OK"
        else:                                              # unweighted (0 or absent)
            if v == "SHIP":
                r["action"] = "PROMOTE"                    # ships as a buy signal, not scored
            elif n_ok and ret is not None and ret < -DISQ_FLIP_DEADBAND:
                r["action"] = "PROMOTE?(disq)"             # strong negative lift, could be a disqualifier
            else:
                r["action"] = "OK"
    return rows


def main(do_run: bool, as_json: bool) -> None:
    if do_run:
        print("[validate_signals] re-running scripts/backtest.py ...")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "backtest.py")], check=False)
        # --weeks 260 (SOTA Round Phase 3): the event-source APIs (PIT,
        # announcements) confirmed live to serve the full 5-year window --
        # don't silently fall back to backtest_events.py's own 156w default,
        # which would understate every event-signal sample size on every
        # future automated revalidation.
        print("[validate_signals] re-running scripts/backtest_events.py --weeks 260 ...")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "backtest_events.py"),
                         "--weeks", "260"], check=False)

    rows = _collect()
    rows.sort(key=lambda r: (r["action"] != "PROMOTE", r["action"] != "DEMOTE",
                             -(r["ret_lift"] or -99)))

    OUT_FILE.write_text(json.dumps({"signals": rows}, indent=2, default=str))

    if as_json:
        print(json.dumps({"signals": rows}, indent=2, default=str))
        return

    print(f"\n{'='*100}")
    print("  SIGNAL VALIDATION — every signal's backtested verdict vs its live weight")
    print(f"{'='*100}")
    print(f"  {'signal':<26}{'source':<8}{'n':>8}{'ret_lift':>10}{'verdict':>20}{'weight':>8}   action")
    print(f"  {'-'*96}")
    for r in rows:
        n = r["n"] if r["n"] is not None else "-"
        ret = f"{r['ret_lift']:+.3f}" if r["ret_lift"] is not None else "-"
        w = r["live_weight"] if r["live_weight"] is not None else "-"
        flag = "🔴" if r["action"] in ("PROMOTE", "DEMOTE") else "  "
        print(f"{flag}{r['signal']:<26}{r['source']:<8}{str(n):>8}{ret:>10}{r['verdict']:>20}{str(w):>8}   {r['action']}")

    promotes = [r for r in rows if r["action"] == "PROMOTE"]
    demotes = [r for r in rows if r["action"] == "DEMOTE"]
    print(f"\n  {'-'*96}")
    if promotes:
        print("  PROMOTE (ships in backtest, not yet weighted live):")
        for r in promotes:
            sw = max(0, min(5, round(3 * (r["ret_lift"] or 0))))
            print(f'    "{r["signal"]}": {sw},   # SHIP ret_lift={r["ret_lift"]} n={r["n"]}')
    if demotes:
        print("  DEMOTE (weighted live, fails backtest):")
        for r in demotes:
            print(f'    "{r["signal"]}": 0,   # NO-SHIP ret_lift={r["ret_lift"]} n={r["n"]} (was {r["live_weight"]})')
    if not promotes and not demotes:
        print("  ✓ live weights are consistent with every available backtest verdict.")
    print(f"\n[validate_signals] saved -> {OUT_FILE.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="Re-run both backtests before reconciling")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON to stdout")
    args = ap.parse_args()
    main(args.run, args.json)
