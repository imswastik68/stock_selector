"""
Holdout-alpha backtest -- the FAST proxy for the live-proof verdict.

The live gate (src.gates.live_alpha_gate) proves an edge only as real forward
days accumulate: it can't return anything but INSUFFICIENT until v2-stamped
picks have 10+ trading days of forward history AND cross n>=30 over 3+ months.
That's months of waiting. This script answers the same question -- "does the
shipped signal clear the live bar?" -- TODAY, by reconstructing the signal's
firings over a historical holdout window and running them through the EXACT
same gate, in the EXACT same currency (cost-netted abnormal_10d vs NIFTY,
one-sample t-test).

It reuses live code, never copies it, so it can't silently drift:
  - src.data.reversal._fires / _turnover_ok  -> the pre-declared firing rule
    (RET_3D_THRESHOLD, RSI2_THRESHOLD are the single source of truth there).
  - src.performance._index_fwd_return         -> the fwd-window convention.
  - src.costs.round_trip_cost_pct             -> the same cost haircut.
  - src.gates.live_alpha_gate                 -> the same PROVEN/NO-EDGE/
    INSUFFICIENT verdict, fed a synthetic performance.json-shaped dict.

SCOPE: reversal_oversold_v2 only -- the headline survivorship suspect (+2.102
backtest lift, the strongest and most-inflated in the codebase) and the most
frequent signal, so it reaches n fastest. It needs only per-ticker OHLCV (no
bhavcopy/announcement archive), so it reconstructs cheaply and correctly. The
framework (synthetic perf -> live_alpha_gate) extends to other signals later.

HONEST CAVEATS (printed in the output, not buried here):
  1. This is IN-SAMPLE-ADJACENT. reversal_oversold_v2 was selected/backtested
     over overlapping history, so a PROVEN here is confirmation the edge holds
     in the live metric -- necessary, NOT sufficient. A truly-fresh verdict
     still needs live months, OR a window you reserved and never analyzed
     (pass --from/--to to point at one).
  2. Entry is the T+1 CLOSE (exit-agnostic, isolates the signal), NOT the live
     limit-zone fill. So abnormal_10d here is independent of, and slightly
     different from, what evaluate_live_alpha records live. Same spirit, same
     benchmark math; different entry timing.
  3. Fixed cost %; real per-name slippage on falling-knife fills is worse --
     hence the 2x-cost stress verdict, reported alongside (same as live).

Usage:
  python scripts/holdout_alpha.py                        # last ~15 months, full NIFTY500
  python scripts/holdout_alpha.py --from 2024-01-01 --to 2025-01-01
  python scripts/holdout_alpha.py --sample 100           # fast sanity check, first 100 tickers
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
    import yfinance as yf
except ImportError:
    print("holdout_alpha requires pandas + yfinance"); sys.exit(1)

from src.data import reversal
from src.performance import _index_fwd_return
from src.costs import round_trip_cost_pct
from src.gates import (
    live_alpha_gate, _format_alpha_line,
    LIVE_ALPHA_MIN_N_PER_SIGNAL, LIVE_ALPHA_MIN_N_AGGREGATE,
)

_NIFTY_TICKER = "^NSEI"
_SIGNAL = "reversal_oversold_v2"
_OUT = ROOT / "outputs" / "holdout_alpha.json"
_LOOKBACK_BARS = 60   # bars before a date needed for _fires (RSI2/3d-ret) + _turnover_ok(tail 30)
_FWD_BARS = 10        # forward horizon for the primary abnormal metric


def reconstruct_events(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    nifty_df: pd.DataFrame,
    from_date: str,
    to_date: str,
) -> dict:
    """Pure, network-free core (so it's unit-testable). For every (ticker, T)
    in [from_date, to_date] where reversal_oversold_v2 fires point-in-time,
    emit a performance.json-shaped synthetic pick carrying cost-netted
    abnormal_10d (+ 2x-cost stress), ready for src.gates.live_alpha_gate.

    Entry = the first bar strictly after the signal date T (its close). fwd and
    NIFTY windows are aligned on that same entry date via _index_fwd_return, so
    the abnormal is a clean stock-minus-benchmark difference over identical
    bars. Returns {T_iso: {ticker: pick_dict}}."""
    cost = round_trip_cost_pct("buy")
    lo, hi = pd.Timestamp(from_date), pd.Timestamp(to_date)
    perf: dict[str, dict] = {}

    for ticker, df in ohlcv_by_ticker.items():
        if df is None or df.empty:
            continue
        df = df.dropna(how="all")
        # candidate signal dates: bars inside the window (need lookback before, fwd after)
        for T in df.index:
            if T < lo or T > hi:
                continue
            slice_ = df[df.index <= T].tail(_LOOKBACK_BARS)
            close = slice_["Close"].dropna()
            fires, _ret3d, _rsi2 = reversal._fires(close)
            if not fires or not reversal._turnover_ok(slice_):
                continue

            future = df[df.index > T]
            if len(future) < _FWD_BARS:
                continue  # not enough forward bars to measure fwd_10d
            entry_date = future.index[0]  # T+1

            g10 = _index_fwd_return(df, entry_date, 10)
            if g10 is None:
                continue
            g5 = _index_fwd_return(df, entry_date, 5)
            g20 = _index_fwd_return(df, entry_date, 20)
            nifty_fwd_10d = (
                _index_fwd_return(nifty_df, entry_date, 10)
                if nifty_df is not None and not nifty_df.empty else None
            )

            fwd_10d = round(g10 - cost, 2)
            abnormal_10d = round(fwd_10d - nifty_fwd_10d, 2) if nifty_fwd_10d is not None else None
            stress_10d = round(g10 - 2 * cost, 2)
            abnormal_10d_stress = (
                round(stress_10d - nifty_fwd_10d, 2) if nifty_fwd_10d is not None else None
            )

            key = T.date().isoformat()
            perf.setdefault(key, {})[ticker] = {
                "eval_method": "next_day_zone_v2",   # so live_alpha_gate accepts it
                "direction": "buy",
                "active_signals": [_SIGNAL],
                "fwd_5d": round(g5 - cost, 2) if g5 is not None else None,
                "fwd_10d": fwd_10d,
                "fwd_20d": round(g20 - cost, 2) if g20 is not None else None,
                "nifty_fwd_10d": nifty_fwd_10d,
                "abnormal_10d": abnormal_10d,
                "abnormal_10d_stress": abnormal_10d_stress,
            }
    return perf


def _download(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    batches = [tickers[i:i + reversal.BATCH_SIZE] for i in range(0, len(tickers), reversal.BATCH_SIZE)]
    for i, batch in enumerate(batches):
        print(f"[holdout] batch {i + 1}/{len(batches)} ({len(batch)} tickers)...")
        try:
            df = yf.download(batch, start=start, end=end, progress=False,
                             auto_adjust=True, group_by="ticker", threads=True)
        except Exception as exc:
            print(f"[holdout] batch {i + 1} failed: {exc}")
            continue
        for t in batch:
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    if t not in df.columns.get_level_values(0):
                        continue
                    out[t] = df.xs(t, axis=1, level=0).dropna(how="all")
                else:
                    out[t] = df.dropna(how="all")
            except Exception:
                continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_to = date.today() - timedelta(days=15)          # fwd_10d headroom
    default_from = default_to - timedelta(days=460)          # ~15 months of signal dates
    ap.add_argument("--from", dest="from_date", default=default_from.isoformat())
    ap.add_argument("--to", dest="to_date", default=default_to.isoformat())
    ap.add_argument("--sample", type=int, default=None, help="limit to first N tickers (fast check)")
    args = ap.parse_args()

    universe = reversal._load_universe()
    if args.sample:
        universe = universe[:args.sample]

    dl_start = (pd.Timestamp(args.from_date) - timedelta(days=120)).date().isoformat()  # lookback
    dl_end = (pd.Timestamp(args.to_date) + timedelta(days=40)).date().isoformat()       # fwd bars
    print(f"[holdout] window {args.from_date} -> {args.to_date}, {len(universe)} tickers")

    ohlcv = _download(universe, dl_start, dl_end)
    nifty = _download([_NIFTY_TICKER], dl_start, dl_end).get(_NIFTY_TICKER, pd.DataFrame())
    if nifty.empty:
        print("[holdout] WARNING: NIFTY benchmark unavailable -- abnormal_10d will be null")

    perf = reconstruct_events(ohlcv, nifty, args.from_date, args.to_date)
    n_events = sum(len(v) for v in perf.values())
    print(f"[holdout] reconstructed {n_events} reversal_oversold_v2 event(s)")

    gate = live_alpha_gate(perf=perf)
    rev = gate["per_signal"].get(_SIGNAL, {"n": 0, "months": 0, "verdict": "INSUFFICIENT"})
    lines = [
        _format_alpha_line(_SIGNAL, rev, LIVE_ALPHA_MIN_N_PER_SIGNAL),
        _format_alpha_line("AGGREGATE (all buys)", gate["aggregate"], LIVE_ALPHA_MIN_N_AGGREGATE),
    ]
    if gate.get("reversal_oversold_v2_stress"):
        lines.append(_format_alpha_line(
            f"{_SIGNAL} (2x-cost stress)", gate["reversal_oversold_v2_stress"],
            LIVE_ALPHA_MIN_N_PER_SIGNAL))

    caveats = [
        "IN-SAMPLE-ADJACENT: signal was selected over overlapping history; PROVEN "
        "here is confirmation, not a substitute for the live verdict.",
        "Entry = T+1 close (exit-agnostic), NOT the live limit-zone fill; independent "
        "of what evaluate_live_alpha records live.",
        "Fixed cost %; the 2x-cost stress row is the honest falling-knife slippage check.",
    ]
    payload = {
        "window": {"from": args.from_date, "to": args.to_date},
        "signal": _SIGNAL,
        "n_events": n_events,
        "per_signal": gate["per_signal"],
        "aggregate": gate["aggregate"],
        "reversal_oversold_v2_stress": gate.get("reversal_oversold_v2_stress"),
        "lines": lines,
        "caveats": caveats,
        "generated": date.today().isoformat(),
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(payload, indent=2, default=str))

    print("\n=== HOLDOUT ALPHA (fast proxy for live proof) ===")
    print("\n".join(lines))
    print("\ncaveats:")
    for c in caveats:
        print(f"  - {c}")
    print(f"\n[holdout] written -> {_OUT}")


if __name__ == "__main__":
    main()
