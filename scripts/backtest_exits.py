"""
Backtest candidate exit policies on the 15k cached trades.

Reuses:
  outputs/backtest_trades.csv  — as_of, ticker, direction, close, atr_pct
  cache/backtest_ohlcv/*.csv   — per-ticker daily OHLCV
  src/technicals.compute_entry_levels — reconstruct sl/t1/t2 from close+atr
  src/trade_sim.simulate_raw   — run each policy

Policies compared:
  static, time_10d, breakeven_1r, partial_t1_be, trail_atr3, partial_t1_trail

Output: outputs/backtest_exits.json + console table

Usage: python scripts/backtest_exits.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("Run: pip install pandas numpy")

from src.technicals import compute_entry_levels
from src.trade_sim import simulate_raw
from src.costs import round_trip_cost_pct

TRADES_CSV  = ROOT / "outputs" / "backtest_trades.csv"
OHLCV_CACHE = ROOT / "cache" / "backtest_ohlcv"
OUT_FILE    = ROOT / "outputs" / "backtest_exits.json"

POLICIES = [
    "static",
    "time_10d",
    "breakeven_1r",
    "partial_t1_be",
    "trail_atr3",
    "partial_t1_trail",
]

EXIT_POLICY_LABELS = {
    "static":           "SL/T1 first touch (baseline)",
    "time_10d":         "static + force-exit day 10",
    "breakeven_1r":     "breakeven SL after +1R",
    "partial_t1_be":    "50% @ T1, rest → breakeven SL → T2",
    "trail_atr3":       "chandelier trail 3×ATR (no T1)",
    "partial_t1_trail": "50% @ T1, trail rest 3×ATR",
}


def _parse_price(s: str) -> float:
    return float(str(s).replace("₹", "").replace(",", "").strip())


def _load_ticker_df(ticker: str) -> pd.DataFrame | None:
    # cache files use underscores: RELIANCE_NS.csv (backtest.py replaces . with _)
    fname = ticker.replace(".", "_")
    path = OHLCV_CACHE / f"{fname}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        # Normalise index to tz-naive
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return None


def run() -> None:
    if not TRADES_CSV.exists():
        sys.exit(f"Not found: {TRADES_CSV}\nRun: python scripts/backtest.py first")

    df = pd.read_csv(TRADES_CSV, parse_dates=["as_of"])
    df = df[df["triggered"] == True].copy()
    df = df[df["outcome"].isin(["t1_hit", "sl_hit"])].copy()
    total = len(df)
    print(f"[exit_bt] {total} closed triggered trades to re-simulate across {len(POLICIES)} policies")

    # 70/30 time split
    df = df.sort_values("as_of").reset_index(drop=True)
    split_idx  = int(len(df) * 0.7)
    split_date = df.iloc[split_idx]["as_of"]
    print(f"[exit_bt] train: {df.iloc[0]['as_of'].date()} → {split_date.date()}  "
          f"({split_idx} trades)")
    print(f"[exit_bt] test:  {split_date.date()} → {df.iloc[-1]['as_of'].date()}  "
          f"({total - split_idx} trades)")

    # Load ticker DFs once
    tickers = df["ticker"].unique()
    print(f"[exit_bt] loading {len(tickers)} tickers from cache...")
    ticker_dfs: dict[str, pd.DataFrame] = {}
    for t in tickers:
        td = _load_ticker_df(t)
        if td is not None:
            ticker_dfs[t] = td
    print(f"[exit_bt] loaded {len(ticker_dfs)}/{len(tickers)}")

    # Results accumulator: policy → {full, oos} → list of return_pct
    results: dict[str, dict[str, list[float]]] = {
        p: {"full": [], "oos": []} for p in POLICIES
    }

    skipped = 0
    for _, row in df.iterrows():
        ticker    = row["ticker"]
        direction = row["direction"]
        close_val = float(row["close"])
        atr_pct   = float(row["atr_pct"])
        as_of     = pd.Timestamp(row["as_of"])
        is_oos    = as_of >= split_date

        if ticker not in ticker_dfs:
            skipped += 1
            continue

        atr = close_val * atr_pct / 100.0
        levels = compute_entry_levels(close_val, atr, direction)
        if not levels:
            skipped += 1
            continue

        try:
            ez = levels["entry_zone"]   # "₹123.45-₹127.89"
            parts = ez.replace("₹", "").split("-")
            entry_lo  = float(parts[0])
            entry_hi  = float(parts[1])
            entry_mid = (entry_lo + entry_hi) / 2
            sl        = _parse_price(levels["stop_loss"])
            t1        = _parse_price(levels["target_1"])
            t2        = _parse_price(levels["target_2"])
        except Exception:
            skipped += 1
            continue

        td = ticker_dfs[ticker]

        for policy in POLICIES:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sim = simulate_raw(
                    entry_lo, entry_hi, entry_mid, sl, t1, direction, as_of, td,
                    t2=t2, atr=atr, exit_policy=policy,
                    cost_pct=round_trip_cost_pct(direction),
                )
            # Include timeout and open (for policies with time-stop / trailing)
            if sim.get("triggered"):
                ret = float(sim["return_pct"])
                results[policy]["full"].append(ret)
                if is_oos:
                    results[policy]["oos"].append(ret)

    if skipped:
        print(f"[exit_bt] skipped {skipped} rows (no cache or bad levels)")

    # ── statistics ─────────────────────────────────────────────────────────────
    def _stats(rets: list[float]) -> dict:
        if not rets:
            return {"n": 0, "expectancy": 0.0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
        arr = np.array(rets)
        wins  = arr[arr > 0]
        losses= arr[arr <= 0]
        return {
            "n":          len(arr),
            "expectancy": round(float(arr.mean()), 3),
            "win_rate":   round(float((arr > 0).mean() * 100), 1),
            "avg_win":    round(float(wins.mean())   if len(wins)   > 0 else 0.0, 3),
            "avg_loss":   round(float(losses.mean()) if len(losses) > 0 else 0.0, 3),
        }

    output: dict = {}
    print(f"\n{'='*80}")
    print(f"  EXIT POLICY COMPARISON — {total} trades | OOS from {split_date.date()}")
    print(f"  {'Policy':<22}  {'N':>5}  {'Expct':>7}  {'WR%':>5}  "
          f"{'OOS N':>6}  {'OOS Exp':>8}  {'OOS WR%':>8}")
    print(f"{'='*80}")

    for policy in POLICIES:
        fs  = _stats(results[policy]["full"])
        oos = _stats(results[policy]["oos"])
        output[policy] = {"full": fs, "oos": oos, "label": EXIT_POLICY_LABELS[policy]}
        print(
            f"  {policy:<22}  {fs['n']:>5}  {fs['expectancy']:>+7.3f}  {fs['win_rate']:>4.1f}%  "
            f"{oos['n']:>6}  {oos['expectancy']:>+8.3f}  {oos['win_rate']:>7.1f}%"
        )

    # Winner: best OOS expectancy (non-baseline)
    non_static = {p: output[p]["oos"]["expectancy"] for p in POLICIES if p != "static"}
    winner = max(non_static, key=non_static.get)
    static_oos = output["static"]["oos"]["expectancy"]
    winner_oos = non_static[winner]

    print(f"\n  Baseline (static) OOS expectancy : {static_oos:+.3f}%")
    print(f"  Winner             ({winner:<20}): {winner_oos:+.3f}%  "
          f"({'BEATS' if winner_oos > static_oos else 'LOSES TO'} baseline)")
    print(f"  Label: {EXIT_POLICY_LABELS[winner]}")
    print(f"{'='*80}")
    print(f"\n  → Update WINNER_POLICY in src/trade_sim.py to '{winner}'")

    output["_meta"] = {
        "total_trades": total,
        "split_date":   split_date.date().isoformat(),
        "winner":       winner,
        "winner_beats_static": winner_oos > static_oos,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"  Results → {OUT_FILE.name}")


if __name__ == "__main__":
    run()
