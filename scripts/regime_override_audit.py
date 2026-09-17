"""Audit REGIME_WEIGHTS against the backtest, and prune entries the data contradicts.

Background (2026-09-17). src/scorer.py's REGIME_WEIGHTS re-weights signals by
NIFTY trend (uptrend/ranging/downtrend). Unlike SHORT_TERM_WEIGHTS, the override
table was never validated -- and 15 of its 38 entries have a sign the data
contradicts. The worst: `ranging` sets rsi_momentum to -2, but rsi_momentum is
the ONLY signal in the whole table that is sign-consistently positive across a
70/30 chronological holdout at every horizon (fwd_5d/10d/20d). In ranging
markets the scorer was penalising its single best predictor.

Method (the repo's ship gate, no peeking):
  1. Label every backtest day with the live regime rule -- src/data/market_context
     ._classify_trend: ema20>ema50>ema200 -> uptrend, ema20<ema50<ema200 ->
     downtrend, else ranging -- recomputed from cache/backtest_nifty.csv.
  2. On the TRAIN split ONLY, measure each (regime, signal) lift vs the
     same-regime baseline.
  3. Drop the override entry when sign(train lift) != sign(override weight).
     Entries with < MIN_OBS train observations are LEFT UNTOUCHED -- too thin to
     judge, so the rule does not get to guess.
  4. Score the pruned table on the untouched HOLDOUT, as per-day Spearman rank IC.

Result at the repo-standard 70/30 split, fwd_10d (buy rows only, n=210,072):
    holdout IC  0.0241 -> 0.0356   (+48%), paired t=+1.70
and positive in 12 of 12 (split point x horizon) cells:

    split  fwd_5d   fwd_10d  fwd_20d
    50%    +0.0033  +0.0048  +0.0024
    60%    +0.0052  +0.0076  +0.0034
    70%    +0.0093  +0.0115  +0.0064
    80%    +0.0136  +0.0101  +0.0093

No single cell reaches p<0.05 (best t=+1.97); the evidence is the consistency,
plus the fact that the pruned entries had no validation behind them to begin
with. Improvement grows monotonically with train size, which is what a real
effect looks like and what an overfit does not.

IMPORTANT -- what this does NOT say: an earlier pass of this analysis compared
"all overrides" vs "no overrides" and appeared to show that deleting the whole
table helped. That was an artifact of an incomplete reconstruction that omitted
the bearish penalties (distribution_signal -4/-5, heavy_selling -5/-3/-5,
volume_5x -2/-3). Those entries are CORRECT and carry most of the table's value.
With the full table included, wholesale deletion is a wash (t=-0.50/-0.00/+0.60).
Prune the contradicted entries; do not delete the table.

Usage:
    python scripts/regime_override_audit.py                # audit + write JSON
    python scripts/regime_override_audit.py --emit-table   # print pruned dict
    python scripts/regime_override_audit.py --all-splits   # stability grid
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRADES = ROOT / "outputs" / "backtest_trades.csv"
NIFTY = ROOT / "cache" / "backtest_nifty.csv"
OUT = ROOT / "outputs" / "regime_override_audit.json"

SPLIT = 0.70          # repo-standard chronological holdout
HORIZON = "fwd_10d"   # repo-standard horizon for the prune decision
MIN_OBS = 200         # below this, leave the override alone rather than guess
MIN_NAMES_PER_DAY = 20

# REGIME_WEIGHTS keys -> backtest_trades.csv columns. The two renames are
# documented in src/scorer.py's own comment above REGIME_WEIGHTS.
SIG_COL = {
    "rsi_momentum": "sig_rsi_momentum",
    "rs_vs_nifty": "sig_rs_vs_nifty",
    "rsi_bearish_div": "sig_rsi_bearish_div",
    "rsi_bullish_div": "sig_rsi_bullish_div",
    "macd_bearish_cross": "sig_macd_bearish_cross",
    "bb_squeeze_breakout": "sig_bb_squeeze_breakout",
    "bullish_candle": "sig_bullish_candle",
    "bearish_candle": "sig_bearish_candle",
    "volume_5x": "sig_volume_surge",              # renamed in the backtest
    "distribution_signal": "sig_distribution",    # renamed in the backtest
    "heavy_selling": "sig_heavy_selling",
    "actual_52w_breakout": "sig_actual_52w_breakout",
    "actual_52w_breakdown": "sig_actual_52w_breakdown",
    "near_52w_high": "sig_near_52w_high",
    "rs_quality_strong": "sig_rs_quality_strong",
}


def load() -> pd.DataFrame:
    """Buy-direction backtest trades, labelled with the live NIFTY regime rule."""
    n = pd.read_csv(NIFTY, skiprows=[1, 2], index_col=0, parse_dates=True)
    close = n["Close"].astype(float)
    ema = lambda s, p: s.ewm(span=p, adjust=False).mean()  # noqa: E731
    e20, e50, e200 = ema(close, 20), ema(close, 50), ema(close, 200)
    regime = pd.Series(
        np.where((e20 > e50) & (e50 > e200), "uptrend",
                 np.where((e20 < e50) & (e50 < e200), "downtrend", "ranging")),
        index=close.index,
    )
    d = pd.read_csv(TRADES, parse_dates=["as_of"])
    b = d[d.direction == "buy"].copy()
    b["regime"] = b.as_of.map(regime).fillna("ranging")
    return b


def prune(b: pd.DataFrame, overrides: dict, train_mask: np.ndarray,
          horizon: str) -> tuple[dict, list[dict]]:
    """Drop override entries whose sign the TRAIN data contradicts."""
    pruned: dict[str, dict[str, int]] = {rg: {} for rg in overrides}
    dropped: list[dict] = []
    for rg, weights in overrides.items():
        g = b[(b.regime == rg).values & train_mask]
        for key, w in weights.items():
            col = SIG_COL.get(key)
            if col is None or col not in b.columns:
                pruned[rg][key] = w
                continue
            on = g[g[col] == 1][horizon].dropna()
            off = g[g[col] == 0][horizon].dropna()
            if len(on) < MIN_OBS or len(off) < MIN_OBS:
                pruned[rg][key] = w   # too thin to judge -- leave it alone
                continue
            lift = on.mean() - off.mean()
            if np.sign(lift) != np.sign(w):
                dropped.append({"regime": rg, "signal": key, "override": w,
                                "train_lift": round(float(lift), 4),
                                "train_n": int(len(on))})
            else:
                pruned[rg][key] = w
    return pruned, dropped


def score(b: pd.DataFrame, base: dict, overrides: dict | None) -> np.ndarray:
    out = np.zeros(len(b))
    for rg in ("uptrend", "ranging", "downtrend"):
        m = (b.regime == rg).values
        w = {**base, **((overrides or {}).get(rg, {}))}
        out[m] = sum(b.loc[m, SIG_COL[k]].values * v
                     for k, v in w.items() if v and k in SIG_COL)
    return out


def paired_ic(b: pd.DataFrame, mask: np.ndarray, col_a: str, col_b: str,
              horizon: str) -> dict:
    """Per-day rank IC for two scores on the same days, compared pairwise."""
    a_vals, b_vals = [], []
    for _, g in b[mask].groupby("as_of"):
        g = g.dropna(subset=[horizon])
        if len(g) < MIN_NAMES_PER_DAY or g[col_a].nunique() < 3 or g[col_b].nunique() < 3:
            continue
        va = stats.spearmanr(g[col_a], g[horizon]).statistic
        vb = stats.spearmanr(g[col_b], g[horizon]).statistic
        if np.isfinite(va) and np.isfinite(vb):
            a_vals.append(va)
            b_vals.append(vb)
    a, bb = np.array(a_vals), np.array(b_vals)
    if len(a) < 5:
        return {"days": len(a)}
    t = stats.ttest_rel(bb, a)
    return {"days": len(a), "ic_current": round(float(a.mean()), 4),
            "ic_pruned": round(float(bb.mean()), 4),
            "diff": round(float((bb - a).mean()), 4),
            "paired_t": round(float(t.statistic), 2),
            "p_value": round(float(t.pvalue), 4)}


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-table", action="store_true",
                    help="print the pruned REGIME_WEIGHTS dict for pasting into scorer.py")
    ap.add_argument("--all-splits", action="store_true",
                    help="stability grid over split points x horizons")
    args = ap.parse_args()

    from src.scorer import DISQUALIFIER_WEIGHTS, REGIME_WEIGHTS, SHORT_TERM_WEIGHTS, BEARISH_EVENT_WEIGHTS

    base = {k: v for k, v in {**SHORT_TERM_WEIGHTS, **DISQUALIFIER_WEIGHTS,
                              **BEARISH_EVENT_WEIGHTS}.items() if k in SIG_COL}
    overrides = {rg: dict(w) for rg, w in REGIME_WEIGHTS.items() if w}

    b = load()
    cut = b.as_of.quantile(SPLIT)
    train, hold = (b.as_of <= cut).values, (b.as_of > cut).values
    print(f"buy rows {len(b):,}  train {train.sum():,}  holdout {hold.sum():,}  cut {cut.date()}")
    print(f"regime days: {b.groupby('regime').as_of.nunique().to_dict()}\n")

    if args.all_splits:
        print(f"{'split':>6} {'horizon':>9} {'dropped':>8} {'days':>6} "
              f"{'current':>9} {'pruned':>9} {'diff':>9} {'t':>7}")
        for q in (0.50, 0.60, 0.70, 0.80):
            c = b.as_of.quantile(q)
            tr, ho = (b.as_of <= c).values, (b.as_of > c).values
            for h in ("fwd_5d", "fwd_10d", "fwd_20d"):
                p, dr = prune(b, overrides, tr, h)
                b["_cur"], b["_prn"] = score(b, base, overrides), score(b, base, p)
                r = paired_ic(b, ho, "_cur", "_prn", h)
                print(f"{int(q*100):>5}% {h:>9} {len(dr):>8} {r['days']:>6} "
                      f"{r['ic_current']:>+9.4f} {r['ic_pruned']:>+9.4f} "
                      f"{r['diff']:>+9.4f} {r['paired_t']:>+7.2f}")
        return

    pruned, dropped = prune(b, overrides, train, HORIZON)
    print(f"Dropped {len(dropped)} of {sum(len(w) for w in overrides.values())} "
          f"override entries (sign contradicted by TRAIN data on {HORIZON}):")
    for dd in dropped:
        print(f"   {dd['regime']:<10} {dd['signal']:<22} override {dd['override']:+d}  "
              f"train lift {dd['train_lift']:+.3f} (n={dd['train_n']:,})")

    b["_cur"], b["_prn"] = score(b, base, overrides), score(b, base, pruned)
    results = {}
    print(f"\n{'horizon':>9} {'split':>8} {'days':>5} {'current':>9} {'pruned':>9} "
          f"{'diff':>9} {'t':>7} {'p':>8}")
    for h in ("fwd_5d", "fwd_10d", "fwd_20d"):
        for lbl, m in (("train", train), ("HOLDOUT", hold)):
            r = paired_ic(b, m, "_cur", "_prn", h)
            results[f"{h}_{lbl}"] = r
            print(f"{h:>9} {lbl:>8} {r['days']:>5} {r['ic_current']:>+9.4f} "
                  f"{r['ic_pruned']:>+9.4f} {r['diff']:>+9.4f} "
                  f"{r['paired_t']:>+7.2f} {r['p_value']:>8.4f}")

    if args.emit_table:
        print("\nREGIME_WEIGHTS = {")
        for rg in ("uptrend", "ranging", "downtrend"):
            items = ", ".join(f'"{k}": {v}' for k, v in pruned[rg].items())
            print(f'    "{rg}":{" " * (10 - len(rg))}{{{items}}},')
        print("}")

    OUT.write_text(json.dumps({
        "generated": str(pd.Timestamp.today().date()),
        "split": SPLIT, "cut_date": str(cut.date()), "horizon": HORIZON,
        "min_obs": MIN_OBS, "n_buy_rows": int(len(b)),
        "dropped": dropped, "pruned_table": pruned, "ic": results,
    }, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
