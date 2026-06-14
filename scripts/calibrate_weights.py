"""
Derive signal weights from backtest_trades.csv using expectancy lift.

Two methods:
  --method simple     (default) weight = round(k * expectancy_lift), floor 0
  --method logistic   logistic regression P(T1|signals); sklearn required

Walk-forward validation:
  Calibrate on rolling 12-month window, test next 3 months OOS.
  Reports calibrated vs current hand-weights expectancy comparison.

Outputs:
  outputs/calibrated_weights.json   — proposed SHORT_TERM_WEIGHTS
  outputs/walk_forward_results.json — OOS expectancy by window

Usage:
  python scripts/calibrate_weights.py
  python scripts/calibrate_weights.py --method logistic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("Run: pip install pandas numpy")

TRADES_CSV  = ROOT / "outputs" / "backtest_trades.csv"
OUT_WEIGHTS = ROOT / "outputs" / "calibrated_weights.json"
OUT_WF      = ROOT / "outputs" / "walk_forward_results.json"

# Current hand weights (from src/scorer.py SHORT_TERM_WEIGHTS + DISQUALIFIER_WEIGHTS, OHLCV-derivable only)
# Updated to reflect calibrated weights applied 2026-06-13
CURRENT_WEIGHTS: dict[str, int] = {
    "rsi_bearish_div":     3,   # mapped from SHORT_TERM_WEIGHTS (was -2 disqualifier)
    "rsi_momentum":        2,
    "rs_quality_strong":   2,
    "rs_vs_nifty":         1,
    "weekly_trend_aligned":1,
    # disqualifiers (negative)
    "rsi_bullish_div":    -5,   # moved to DISQUALIFIER (was +1)
    "distribution":       -1,   # mapped from distribution_signal (was -5)
    "volume_surge":       -1,   # mapped from volume_5x (was +3)
    "bb_squeeze_breakout":-1,   # moved to DISQUALIFIER (was +2)
    "macd_bearish_cross": -1,
    "bullish_candle":     -1,   # moved to DISQUALIFIER (was +1)
    "bearish_candle":     -1,
    # zero-weight (removed): momentum_6m_strong, macd_bullish_cross, obv_accumulation
}

SIGNAL_COLS = [f"sig_{s}" for s in [
    "rsi_momentum","rs_vs_nifty","rsi_bearish_div","rsi_bullish_div",
    "macd_bearish_cross","bb_squeeze_breakout","bullish_candle","bearish_candle",
    "weekly_trend_aligned","rs_quality_strong","volume_surge","distribution",
]]


def _load_trades() -> pd.DataFrame:
    if not TRADES_CSV.exists():
        sys.exit(f"Not found: {TRADES_CSV}\nRun: python scripts/backtest.py first")
    df = pd.read_csv(TRADES_CSV, parse_dates=["as_of"])
    df = df[df["triggered"] == True].copy()
    df = df[df["outcome"].isin(["t1_hit", "sl_hit"])].copy()
    df["win"] = (df["outcome"] == "t1_hit").astype(int)
    print(f"[calibrate] {len(df)} closed trades loaded from {TRADES_CSV.name}")
    return df


# ── per-signal expectancy lift ────────────────────────────────────────────────

def _signal_expectancy_lift(df: pd.DataFrame) -> dict[str, float]:
    """
    For each signal: expectancy of trades WITH signal minus expectancy of ALL trades.
    Expectancy = avg(return_pct) * win_rate - avg(abs_loss) * (1 - win_rate).
    Simplified: just avg return_pct as expectancy proxy (monotone with full formula).
    """
    baseline_exp = float(df["return_pct"].mean())
    lifts: dict[str, float] = {}
    for col in SIGNAL_COLS:
        if col not in df.columns:
            continue
        sig_name = col.replace("sig_", "")
        with_sig = df[df[col] == True]
        if len(with_sig) < 5:
            continue
        exp_sig = float(with_sig["return_pct"].mean())
        lifts[sig_name] = round(exp_sig - baseline_exp, 4)
    return lifts


def _simple_weights(lifts: dict[str, float], k: float = 3.0) -> dict[str, int]:
    """weight = round(k * lift), clamp to [-5, +5], floor non-positive to 0 for positive signals."""
    weights: dict[str, int] = {}
    for sig, lift in lifts.items():
        raw = k * lift
        w   = int(round(raw))
        w   = max(-5, min(5, w))
        weights[sig] = w
    return weights


def _logistic_weights(df: pd.DataFrame) -> dict[str, float]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        sys.exit("Run: pip install scikit-learn")

    cols = [c for c in SIGNAL_COLS if c in df.columns]
    X = df[cols].fillna(0).astype(float).values
    y = df["win"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(Xs, y)

    weights: dict[str, float] = {}
    for col, coef in zip(cols, lr.coef_[0]):
        weights[col.replace("sig_","")] = round(float(coef), 4)
    return weights


# ── walk-forward validation ───────────────────────────────────────────────────

def _score_with_weights(row: pd.Series, weights: dict[str, int | float]) -> float:
    score = 0.0
    for sig, w in weights.items():
        col = f"sig_{sig}"
        if col in row.index and row[col]:
            score += w
    return score


def _expectancy_from_picks(df: pd.DataFrame, weights: dict) -> float | None:
    if df.empty:
        return None
    df = df.copy()
    df["score"] = df.apply(lambda r: _score_with_weights(r, weights), axis=1)
    picks = df[df["score"] >= 3]
    if picks.empty:
        return None
    return round(float(picks["return_pct"].mean()), 4)


def walk_forward(df: pd.DataFrame, method: str) -> list[dict]:
    df = df.copy()
    df["as_of"] = pd.to_datetime(df["as_of"])
    df = df.sort_values("as_of")

    results: list[dict] = []
    min_date = df["as_of"].min()
    max_date = df["as_of"].max()

    window_months = 12
    oos_months    = 3
    step_months   = 3

    cur = min_date + pd.DateOffset(months=window_months)
    while cur + pd.DateOffset(months=oos_months) <= max_date:
        train_start = cur - pd.DateOffset(months=window_months)
        train_end   = cur
        oos_end     = cur + pd.DateOffset(months=oos_months)

        train = df[(df["as_of"] >= train_start) & (df["as_of"] < train_end)]
        oos   = df[(df["as_of"] >= train_end)   & (df["as_of"] < oos_end)]

        if len(train) < 30 or len(oos) < 5:
            cur += pd.DateOffset(months=step_months)
            continue

        # Calibrated weights
        if method == "logistic":
            cal_w = _logistic_weights(train)
        else:
            lifts = _signal_expectancy_lift(train)
            cal_w = _simple_weights(lifts)

        cal_exp  = _expectancy_from_picks(oos, cal_w)
        hand_exp = _expectancy_from_picks(oos, CURRENT_WEIGHTS)
        base_exp = float(oos["return_pct"].mean())

        results.append({
            "train_start": train_start.date().isoformat(),
            "train_end":   train_end.date().isoformat(),
            "oos_end":     oos_end.date().isoformat(),
            "train_n":     len(train),
            "oos_n":       len(oos),
            "calibrated_expectancy":    cal_exp,
            "hand_weight_expectancy":   hand_exp,
            "baseline_expectancy":      round(base_exp, 4),
            "calibrated_beats_hand":    (cal_exp is not None and hand_exp is not None
                                         and cal_exp > hand_exp),
        })
        cur += pd.DateOffset(months=step_months)

    return results


# ── collinearity report ───────────────────────────────────────────────────────

def run_collinearity() -> None:
    """
    Correlation matrix of signal flags across the backtest trades.
    Flags pairs with |r| > 0.7 (high collinearity → double-counting risk).
    """
    df = _load_trades()
    cols = [c for c in SIGNAL_COLS if c in df.columns]
    if not cols:
        print("[collinearity] no signal columns found in CSV — run backtest.py first")
        return

    sig_df = df[cols].astype(float)
    corr   = sig_df.corr()
    threshold = 0.7

    print(f"\n{'='*60}")
    print(f"  SIGNAL COLLINEARITY REPORT (|r| > {threshold})")
    print(f"  {len(cols)} signals, {len(df)} trades")
    print(f"{'='*60}")
    pairs_found = 0
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = float(corr.loc[a, b]) if (a in corr.index and b in corr.columns) else 0.0
            if abs(r) > threshold:
                sa = a.replace("sig_", "")
                sb = b.replace("sig_", "")
                print(f"  {sa:<28} ↔ {sb:<28}  r={r:+.3f}  ← HIGH")
                pairs_found += 1
    if pairs_found == 0:
        print(f"  ✓ No pairs above {threshold} — no double-counting risk detected")
    print(f"{'='*60}\n")


# ── regime-dependent calibration ──────────────────────────────────────────────

MIN_BUCKET_TRADES = 1000  # below this, fall back to flat weights for that bucket


def _classify_nifty_trend(nifty_df: pd.DataFrame) -> pd.Series:
    """
    Classify each date in nifty_df as 'uptrend' | 'downtrend' | 'ranging'
    based on EMA20 > EMA50 > EMA200 stack.
    Returns a Series indexed by date with string labels.
    """
    close = nifty_df["Close"].squeeze()
    ema20  = close.ewm(span=20,  adjust=False).mean()
    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    labels = pd.Series("ranging", index=close.index)
    labels[ema20 > ema50] = "ranging"           # start neutral
    labels[(ema20 > ema50) & (ema50 > ema200)] = "uptrend"
    labels[(ema20 < ema50) & (ema50 < ema200)] = "downtrend"
    return labels


def run_by_regime() -> dict[str, dict[str, int] | None]:
    """
    Calibrate per signal, per Nifty trend regime (uptrend/ranging/downtrend).
    Returns {regime: weights_dict or None} — None means fewer than MIN_BUCKET_TRADES.
    Prints a table and writes outputs/regime_weights.json.
    Prints the REGIME_WEIGHTS block to paste into scorer.py.
    """
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Run: pip install yfinance")

    print("[by-regime] downloading ^NSEI history (7y) ...")
    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        nifty_df = yf.download("^NSEI", period="7y", auto_adjust=True, progress=False)
    if nifty_df.empty:
        sys.exit("[by-regime] Nifty download failed")

    trend_map = _classify_nifty_trend(nifty_df)
    # Normalise to date (strip time)
    trend_map.index = pd.to_datetime(trend_map.index).normalize()

    df = _load_trades()
    df["as_of"]  = pd.to_datetime(df["as_of"]).dt.normalize()
    df["regime"] = df["as_of"].map(trend_map)
    df["regime"] = df["regime"].fillna("ranging")

    print(f"\n[by-regime] Regime distribution in {len(df)} trades:")
    vc = df["regime"].value_counts()
    for regime, count in vc.items():
        print(f"  {regime:<12} {count:>6} trades ({count/len(df)*100:.1f}%)")

    REGIMES = ["uptrend", "ranging", "downtrend"]
    regime_weights: dict[str, dict[str, int] | None] = {}

    print(f"\n{'='*70}")
    print(f"  REGIME-DEPENDENT CALIBRATION (min {MIN_BUCKET_TRADES} trades/bucket)")
    print(f"{'='*70}")

    for regime in REGIMES:
        bucket = df[df["regime"] == regime]
        n = len(bucket)
        if n < MIN_BUCKET_TRADES:
            print(f"\n  [{regime}] {n} trades < {MIN_BUCKET_TRADES} — SKIP (use flat weights)")
            regime_weights[regime] = None
            continue

        lifts = _signal_expectancy_lift(bucket)
        cal_w = _simple_weights(lifts)
        baseline = float(bucket["return_pct"].mean())

        # 70/30 OOS check within bucket
        bucket_sorted = bucket.sort_values("as_of")
        split = int(len(bucket_sorted) * 0.7)
        train_b = bucket_sorted.iloc[:split]
        oos_b   = bucket_sorted.iloc[split:]

        if len(oos_b) >= 5:
            train_lifts = _signal_expectancy_lift(train_b)
            train_w     = _simple_weights(train_lifts)
            oos_cal_exp  = _expectancy_from_picks(oos_b, train_w)
            oos_flat_exp = _expectancy_from_picks(oos_b, CURRENT_WEIGHTS)
            oos_base     = float(oos_b["return_pct"].mean())
            beats = oos_cal_exp is not None and oos_flat_exp is not None and oos_cal_exp > oos_flat_exp
        else:
            oos_cal_exp = oos_flat_exp = oos_base = None
            beats = False

        print(f"\n  [{regime}]  n={n}  baseline={baseline:+.3f}%  "
              f"OOS cal={oos_cal_exp if oos_cal_exp is not None else 'n/a':>8}  "
              f"OOS flat={oos_flat_exp if oos_flat_exp is not None else 'n/a':>8}  "
              f"{'BEATS flat' if beats else 'LOSES to flat'}")
        print(f"  {'Signal':<28}  {'Lift':>6}  {'Hand':>5}  {'Cal':>5}")
        for sig, lift in sorted(lifts.items(), key=lambda x: -abs(x[1])):
            print(f"  {sig:<28}  {lift:>+6.3f}  {str(CURRENT_WEIGHTS.get(sig,'—')):>5}  "
                  f"{str(cal_w.get(sig,'—')):>5}")

        # Only ship if beats flat OOS; otherwise store None to fall back to flat
        if beats:
            regime_weights[regime] = cal_w
            print(f"  → SHIPPING regime weights for {regime}")
        else:
            regime_weights[regime] = None
            print(f"  → USING flat weights for {regime} (no OOS improvement)")

    print(f"\n{'='*70}")

    # Emit Python block to paste into scorer.py
    print(f"\n  Paste into src/scorer.py:\n")
    print(f"REGIME_WEIGHTS: dict[str, dict[str, int] | None] = {{")
    for regime in REGIMES:
        w = regime_weights[regime]
        if w is None:
            print(f'    "{regime}": None,  # flat weights (no OOS improvement)')
        else:
            print(f'    "{regime}": {w!r},')
    print(f"}}")

    # Save JSON
    out = ROOT / "outputs" / "regime_weights.json"
    out.write_text(json.dumps({r: v for r, v in regime_weights.items()}, indent=2))
    print(f"\n  Results → {out.name}")

    return regime_weights


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["simple", "logistic"], default="simple")
    ap.add_argument("--collinearity", action="store_true",
                    help="Print signal correlation matrix and exit")
    ap.add_argument("--by-regime", action="store_true",
                    help="Calibrate per Nifty trend regime (uptrend/ranging/downtrend)")
    args = ap.parse_args()

    if args.collinearity:
        run_collinearity()
        return

    if args.by_regime:
        run_by_regime()
        return

    df = _load_trades()

    # Full-sample calibration
    lifts = _signal_expectancy_lift(df)

    if args.method == "logistic":
        print("[calibrate] fitting logistic regression ...")
        cal_weights_raw = _logistic_weights(df)
        # Normalise to integer scale (max abs = 3)
        max_abs = max(abs(v) for v in cal_weights_raw.values()) or 1
        cal_weights = {s: int(round(v / max_abs * 3)) for s, v in cal_weights_raw.items()}
    else:
        cal_weights = _simple_weights(lifts)

    # Walk-forward
    print("[calibrate] running walk-forward validation ...")
    wf = walk_forward(df, args.method)

    # Summary
    if wf:
        beats = sum(1 for r in wf if r.get("calibrated_beats_hand"))
        cal_avg  = [r["calibrated_expectancy"]  for r in wf if r["calibrated_expectancy"]  is not None]
        hand_avg = [r["hand_weight_expectancy"] for r in wf if r["hand_weight_expectancy"] is not None]
        print(f"\n{'='*60}")
        print(f"  WALK-FORWARD RESULTS ({len(wf)} windows)")
        print(f"{'='*60}")
        print(f"  Calibrated beats hand weights: {beats}/{len(wf)} windows")
        if cal_avg:
            print(f"  Avg OOS expectancy (calibrated): {sum(cal_avg)/len(cal_avg):+.2f}%")
        if hand_avg:
            print(f"  Avg OOS expectancy (hand):       {sum(hand_avg)/len(hand_avg):+.2f}%")

    # Per-signal table
    print(f"\n  SIGNAL EXPECTANCY LIFT (full sample):")
    print(f"  {'Signal':<28}  {'Lift':>6}  {'Hand w':>6}  {'Cal w':>6}")
    for sig, lift in sorted(lifts.items(), key=lambda x: -x[1]):
        hw = CURRENT_WEIGHTS.get(sig, "—")
        cw = cal_weights.get(sig, "—")
        print(f"  {sig:<28}  {lift:>+6.3f}  {str(hw):>6}  {str(cw):>6}")

    print(f"\n  ⚠  Unvalidated signals (keep hand weights):")
    print(f"  bulk_deals, sast, promoter, delivery, options PCR, results,")
    print(f"  bse_announcements, fii_dii, gift_nifty, sector_rotation")

    # Save outputs
    OUT_WEIGHTS.write_text(json.dumps({
        "method": args.method,
        "note": "OHLCV-derivable signals only; event signals not included",
        "calibrated": cal_weights,
        "expectancy_lifts": lifts,
        "hand_weights_for_comparison": CURRENT_WEIGHTS,
    }, indent=2))
    print(f"\n  Calibrated weights → {OUT_WEIGHTS.name}")

    OUT_WF.write_text(json.dumps(wf, indent=2))
    print(f"  Walk-forward results → {OUT_WF.name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
