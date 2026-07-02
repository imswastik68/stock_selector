"""
Mine outputs/backtest_trades.csv + cached OHLCV to find what predicts a
>=10% favorable move within 10 trading bars of entry, both directions.

This gates the BIG MOVER tier (Phase 4): do not hardcode a volatility/score
gate without running this first. mfe_pct in backtest_trades.csv runs to
end-of-cached-data, not a 10-day horizon — it is NOT usable as the big-mover
label. This script recomputes a proper 10-bar excursion from the same
per-ticker OHLCV cache backtest_exits.py uses, and applies a turnover filter
to kill the spurious high-base-rate reading from illiquid names.

Reuses:
  outputs/backtest_trades.csv   — as_of, ticker, direction, phase, close, atr_pct, score, sig_*
  cache/backtest_ohlcv/*.csv    — per-ticker daily OHLCV (backtest.py/backtest_exits.py cache)
  cache/backtest_nifty.csv      — Nifty daily close (backtest.py cache)
  src/technicals.compute_entry_levels — reconstruct entry/sl/t1 from close+atr
  (entry-trigger logic below mirrors src/trade_sim.simulate_raw's first-2-bar rule
   exactly, so entry_idx/entry_price match what the original backtest computed)

Usage: python scripts/mine_big_movers.py
Output: outputs/big_mover_analysis.json + outputs/big_mover_samples.csv + console tables
"""

from __future__ import annotations

import itertools
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

from src.technicals import compute_entry_levels

TRADES_CSV  = ROOT / "outputs" / "backtest_trades.csv"
OHLCV_CACHE = ROOT / "cache" / "backtest_ohlcv"
NIFTY_CSV   = ROOT / "cache" / "backtest_nifty.csv"
SAMPLES_OUT = ROOT / "outputs" / "big_mover_samples.csv"
ANALYSIS_OUT = ROOT / "outputs" / "big_mover_analysis.json"

HORIZON          = 10     # trading bars
BIG_MOVE_PCT     = 10.0   # target favorable move %
ADVERSE_R_MULT   = 2.0    # adverse excursion treated as "loss first", in ATR multiples
MIN_TURNOVER_CR  = 2.0    # median(close*volume) over trailing 30 bars, in INR crore
COST_BUY_PCT     = 0.30   # round-trip cost assumption, matches Phase 1 src/costs.py
COST_SELL_PCT    = 0.15

SIGNAL_NAMES = [
    "rsi_momentum", "rs_vs_nifty", "rsi_bearish_div", "rsi_bullish_div",
    "macd_bearish_cross", "bb_squeeze_breakout", "bullish_candle", "bearish_candle",
    "weekly_trend_aligned", "rs_quality_strong", "volume_surge", "distribution",
]

ATR_BINS   = [0, 2.5, 3.5, 5.0, 7.0, np.inf]
ATR_LABELS = ["<2.5", "2.5-3.5", "3.5-5", "5-7", ">=7"]
SCORE_BINS   = [-np.inf, 2, 4, 8, np.inf]
SCORE_LABELS = ["<2", "2-4", "5-7", ">=8"]

GATE_ATR_MINS   = [2.5, 3.0, 3.5, 4.0, 5.0]
GATE_SCORE_MINS = [2, 4, 6]


# ── loaders (mirror backtest_exits.py conventions) ──────────────────────────

def _load_ticker_df(ticker: str) -> pd.DataFrame | None:
    fname = ticker.replace(".", "_")
    path = OHLCV_CACHE / f"{fname}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return None


def _load_nifty() -> pd.DataFrame | None:
    if not NIFTY_CSV.exists():
        return None
    try:
        df = pd.read_csv(NIFTY_CSV, index_col=0)
        # older yfinance multi-index CSV dumps leave a 2-row "Ticker/Date" header
        # artifact as data rows (non-numeric Close, unparseable index) — strip them.
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df[df["Close"].notna()]
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()]
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return None


def _classify_nifty_trend(nifty_df: pd.DataFrame) -> pd.Series:
    """Copied from scripts/calibrate_weights.py:_classify_nifty_trend (scripts aren't a package)."""
    close = nifty_df["Close"].squeeze()
    ema20  = close.ewm(span=20,  adjust=False).mean()
    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    labels = pd.Series("ranging", index=close.index)
    labels[ema20 > ema50] = "ranging"
    labels[(ema20 > ema50) & (ema50 > ema200)] = "uptrend"
    labels[(ema20 < ema50) & (ema50 < ema200)] = "downtrend"
    return labels


def _find_entry(df: pd.DataFrame, as_of: pd.Timestamp, entry_lo: float, entry_hi: float,
                 entry_mid: float, direction: str):
    """Mirrors src/trade_sim.simulate_raw's entry-trigger rule (first 2 bars after as_of)."""
    future = df[df.index > as_of]
    window = future.iloc[:2]
    for idx, row in window.iterrows():
        bar_low, bar_high, bar_open = float(row["Low"]), float(row["High"]), float(row["Open"])
        if direction == "buy":
            if bar_low <= entry_hi:
                return idx, (entry_mid if bar_open <= entry_hi else entry_hi)
        else:
            if bar_high >= entry_lo:
                return idx, (entry_mid if bar_open >= entry_lo else entry_lo)
    return None, None


def _turnover_ok(df: pd.DataFrame, as_of: pd.Timestamp) -> bool:
    hist = df[df.index <= as_of].tail(30)
    if len(hist) < 10:
        return False
    turnover_cr = float((hist["Close"] * hist["Volume"]).median()) / 1e7
    last = hist.iloc[-1]
    return turnover_cr >= MIN_TURNOVER_CR and float(last["High"]) != float(last["Low"])


def _excursions(df: pd.DataFrame, entry_idx, entry_price: float, direction: str,
                 atr: float, horizon: int = HORIZON) -> dict:
    """
    Over the first `horizon` bars strictly after entry: mfe_h/mae_h (favorable/adverse
    excursion %, both reported as positive-is-good), hit_big_first (10% touched before
    a 2xATR adverse touch, bar-ordered; same-bar tie counts as False — matches the
    live simulator's SL-wins-ties convention), bars_to_big.
    """
    future = df[df.index > entry_idx]
    window = future.iloc[:horizon]
    if window.empty:
        return {"mfe_h": None, "mae_h": None, "hit_big_first": False, "bars_to_big": None}

    adverse_dist = ADVERSE_R_MULT * atr

    if direction == "buy":
        mfe_h = (float(window["High"].max()) - entry_price) / entry_price * 100
        mae_h = (float(window["Low"].min())  - entry_price) / entry_price * 100
    else:
        mfe_h = (entry_price - float(window["Low"].min()))  / entry_price * 100
        mae_h = (entry_price - float(window["High"].max())) / entry_price * 100

    hit_big_first = False
    bars_to_big = None
    for bar_n, (_, row) in enumerate(window.iterrows(), start=1):
        if direction == "buy":
            big_touch     = float(row["High"]) >= entry_price * (1 + BIG_MOVE_PCT / 100)
            adverse_touch = float(row["Low"])  <= entry_price - adverse_dist
        else:
            big_touch     = float(row["Low"])  <= entry_price * (1 - BIG_MOVE_PCT / 100)
            adverse_touch = float(row["High"]) >= entry_price + adverse_dist
        if big_touch and adverse_touch:
            break
        if big_touch:
            hit_big_first, bars_to_big = True, bar_n
            break
        if adverse_touch:
            break

    return {"mfe_h": round(mfe_h, 2), "mae_h": round(mae_h, 2),
            "hit_big_first": hit_big_first, "bars_to_big": bars_to_big}


# ── build the sample set ─────────────────────────────────────────────────────

def build_samples() -> pd.DataFrame:
    if not TRADES_CSV.exists():
        sys.exit(f"Not found: {TRADES_CSV}\nRun: python scripts/backtest.py first")

    df = pd.read_csv(TRADES_CSV, parse_dates=["as_of"])
    df = df[df["triggered"] == True].copy()
    print(f"[mine] {len(df)} triggered trades to analyze")

    nifty_df = _load_nifty()
    trend_map = None
    if nifty_df is not None and not nifty_df.empty:
        trend_map = _classify_nifty_trend(nifty_df)
        trend_map.index = pd.to_datetime(trend_map.index).normalize()
    else:
        print("[mine] WARNING: no cached Nifty history — regime bucketing disabled")

    tickers = df["ticker"].unique()
    print(f"[mine] loading {len(tickers)} tickers from OHLCV cache...")
    ticker_dfs: dict[str, pd.DataFrame] = {}
    for t in tickers:
        td = _load_ticker_df(t)
        if td is not None:
            ticker_dfs[t] = td
    print(f"[mine] loaded {len(ticker_dfs)}/{len(tickers)}")

    records: list[dict] = []
    skipped_no_cache = skipped_no_entry = skipped_illiquid = 0

    for _, row in df.iterrows():
        ticker    = row["ticker"]
        direction = row["direction"]
        close     = float(row["close"])
        atr_pct   = float(row["atr_pct"])
        as_of     = pd.Timestamp(row["as_of"])

        if ticker not in ticker_dfs:
            skipped_no_cache += 1
            continue
        td = ticker_dfs[ticker]

        atr = close * atr_pct / 100.0
        levels = compute_entry_levels(close, atr, direction)
        if not levels:
            skipped_no_entry += 1
            continue
        try:
            ez = levels["entry_zone"].replace("₹", "")
            entry_lo, entry_hi = (float(x) for x in ez.split("-"))
            entry_mid = (entry_lo + entry_hi) / 2
        except Exception:
            skipped_no_entry += 1
            continue

        entry_idx, entry_price = _find_entry(td, as_of, entry_lo, entry_hi, entry_mid, direction)
        if entry_idx is None:
            skipped_no_entry += 1
            continue

        if not _turnover_ok(td, as_of):
            skipped_illiquid += 1
            continue

        exc = _excursions(td, entry_idx, entry_price, direction, atr)
        if exc["mfe_h"] is None:
            skipped_no_entry += 1
            continue

        regime = "ranging"
        if trend_map is not None:
            key = as_of.normalize()
            regime = trend_map.get(key, "ranging")

        rec = {
            "as_of":     as_of,
            "ticker":    ticker,
            "direction": direction,
            "phase":     row.get("phase"),
            "score":     row.get("score"),
            "atr_pct":   atr_pct,
            "regime":    regime,
            "return_pct": row.get("return_pct"),
            "mfe_h":     exc["mfe_h"],
            "mae_h":     exc["mae_h"],
            "hit_big_first": exc["hit_big_first"],
            "big_10_touch":  exc["mfe_h"] >= BIG_MOVE_PCT,
            "bars_to_big":   exc["bars_to_big"],
        }
        for s in SIGNAL_NAMES:
            col = f"sig_{s}"
            rec[col] = bool(row[col]) if col in row.index else False
        records.append(rec)

    print(f"[mine] skipped: no_cache={skipped_no_cache} no_entry={skipped_no_entry} illiquid={skipped_illiquid}")
    print(f"[mine] {len(records)} valid liquid samples for analysis")

    out = pd.DataFrame(records)
    if out.empty:
        sys.exit("[mine] no valid samples — check OHLCV cache")
    out = out.sort_values("as_of").reset_index(drop=True)
    out.to_csv(SAMPLES_OUT, index=False)
    print(f"[mine] samples → {SAMPLES_OUT.name}")
    return out


# ── reporting ─────────────────────────────────────────────────────────────────

def _net_expectancy(sub: pd.DataFrame) -> float | None:
    if sub.empty:
        return None
    cost = sub["direction"].map({"buy": COST_BUY_PCT, "sell": COST_SELL_PCT}).fillna(COST_BUY_PCT)
    net = sub["return_pct"] - cost
    return round(float(net.mean()), 3)


def report_base_rates(df: pd.DataFrame) -> dict:
    out = {}
    print(f"\n{'='*70}\n  BASE RATES — P(big move) per direction\n{'='*70}")
    for direction in ("buy", "sell"):
        sub = df[df["direction"] == direction]
        n = len(sub)
        if n == 0:
            continue
        p_touch = float(sub["big_10_touch"].mean())
        p_hbf   = float(sub["hit_big_first"].mean())
        net_exp = _net_expectancy(sub)
        gross_exp = round(float(sub["return_pct"].mean()), 3) if "return_pct" in sub else None
        print(f"  {direction:<5}  n={n:>6}  P(touch>=10%)={p_touch*100:5.1f}%  "
              f"P(hit_big_first)={p_hbf*100:5.1f}%  gross_exp={gross_exp}  net_exp={net_exp}")
        out[direction] = {"n": n, "p_big_touch": round(p_touch, 4), "p_hit_big_first": round(p_hbf, 4),
                           "gross_expectancy": gross_exp, "net_expectancy": net_exp}
    return out


def report_atr_buckets(df: pd.DataFrame) -> dict:
    out = {}
    print(f"\n{'='*70}\n  ATR TIER x DIRECTION\n{'='*70}")
    print(f"  {'ATR tier':<10} {'Dir':<5} {'N':>6} {'P(big)%':>9} {'NetExp':>8}")
    df = df.copy()
    df["atr_tier"] = pd.cut(df["atr_pct"], bins=ATR_BINS, labels=ATR_LABELS)
    for tier in ATR_LABELS:
        for direction in ("buy", "sell"):
            sub = df[(df["atr_tier"] == tier) & (df["direction"] == direction)]
            n = len(sub)
            if n == 0:
                continue
            p_big = float(sub["big_10_touch"].mean())
            net_exp = _net_expectancy(sub)
            print(f"  {tier:<10} {direction:<5} {n:>6} {p_big*100:>8.1f}% {net_exp:>+8.3f}")
            out[f"{tier}|{direction}"] = {"n": n, "p_big_touch": round(p_big, 4), "net_expectancy": net_exp}
    return out


def report_phase_regime_score(df: pd.DataFrame) -> dict:
    out = {"phase": {}, "regime": {}, "score_band": {}}
    df = df.copy()
    df["score_band"] = pd.cut(df["score"], bins=SCORE_BINS, labels=SCORE_LABELS)

    for key, label in [("phase", "PHASE"), ("regime", "REGIME"), ("score_band", "SCORE BAND")]:
        print(f"\n{'='*70}\n  {label}\n{'='*70}")
        print(f"  {'Value':<16} {'Dir':<5} {'N':>6} {'P(big)%':>9}")
        for val in sorted(df[key].dropna().unique(), key=str):
            for direction in ("buy", "sell"):
                sub = df[(df[key] == val) & (df["direction"] == direction)]
                n = len(sub)
                if n < 20:
                    continue
                p_big = float(sub["big_10_touch"].mean())
                print(f"  {str(val):<16} {direction:<5} {n:>6} {p_big*100:>8.1f}%")
                out[key][f"{val}|{direction}"] = {"n": n, "p_big_touch": round(p_big, 4)}
    return out


def report_signal_lift(df: pd.DataFrame) -> dict:
    out = {}
    print(f"\n{'='*70}\n  PER-SIGNAL LIFT on P(big_10_touch) and P(hit_big_first)\n{'='*70}")
    print(f"  {'Signal':<24} {'N w/sig':>8} {'P(big)%':>9} {'Base%':>7} {'Lift pp':>8} {'P(hbf)%':>9}")
    base_touch = float(df["big_10_touch"].mean())
    for s in SIGNAL_NAMES:
        col = f"sig_{s}"
        if col not in df.columns:
            continue
        sub = df[df[col] == True]
        n = len(sub)
        if n < 30:
            continue
        p_big = float(sub["big_10_touch"].mean())
        p_hbf = float(sub["hit_big_first"].mean())
        lift = (p_big - base_touch) * 100
        print(f"  {s:<24} {n:>8} {p_big*100:>8.1f}% {base_touch*100:>6.1f}% {lift:>+7.1f} {p_hbf*100:>8.1f}%")
        out[s] = {"n": n, "p_big_touch": round(p_big, 4), "lift_pp": round(lift, 2),
                   "p_hit_big_first": round(p_hbf, 4)}
    return out


def report_signal_pairs(df: pd.DataFrame) -> list[dict]:
    print(f"\n{'='*70}\n  SIGNAL PAIRS (n>=300), top 20 by lift on P(big_10_touch)\n{'='*70}")
    base_touch = float(df["big_10_touch"].mean())
    pairs = []
    for a, b in itertools.combinations(SIGNAL_NAMES, 2):
        ca, cb = f"sig_{a}", f"sig_{b}"
        if ca not in df.columns or cb not in df.columns:
            continue
        sub = df[(df[ca] == True) & (df[cb] == True)]
        n = len(sub)
        if n < 300:
            continue
        p_big = float(sub["big_10_touch"].mean())
        lift = (p_big - base_touch) * 100
        pairs.append({"signals": [a, b], "n": n, "p_big_touch": round(p_big, 4), "lift_pp": round(lift, 2)})
    pairs.sort(key=lambda r: -r["lift_pp"])
    for p in pairs[:20]:
        print(f"  {' + '.join(p['signals']):<48} n={p['n']:>6}  P(big)={p['p_big_touch']*100:5.1f}%  lift={p['lift_pp']:+.1f}pp")
    if not pairs:
        print("  (no signal pairs reach n>=300)")
    return pairs[:20]


def report_downtrend_short_edge(df: pd.DataFrame, fo_set: set[str] | None) -> dict:
    print(f"\n{'='*70}\n  DOWNTREND: shorts (F&O subset) vs longs\n{'='*70}")
    dt = df[df["regime"] == "downtrend"]
    out: dict = {}

    shorts = dt[dt["direction"] == "sell"]
    if fo_set:
        shorts_fo = shorts[shorts["ticker"].isin(fo_set)]
    else:
        shorts_fo = shorts
        print("  WARNING: no F&O set available — reporting all downtrend shorts unfiltered")
    longs = dt[dt["direction"] == "buy"]

    for label, sub in [("downtrend_shorts_fo", shorts_fo), ("downtrend_longs", longs)]:
        n = len(sub)
        if n == 0:
            print(f"  {label:<22} n=0")
            out[label] = {"n": 0}
            continue
        p_big = float(sub["big_10_touch"].mean())
        net_exp = _net_expectancy(sub)
        print(f"  {label:<22} n={n:>6}  P(big)={p_big*100:5.1f}%  net_exp={net_exp:+.3f}%")
        out[label] = {"n": n, "p_big_touch": round(p_big, 4), "net_expectancy": net_exp}

    # verdict: shorts must have n>=200 and net_expectancy within 0.5pp of longs (or better)
    s_stats, l_stats = out.get("downtrend_shorts_fo", {}), out.get("downtrend_longs", {})
    verdict = "insufficient_data"
    if s_stats.get("n", 0) >= 200 and l_stats.get("net_expectancy") is not None and s_stats.get("net_expectancy") is not None:
        verdict = "short_edge_positive" if s_stats["net_expectancy"] >= l_stats["net_expectancy"] - 0.5 else "short_edge_negative"
    print(f"  VERDICT: {verdict}")
    out["verdict"] = verdict
    return out


def gate_grid_search(df: pd.DataFrame) -> list[dict]:
    print(f"\n{'='*70}\n  GATE GRID SEARCH  (atr_min x score_min), both directions combined\n{'='*70}")
    print(f"  {'atr_min':>7} {'score_min':>9} {'N':>7} {'picks/day':>10} {'P(big)%':>9} {'NetExp':>8}")

    n_dates = df["as_of"].nunique() or 1
    base_touch = float(df["big_10_touch"].mean())
    results = []
    for atr_min in GATE_ATR_MINS:
        for score_min in GATE_SCORE_MINS:
            sub = df[(df["atr_pct"] >= atr_min) & (df["score"] >= score_min)]
            n = len(sub)
            if n < 20:
                continue
            picks_per_day = n / n_dates
            p_big = float(sub["big_10_touch"].mean())
            net_exp = _net_expectancy(sub)
            print(f"  {atr_min:>7.1f} {score_min:>9d} {n:>7} {picks_per_day:>10.2f} {p_big*100:>8.1f}% {net_exp:>+8.3f}")
            results.append({
                "atr_min": atr_min, "score_min": score_min, "n": n,
                "picks_per_day": round(picks_per_day, 3),
                "p_big_touch": round(p_big, 4), "net_expectancy": net_exp,
                "lift_vs_base": round((p_big - base_touch) * 100, 2),
            })
    return results


def choose_gate(df: pd.DataFrame, grid: list[dict]) -> dict:
    """
    Ship criteria (from plan): holdout P(big|gate) >= 1.5x holdout base rate,
    holdout n >= 200, implied pick rate <= 3/day. 70/30 date-split stability check
    on the best full-sample candidate.
    """
    print(f"\n{'='*70}\n  GATE SELECTION — 70/30 holdout check\n{'='*70}")

    df_sorted = df.sort_values("as_of").reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.7)
    split_date = df_sorted.iloc[split_idx]["as_of"]
    train = df_sorted[df_sorted["as_of"] < split_date]
    holdout = df_sorted[df_sorted["as_of"] >= split_date]
    holdout_base = float(holdout["big_10_touch"].mean()) if len(holdout) else 0.0
    holdout_dates = holdout["as_of"].nunique() or 1

    print(f"  train: {train['as_of'].min().date()} -> {split_date.date()} ({len(train)})")
    print(f"  holdout: {split_date.date()} -> {df_sorted['as_of'].max().date()} ({len(holdout)}, base rate {holdout_base*100:.1f}%)")

    candidates = sorted(grid, key=lambda r: -r["lift_vs_base"])
    for cand in candidates:
        atr_min, score_min = cand["atr_min"], cand["score_min"]
        h_sub = holdout[(holdout["atr_pct"] >= atr_min) & (holdout["score"] >= score_min)]
        h_n = len(h_sub)
        h_picks_per_day = h_n / holdout_dates
        h_p_big = float(h_sub["big_10_touch"].mean()) if h_n else 0.0

        ships = (h_n >= 200 and holdout_base > 0
                 and h_p_big >= 1.5 * holdout_base
                 and h_picks_per_day <= 3.0)
        print(f"  atr>={atr_min} score>={score_min}: holdout n={h_n} P(big)={h_p_big*100:.1f}% "
              f"(need >={1.5*holdout_base*100:.1f}%) picks/day={h_picks_per_day:.2f}  "
              f"{'SHIPS' if ships else 'fails'}")
        if ships:
            chosen = {
                "min_atr_pct": atr_min, "min_score": score_min,
                "holdout_n": h_n, "holdout_p_big_touch": round(h_p_big, 4),
                "holdout_base_rate": round(holdout_base, 4),
                "holdout_picks_per_day": round(h_picks_per_day, 3),
                "max_per_day": 3,
                "status": "shipped",
                "provenance": "scripts/mine_big_movers.py, 70/30 holdout on outputs/big_mover_samples.csv",
            }
            print(f"\n  CHOSEN GATE: {chosen}")
            return chosen

    print("\n  NO GATE COMBINATION CLEARS THE SHIP BAR.")
    print("  -> Phase 4 ships ATR-gate-only informational flag (no target/exit override).")
    fallback = {
        "min_atr_pct": max(GATE_ATR_MINS),
        "min_score": None,
        "status": "no_gate_found",
        "note": "ATR-gate-only informational flag; target/exit override dropped per plan Phase 0 ship gate",
        "provenance": "scripts/mine_big_movers.py, 70/30 holdout on outputs/big_mover_samples.csv",
    }
    return fallback


def main() -> None:
    df = build_samples()

    fo_set = None
    try:
        from src.data.fo_list import fetch_fo_eligible
        fo_set = fetch_fo_eligible()
    except Exception as exc:
        print(f"[mine] could not load F&O set for downtrend-short check: {exc}")

    analysis: dict = {}
    analysis["base_rates"] = report_base_rates(df)
    analysis["atr_buckets"] = report_atr_buckets(df)
    analysis.update(report_phase_regime_score(df))
    analysis["signal_lift"] = report_signal_lift(df)
    analysis["signal_pairs_top20"] = report_signal_pairs(df)
    analysis["downtrend_short_edge"] = report_downtrend_short_edge(df, fo_set)
    grid = gate_grid_search(df)
    analysis["gate_grid"] = grid
    analysis["chosen_gate"] = choose_gate(df, grid)

    analysis["meta"] = {
        "total_samples": len(df),
        "horizon_bars": HORIZON,
        "big_move_pct": BIG_MOVE_PCT,
        "adverse_r_mult": ADVERSE_R_MULT,
        "min_turnover_cr": MIN_TURNOVER_CR,
        "cost_buy_pct": COST_BUY_PCT,
        "cost_sell_pct": COST_SELL_PCT,
        "date_range": [df["as_of"].min().date().isoformat(), df["as_of"].max().date().isoformat()],
    }

    ANALYSIS_OUT.write_text(json.dumps(analysis, indent=2, default=str))
    print(f"\n{'='*70}\n  Results -> {ANALYSIS_OUT.name}\n{'='*70}")


if __name__ == "__main__":
    main()
