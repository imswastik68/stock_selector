"""
Real signal backtest — replays enrich_with_technicals over historical daily OHLCV.

Key design decisions:
  - Calls the REAL src/technicals.py functions (same code path as live system).
  - Simulates real SL=2ATR / T1=3ATR trade mechanics via src/trade_sim.py.
  - Caches OHLCV to cache/backtest_ohlcv.parquet (re-runs are instant).
  - Emits outputs/backtest_trades.csv and outputs/backtest_signal_stats.json.

Limitations (documented):
  - Only OHLCV-derivable signals are tested (13/30). Event signals (bulk deals,
    SAST, promoter, delivery, options PCR, BSE filings) have no free archive
    and keep hand weights in the live scorer.
  - Survivorship bias: universe = current NIFTY500 + SME constituents.
  - Nifty 20d return computed from the same download (no point-in-time issue there).

Usage:
  python scripts/backtest.py                   # full decade, full universe
  python scripts/backtest.py --weeks 52        # last 52 weeks
  python scripts/backtest.py --sample 50       # top-50 liquid stocks (fast sanity check)
  python scripts/backtest.py --weeks 52 --sample 50
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
    import yfinance as yf
except ImportError:
    sys.exit("Run: pip install yfinance pandas")

from src.technicals import enrich_with_technicals, compute_entry_levels
from src.trade_sim import simulate_raw
from src.scorer import _compute_score
from src.costs import round_trip_cost_pct

CACHE_DIR   = ROOT / "cache"
OUTPUTS     = ROOT / "outputs"
_NIFTY      = "^NSEI"
_CACHE_DIR2 = CACHE_DIR / "backtest_ohlcv"   # per-ticker CSV cache directory
_NIFTY_CSV  = CACHE_DIR / "backtest_nifty.csv"

# OHLCV-derivable signals (subset of SHORT_TERM_WEIGHTS that we can test)
TESTABLE_SIGNALS = [
    "rsi_momentum", "rs_vs_nifty", "rsi_bearish_div", "rsi_bullish_div",
    "macd_bearish_cross",
    "bb_squeeze_breakout", "bullish_candle", "bearish_candle",
    "weekly_trend_aligned", "rs_quality_strong",
    # volume-derived (computed from OHLCV):
    "volume_surge", "distribution",
]

# Direction-congruent signal sets — a trade only gets generated if it has >=1 active
# signal congruent with its Wyckoff-derived direction. Matches the live scorer's own
# empirical weight signs (SHORT_TERM_WEIGHTS/DISQUALIFIER_WEIGHTS in src/scorer.py),
# not a literal reading of each signal's name — e.g. rsi_bearish_div carries a
# positive buy-side weight there (+3, "data shows +1.04 lift over 15k trades").
# Previously every direction was gated on the SAME bullish-only signal set, which
# biased sell-side stats (a sell only ever fired because a bullish signal happened
# to also be present that day, not because of an actual bearish signal).
BUY_CONGRUENT = {
    "rsi_momentum", "rs_vs_nifty", "rsi_bearish_div", "bb_squeeze_breakout",
    "bullish_candle", "weekly_trend_aligned", "rs_quality_strong", "volume_surge",
}
SELL_CONGRUENT = {
    "macd_bearish_cross", "bearish_candle", "rsi_bullish_div", "distribution", "volume_surge",
}

BATCH = 50


# ── universe ──────────────────────────────────────────────────────────────────

def _load_universe(sample: int | None) -> list[str]:
    tickers: list[str] = []
    for csv_path in [ROOT / "data" / "nifty500.csv", ROOT / "data" / "sme_list.csv"]:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            col = next((c for c in df.columns if "symbol" in c.lower()), df.columns[0])
            tickers += [f"{s.strip().upper()}.NS" for s in df[col].dropna()]
    if not tickers:
        print("[backtest] WARNING: no universe CSVs, using fallback 20 tickers")
        tickers = [
            "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
            "SBIN.NS","AXISBANK.NS","WIPRO.NS","SUNPHARMA.NS","HCLTECH.NS",
            "MARUTI.NS","TITAN.NS","BAJFINANCE.NS","NESTLEIND.NS","LT.NS",
            "ASIANPAINT.NS","ULTRACEMCO.NS","ITC.NS","BHARTIARTL.NS","KOTAKBANK.NS",
        ]
    uniq = list(dict.fromkeys(tickers))
    # For --sample, take the first N (NIFTY500 comes first → large/mid caps)
    return uniq[:sample] if sample else uniq


# ── OHLCV download with parquet cache ────────────────────────────────────────

def _ticker_csv(ticker: str) -> Path:
    safe = ticker.replace(".", "_").replace("/", "_")
    return _CACHE_DIR2 / f"{safe}.csv"


def _download_and_cache(tickers: list[str], start: date, end: date,
                        force: bool = False) -> dict[str, pd.DataFrame]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_DIR2.mkdir(parents=True, exist_ok=True)

    cached_map: dict[str, pd.DataFrame] = {}
    if not force:
        for t in tickers:
            csv = _ticker_csv(t)
            if csv.exists():
                try:
                    df = pd.read_csv(csv, index_col=0, parse_dates=True)
                    df = df.dropna(how="all")
                    df = df[df["Close"].notna()]
                    if not df.empty:
                        cached_map[t] = df
                except Exception:
                    pass
        if cached_map:
            print(f"[backtest] loaded {len(cached_map)}/{len(tickers)} tickers from CSV cache")

    missing = [t for t in tickers if t not in cached_map]
    if not missing:
        return {t: cached_map[t] for t in tickers if t in cached_map}

    print(f"[backtest] downloading {len(missing)} tickers in batches of {BATCH} ...")
    fresh: dict[str, pd.DataFrame] = {}
    batches = [missing[i:i+BATCH] for i in range(0, len(missing), BATCH)]
    for i, batch in enumerate(batches, 1):
        print(f"[backtest] batch {i}/{len(batches)} ({len(batch)} tickers)...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(
                    batch, start=start.isoformat(), end=end.isoformat(),
                    auto_adjust=True, progress=False, group_by="ticker", threads=True,
                )
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                for t in batch:
                    if t in raw.columns.get_level_values(0):
                        df = raw[t].dropna(how="all")
                        df = df[df["Close"].notna()]
                        if not df.empty:
                            fresh[t] = df
                            try:
                                df.to_csv(_ticker_csv(t))
                            except Exception:
                                pass
            elif len(batch) == 1:
                df = raw.dropna(how="all")
                df = df[df["Close"].notna()]
                if not df.empty:
                    fresh[batch[0]] = df
                    try:
                        df.to_csv(_ticker_csv(batch[0]))
                    except Exception:
                        pass
        except Exception as exc:
            print(f"[backtest] batch {i} error: {exc}")

    all_map = {**cached_map, **fresh}
    print(f"[backtest] {len(fresh)} new tickers downloaded, {len(all_map)} total")
    return {t: all_map[t] for t in tickers if t in all_map}


def _download_nifty(start: date, end: date) -> pd.DataFrame:
    if _NIFTY_CSV.exists():
        try:
            df = pd.read_csv(_NIFTY_CSV, index_col=0, parse_dates=True)
            df = df.dropna(how="all")
            df = df[df["Close"].notna()]
            if not df.empty and df.index[-1].date() >= end - timedelta(days=5):
                print(f"[backtest] Nifty loaded from cache ({len(df)} bars)")
                return df
        except Exception:
            pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(_NIFTY, start=start.isoformat(), end=end.isoformat(),
                          auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw[_NIFTY].dropna(how="all") if _NIFTY in raw.columns.get_level_values(0) else raw.iloc[:, :5]
    else:
        df = raw
    df = df.dropna(how="all")
    df = df[df["Close"].notna()]
    try:
        df.to_csv(_NIFTY_CSV)
    except Exception:
        pass
    return df


# ── signal extraction ─────────────────────────────────────────────────────────

def _volume_signals(df_slice: pd.DataFrame) -> dict[str, bool]:
    """Volume surge and distribution from daily OHLCV slice."""
    if len(df_slice) < 5:
        return {"volume_surge": False, "distribution": False}
    vol   = df_slice["Volume"].squeeze()
    close = df_slice["Close"].squeeze()
    today_vol = float(vol.iloc[-1])
    avg_30d   = float(vol.iloc[:-1].tail(30).mean()) if len(vol) > 1 else today_vol
    if avg_30d == 0:
        return {"volume_surge": False, "distribution": False}
    ratio = today_vol / avg_30d
    surge = ratio >= 2.5
    dist  = surge and float(close.iloc[-1]) < float(close.iloc[-2])
    return {"volume_surge": surge, "distribution": dist}


def _extract_signals(df_slice: pd.DataFrame, nifty_20d: float | None,
                     atr: float, close: float) -> dict[str, bool]:
    """Call real enrich_with_technicals + volume signals."""
    if len(df_slice) < 50:
        return {s: False for s in TESTABLE_SIGNALS}
    try:
        tech = enrich_with_technicals(df_slice, close, atr, nifty_20d_return=nifty_20d)
    except Exception:
        return {s: False for s in TESTABLE_SIGNALS}
    vol_sigs = _volume_signals(df_slice)
    return {
        "rsi_momentum":        tech.get("rsi_momentum", False),
        "rs_vs_nifty":         tech.get("rs_vs_nifty", False),
        "rsi_bearish_div":     tech.get("rsi_bearish_div", False),
        "rsi_bullish_div":     tech.get("rsi_bullish_div", False),
        "macd_bearish_cross":  tech.get("macd_bearish_cross", False),
        "bb_squeeze_breakout": tech.get("bb_squeeze_breakout", False),
        "bullish_candle":      tech.get("bullish_candle", False),
        "bearish_candle":      tech.get("bearish_candle", False),
        "weekly_trend_aligned":tech.get("weekly_trend_aligned", False),
        "rs_quality_strong":   tech.get("rs_quality_strong", False),
        "volume_surge":        vol_sigs["volume_surge"],
        "distribution":        vol_sigs["distribution"],
    }


def _classify_nifty_trend_series(nifty_close: pd.Series) -> pd.Series:
    """
    EMA20/50/200 stack classifier — copied from calibrate_weights.py:_classify_nifty_trend
    (scripts aren't a package). Kept identical so the per-date regime used to score a
    trade here matches what calibrate_weights.py / mine_big_movers.py compute for the
    same date. Previously every date was scored with nifty_trend='ranging' regardless
    of the actual regime, so REGIME_WEIGHTS never applied correctly in the backtest.
    """
    ema20  = nifty_close.ewm(span=20,  adjust=False).mean()
    ema50  = nifty_close.ewm(span=50,  adjust=False).mean()
    ema200 = nifty_close.ewm(span=200, adjust=False).mean()
    labels = pd.Series("ranging", index=nifty_close.index)
    labels[ema20 > ema50] = "ranging"
    labels[(ema20 > ema50) & (ema50 > ema200)] = "uptrend"
    labels[(ema20 < ema50) & (ema50 < ema200)] = "downtrend"
    return labels


def _compute_atr(df_slice: pd.DataFrame) -> float:
    if len(df_slice) < 15:
        return 0.0
    h = df_slice["High"].squeeze()
    l = df_slice["Low"].squeeze()
    c = df_slice["Close"].squeeze()
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1])


# ── main backtest loop ────────────────────────────────────────────────────────

def run_backtest(weeks: int, sample: int | None) -> None:
    today = date.today()
    end   = today
    start = today - timedelta(days=weeks * 7 + 252)  # extra year for warmup
    full_start = today - timedelta(days=30 * 365)     # 30-year max for cache

    universe = _load_universe(sample)
    print(f"[backtest] universe: {len(universe)} tickers | test window: {weeks} weeks back")

    # Download / load cached OHLCV
    ohlcv = _download_and_cache(universe, full_start, end)
    nifty_df = _download_nifty(full_start, end)
    nifty_close = nifty_df["Close"].squeeze() if not nifty_df.empty else pd.Series(dtype=float)

    print(f"[backtest] got OHLCV for {len(ohlcv)}/{len(universe)} tickers, "
          f"{len(nifty_close)} Nifty bars")

    trend_series = _classify_nifty_trend_series(nifty_close) if not nifty_close.empty else pd.Series(dtype=object)

    # Walk-forward: every 5 trading days over the test window
    signal_dates: list[pd.Timestamp] = []
    nifty_ts = nifty_close[nifty_close.index >= pd.Timestamp(start)]
    step = 5
    idxs = list(range(0, len(nifty_ts), step))
    for i in idxs:
        signal_dates.append(nifty_ts.index[i])
    print(f"[backtest] {len(signal_dates)} signal dates × {len(ohlcv)} tickers ...")

    rows: list[dict] = []
    total_evals = 0

    for as_of in signal_dates:
        # Nifty 20d return at this date
        nifty_hist = nifty_close[nifty_close.index <= as_of]
        nifty_20d: float | None = None
        if len(nifty_hist) >= 20:
            nifty_20d = float(nifty_hist.iloc[-1] / nifty_hist.iloc[-20] - 1)

        regime = trend_series.get(as_of, "ranging") if not trend_series.empty else "ranging"

        for ticker, full_df in ohlcv.items():
            df_slice = full_df[full_df.index <= as_of]
            if len(df_slice) < 60:
                continue

            close = float(df_slice["Close"].iloc[-1])
            if close <= 0:
                continue
            atr = _compute_atr(df_slice)
            if atr <= 0:
                continue

            signals = _extract_signals(df_slice, nifty_20d, atr, close)

            # Compute direction first — the active-signal filter below needs to know
            # which congruent set to check (see BUY_CONGRUENT/SELL_CONGRUENT comment).
            from src.technicals import classify_wyckoff_phase
            phase = classify_wyckoff_phase(df_slice.tail(90))
            direction = "buy" if phase in {"MARKUP","ACCUMULATION_C","ACCUMULATION_D"} else \
                        "sell" if phase in {"MARKDOWN","DISTRIBUTION_C","DISTRIBUTION_D"} else None
            if direction is None:
                continue

            # Skip bars with no direction-congruent active signal (no edge, no trade)
            congruent = BUY_CONGRUENT if direction == "buy" else SELL_CONGRUENT
            active = [s for s in TESTABLE_SIGNALS if signals.get(s) and s in congruent]
            if not active:
                continue

            levels = compute_entry_levels(close, atr, direction)
            if not levels:
                continue

            def _p(s: str) -> float:
                return float(s.replace("₹", "").replace(",", ""))

            entry_lo  = _p(levels["entry_zone"].split("-₹")[0].replace("₹",""))
            entry_hi  = _p(levels["entry_zone"].split("-₹")[1])
            entry_mid = (entry_lo + entry_hi) / 2
            sl        = _p(levels["stop_loss"])
            t1        = _p(levels["target_1"])

            # Simulate trade forward (next N bars after as_of)
            result = simulate_raw(entry_lo, entry_hi, entry_mid, sl, t1,
                                  direction, as_of, full_df,
                                  cost_pct=round_trip_cost_pct(direction))
            total_evals += 1

            score, _ = _compute_score(signals, nifty_trend=regime)
            row = {
                "as_of":     as_of.date().isoformat(),
                "ticker":    ticker,
                "direction": direction,
                "phase":     phase,
                "close":     close,
                "atr_pct":   round(atr/close*100, 2),
                "score":     score,
                "outcome":   result.get("outcome", "no_data"),
                "return_pct":result.get("return_pct"),
                "return_pct_gross": result.get("return_pct_gross"),
                "cost_pct":  result.get("cost_pct"),
                "days_held": result.get("days_held"),
                "mae_pct":   result.get("mae_pct"),
                "mfe_pct":   result.get("mfe_pct"),
                "fwd_5d":    result.get("fwd_5d_pct"),
                "fwd_10d":   result.get("fwd_10d_pct"),
                "fwd_20d":   result.get("fwd_20d_pct"),
                "triggered": result.get("triggered", False),
                **{f"sig_{s}": signals.get(s, False) for s in TESTABLE_SIGNALS},
            }
            rows.append(row)

    print(f"[backtest] {total_evals} evaluations → {len(rows)} with active signals")

    if not rows:
        print("[backtest] no trades generated — check universe or date range")
        return

    df_trades = pd.DataFrame(rows)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    trades_path = OUTPUTS / "backtest_trades.csv"
    df_trades.to_csv(trades_path, index=False)
    print(f"[backtest] trades saved → {trades_path.name}")

    # ── per-signal stats ──────────────────────────────────────────────────────
    triggered = df_trades[df_trades["triggered"] == True].copy()
    closed    = triggered[triggered["outcome"].isin(["t1_hit", "sl_hit"])].copy()
    print(f"[backtest] triggered={len(triggered)}, closed={len(closed)}")

    if closed.empty:
        print("[backtest] no closed trades yet — extend window or wait for results")
        return

    baseline_wr  = (closed["outcome"] == "t1_hit").mean()
    baseline_ret = closed["return_pct"].mean()          # net of round-trip cost (Phase 1)
    baseline_ret_gross = closed["return_pct_gross"].mean() if "return_pct_gross" in closed.columns else None
    baseline_exp = (closed["return_pct"] * (2/3) - closed["return_pct"].abs() * (1/3)).mean()

    sig_stats: dict[str, dict] = {}
    for sig in TESTABLE_SIGNALS:
        col = f"sig_{sig}"
        if col not in closed.columns:
            continue
        with_sig    = closed[closed[col] == True]
        without_sig = closed[closed[col] == False]
        n = len(with_sig)
        if n < 5:
            continue
        wr   = float((with_sig["outcome"] == "t1_hit").mean())
        ret  = float(with_sig["return_pct"].mean())
        wr_lift  = round((wr  - baseline_wr) * 100, 2)
        ret_lift = round(ret  - baseline_ret, 2)
        sig_stats[sig] = {
            "n":          n,
            "win_rate":   round(wr  * 100, 1),
            "avg_return": round(ret, 2),
            "wr_lift_pp": wr_lift,
            "ret_lift":   ret_lift,
            "n_without":  len(without_sig),
            "wr_without": round(float((without_sig["outcome"]=="t1_hit").mean())*100,1) if len(without_sig) else None,
        }

    import json
    stats_path = OUTPUTS / "backtest_signal_stats.json"
    stats_path.write_text(json.dumps({
        "meta": {
            "weeks": weeks,
            "universe": len(universe),
            "tickers_with_data": len(ohlcv),
            "signal_dates": len(signal_dates),
            "triggered_trades": len(triggered),
            "closed_trades": len(closed),
            "baseline_win_rate_pct": round(baseline_wr*100,1),
            "baseline_avg_return_net_pct": round(baseline_ret,2),
            "baseline_avg_return_gross_pct": round(baseline_ret_gross,2) if baseline_ret_gross is not None else None,
            "note_costs": "return_pct/baseline_avg_return_net_pct are net of round-trip cost (src/costs.py); *_gross fields are pre-cost",
            "note_survivorship_bias": "universe=current constituents only",
            "note_untested_signals": "bulk_deals,sast,promoter,delivery,options,results,bse,fii_dii,gift_nifty,sector_rotation",
        },
        "signals": dict(sorted(sig_stats.items(), key=lambda x: -x[1]["wr_lift_pp"])),
    }, indent=2))
    print(f"[backtest] signal stats saved → {stats_path.name}")

    # ── console summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS — {weeks}w | {len(closed)} closed trades")
    print(f"{'='*60}")
    print(f"  Baseline win rate : {baseline_wr*100:.1f}%")
    print(f"  Baseline avg ret  : {baseline_ret:+.2f}%  (net of cost; gross {baseline_ret_gross:+.2f}%)"
          if baseline_ret_gross is not None else f"  Baseline avg ret  : {baseline_ret:+.2f}%")
    print(f"\n  SIGNAL WIN-RATE LIFT (ranked):")
    print(f"  {'Signal':<28}  {'N':>5}  {'WR%':>5}  {'Lift pp':>7}  {'Ret%':>6}")
    for sig, st in sorted(sig_stats.items(), key=lambda x: -x[1]["wr_lift_pp"]):
        print(f"  {sig:<28}  {st['n']:>5}  {st['win_rate']:>5}  "
              f"{st['wr_lift_pp']:>+7.1f}  {st['avg_return']:>+6.2f}")
    print(f"{'='*60}")
    print(f"\nNext step: python scripts/calibrate_weights.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks",  type=int, default=260, help="Weeks to test (default 260 = 5y)")
    ap.add_argument("--sample", type=int, default=None, help="Limit universe to N tickers (fast sanity check)")
    ap.add_argument("--force",  action="store_true", help="Re-download ignoring cache")
    args = ap.parse_args()
    run_backtest(args.weeks, args.sample)
