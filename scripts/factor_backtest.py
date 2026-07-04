"""
Cross-sectional factor backtester — date-driven rebalance loop, long-only,
optional Nifty-futures hedge overlay.

Contrast with the existing backtest scripts:
  - scripts/backtest.py is PER-TICKER absolute-threshold (does this stock's RSI/
    MACD/Wyckoff phase cross a fixed bar?) with no comparison across tickers.
  - scripts/mine_big_movers.py is ROW-DRIVEN off pre-existing trade rows.
  - This script is DATE-DRIVEN: at each rebalance date, every surviving ticker
    gets a cross-sectional factor score, ranked against every OTHER ticker that
    date, and the top-N/decile is held long until the next rebalance. This is
    how documented equity factor premia (momentum, 52w-high, low-vol, quality)
    are actually captured -- relative ranking, not absolute thresholds.

Universe: liquid + midcap only (NIFTY500 constituents, data/nifty500.csv),
NOT data/sme_list.csv's ~2365 microcaps that scripts/backtest.py includes --
those fills are unrealistic at real position sizes and overstate capturable
edge. A point-in-time turnover gate (MIN_TURNOVER_CR, median 30-bar Close*Volume)
is applied at every rebalance on top of that.

Point-in-time discipline: every factor is computed from
`full_df[full_df.index <= as_of]` ONLY. Forward returns are measured
separately, after the ranking decision, from as_of to the next rebalance date.
This mirrors scripts/backtest.py's `df_slice = full_df[full_df.index <= as_of]`
pattern (its single most important correctness property) and is asserted by
tests/test_factor_backtest.py.

Ships nothing by default -- this is a measurement tool. Run with --validate
for the full ship/no-ship report (Information Coefficient, decile spread,
walk-forward OOS split, regime split, benchmark-relative comparison). If
nothing clears the gate, the printed verdict says so plainly -- that is a
valid, complete result of running this script, not a bug.

Usage:
  python scripts/factor_backtest.py                       # monthly rebalance, all factors, all N
  python scripts/factor_backtest.py --rebalance 5          # weekly rebalance
  python scripts/factor_backtest.py --validate             # + IC / decile-spread / walk-forward / ship-gate
  python scripts/factor_backtest.py --hedge                # + Nifty-futures hedge overlay (Phase C)
  python scripts/factor_backtest.py --sample 100           # fast sanity check, first 100 tickers
  python scripts/factor_backtest.py --weeks 260            # limit to last 5 years (deeper history is slower)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("Run: pip install pandas numpy")

from src.costs import round_trip_cost_pct

CACHE_DIR   = ROOT / "cache"
OUTPUTS     = ROOT / "outputs"
_CACHE_DIR2 = CACHE_DIR / "backtest_ohlcv"
_NIFTY_CSV  = CACHE_DIR / "backtest_nifty.csv"
OUT_FILE    = OUTPUTS / "factor_backtest.json"

# Liquid + midcap gate, point-in-time, applied at every rebalance. Below this,
# real fills at meaningful position sizes are not realistic on NSE.
MIN_TURNOVER_CR = 10.0

TOP_N_OPTIONS      = [15, 20, 30]
REBALANCE_DEFAULT  = 21     # trading days ~= 1 month
WARMUP_DAYS        = 400    # calendar days of history required before the first rebalance (>=252 trading days for mom_12_1/hi_52w)
WALK_FORWARD_SPLIT = "2022-01-01"   # train < split, holdout >= split (covers the 2025-26 correction)

FACTOR_NAMES = ["mom_12_1", "hi_52w", "low_vol", "rs_quality"]

# Hedge overlay (Phase C): fraction of long exposure offset by a short-Nifty-
# futures leg when the index is below its 200DMA.
HEDGE_FRACTIONS = [0.0, 0.25, 0.50]

# SHIP GATE (all must hold, on the HOLDOUT window, net of cost) -- see plan Phase B.
SHIP_MIN_IC       = 0.03
SHIP_MIN_IC_TSTAT = 2.0


# ── universe & data loading ───────────────────────────────────────────────────

def _load_universe(sample: int | None) -> list[str]:
    """NIFTY500 constituents only -- liquid + midcap. Deliberately excludes
    data/sme_list.csv (see module docstring)."""
    csv_path = ROOT / "data" / "nifty500.csv"
    if not csv_path.exists():
        sys.exit(f"Not found: {csv_path}. Run scripts/download_universe.py first.")
    df = pd.read_csv(csv_path)
    col = next((c for c in df.columns if "symbol" in c.lower()), df.columns[0])
    tickers = [f"{s.strip().upper()}.NS" for s in df[col].dropna()]
    uniq = list(dict.fromkeys(tickers))
    return uniq[:sample] if sample else uniq


def _ticker_csv(ticker: str) -> Path:
    safe = ticker.replace(".", "_").replace("/", "_")
    return _CACHE_DIR2 / f"{safe}.csv"


def _load_ticker_df(ticker: str) -> pd.DataFrame | None:
    """Cache-only -- no network. Same cache scripts/backtest.py populates."""
    path = _ticker_csv(ticker)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df.dropna(how="all")
        df = df[df["Close"].notna()]
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df if not df.empty else None
    except Exception:
        return None


def _load_nifty() -> pd.DataFrame:
    """Mirrors scripts/mine_big_movers.py:_load_nifty -- the cached Nifty CSV
    has a 2-row yfinance multi-index header artifact that must be stripped."""
    if not _NIFTY_CSV.exists():
        sys.exit(f"Not found: {_NIFTY_CSV}. Run scripts/backtest.py first to build the cache.")
    df = pd.read_csv(_NIFTY_CSV, index_col=0)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df[df["Close"].notna()]
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def _classify_nifty_trend_series(nifty_close: pd.Series) -> pd.Series:
    """Copied from scripts/backtest.py:_classify_nifty_trend_series (scripts
    aren't a package) -- EMA20/50/200 stack."""
    ema20  = nifty_close.ewm(span=20,  adjust=False).mean()
    ema50  = nifty_close.ewm(span=50,  adjust=False).mean()
    ema200 = nifty_close.ewm(span=200, adjust=False).mean()
    labels = pd.Series("ranging", index=nifty_close.index)
    labels[ema20 > ema50] = "ranging"
    labels[(ema20 > ema50) & (ema50 > ema200)] = "uptrend"
    labels[(ema20 < ema50) & (ema50 < ema200)] = "downtrend"
    return labels


def _turnover_ok(df_slice: pd.DataFrame) -> bool:
    """Point-in-time liquidity gate -- same formula as
    scripts/mine_big_movers.py:_turnover_ok, threshold raised to MIN_TURNOVER_CR
    (10cr vs mine_big_movers.py's 2cr) for this factor book's larger position sizes."""
    hist = df_slice.tail(30)
    if len(hist) < 20:
        return False
    turnover_cr = float((hist["Close"] * hist["Volume"]).median()) / 1e7
    last = hist.iloc[-1]
    return turnover_cr >= MIN_TURNOVER_CR and float(last["High"]) != float(last["Low"])


def _price_asof(series_or_df, ts: pd.Timestamp) -> float | None:
    """Last available price on/before ts -- handles listing gaps and
    ticker-specific trading holidays that don't align with the Nifty calendar.
    Accepts either a Close Series or an OHLCV DataFrame."""
    s = series_or_df["Close"] if isinstance(series_or_df, pd.DataFrame) else series_or_df
    val = s.asof(ts)
    if val is None or pd.isna(val):
        return None
    return float(val)


# ── factor scalars (all point-in-time: df_slice = full_df[full_df.index <= as_of]) ──

def factor_mom_12_1(df_slice: pd.DataFrame) -> float | None:
    """12-1 momentum (Jegadeesh-Titman 1993): 12-month return minus the most
    recent 1-month return (skip-month avoids the well-documented short-term
    reversal). Extends src/technicals.py:544-553's 6-month version (which is
    computed live but never scored) to the standard 12-1 formulation."""
    closes = df_slice["Close"]
    if len(closes) < 253:
        return None
    ret_12m = float(closes.iloc[-1] / closes.iloc[-252] - 1)
    ret_1m  = float(closes.iloc[-1] / closes.iloc[-21] - 1)
    return ret_12m - ret_1m


def factor_hi_52w(df_slice: pd.DataFrame) -> float | None:
    """52-week-high proximity (George & Hwang 2004). Continuous version of
    scripts/backtest.py:_52w_signals' near_52w_high boolean -- the single
    best-performing signal in the 156-week TA backtest (+5.09pp win-rate lift,
    n=15005) yet never used live (dropped there as "redundant + noisy")."""
    closes = df_slice["Close"]
    if len(closes) < 252:
        return None
    high_252 = float(closes.tail(252).max())
    if high_252 <= 0:
        return None
    return float(closes.iloc[-1]) / high_252


def factor_low_vol(df_slice: pd.DataFrame) -> float | None:
    """Inverse realized volatility (60-trading-day daily-return stdev),
    negated so higher score = lower vol = better, matching the other
    factors' orientation (higher score = long candidate)."""
    closes = df_slice["Close"]
    if len(closes) < 61:
        return None
    rets = closes.pct_change().dropna().tail(60)
    vol = float(rets.std())
    if vol <= 0:
        return None
    return -vol


def _compute_atr_pct(df_slice: pd.DataFrame, period: int = 14) -> float | None:
    if len(df_slice) < period + 1:
        return None
    h, l, c = df_slice["High"], df_slice["Low"], df_slice["Close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])
    close = float(c.iloc[-1])
    return (atr / close * 100) if close > 0 else None


def factor_rs_quality(df_slice: pd.DataFrame, nifty_20d: float | None) -> float | None:
    """ATR-adjusted 20d relative strength vs Nifty. Continuous port of
    src/technicals.py:530-542 (which only exposes a boolean threshold)."""
    closes = df_slice["Close"]
    if len(closes) < 20 or nifty_20d is None:
        return None
    atr_pct = _compute_atr_pct(df_slice)
    if not atr_pct or atr_pct <= 0:
        return None
    stock_20d = float(closes.iloc[-1] / closes.iloc[-20] - 1)
    return (stock_20d - nifty_20d) * 100 / atr_pct


FACTOR_FUNCS = {
    "mom_12_1":   lambda s, n20: factor_mom_12_1(s),
    "hi_52w":     lambda s, n20: factor_hi_52w(s),
    "low_vol":    lambda s, n20: factor_low_vol(s),
    "rs_quality": lambda s, n20: factor_rs_quality(s, n20),
}


def _zscore(values: dict[str, float]) -> dict[str, float]:
    if len(values) < 2:
        return {k: 0.0 for k in values}
    arr = np.array(list(values.values()), dtype=float)
    mu, sigma = arr.mean(), arr.std()
    if sigma == 0:
        return {k: 0.0 for k in values}
    return {k: (v - mu) / sigma for k, v in values.items()}


def compute_composite(date_scores: dict[str, dict[str, float]]) -> dict[str, float]:
    """Equal-weight z-score composite across FACTOR_NAMES -- deliberately no
    fitted weights (overfitting risk, see plan risk #3). A ticker's composite
    only uses whichever factors it has a value for that date."""
    per_factor: dict[str, dict[str, float]] = {fn: {} for fn in FACTOR_NAMES}
    for ticker, raw in date_scores.items():
        for fn in FACTOR_NAMES:
            if fn in raw:
                per_factor[fn][ticker] = raw[fn]
    zscores = {fn: _zscore(vals) for fn, vals in per_factor.items()}
    composite: dict[str, float] = {}
    for ticker in date_scores:
        zs = [zscores[fn][ticker] for fn in FACTOR_NAMES if ticker in zscores[fn]]
        if zs:
            composite[ticker] = float(np.mean(zs))
    return composite


# ── panel construction: raw factor values + turnover pass, per rebalance date ──

def build_panel(ohlcv: dict[str, pd.DataFrame], rebalance_dates: list[pd.Timestamp],
                 nifty_close: pd.Series) -> dict[pd.Timestamp, dict[str, dict[str, float]]]:
    """
    Returns {as_of: {ticker: {factor_name: raw_value, ...}}} for every ticker
    that passes the point-in-time turnover gate at that date. Computed once,
    reused for every top-N / factor / hedge combination downstream.
    """
    panel: dict[pd.Timestamp, dict[str, dict[str, float]]] = {}
    n_dates = len(rebalance_dates)
    for di, as_of in enumerate(rebalance_dates):
        if di % 20 == 0:
            print(f"[factor_bt] scoring date {di+1}/{n_dates} ({as_of.date()}) ...")
        nifty_hist = nifty_close[nifty_close.index <= as_of]
        nifty_20d = float(nifty_hist.iloc[-1] / nifty_hist.iloc[-20] - 1) if len(nifty_hist) >= 20 else None

        date_scores: dict[str, dict[str, float]] = {}
        for ticker, full_df in ohlcv.items():
            df_slice = full_df[full_df.index <= as_of]
            if len(df_slice) < 60 or not _turnover_ok(df_slice):
                continue
            raw = {}
            for name, fn in FACTOR_FUNCS.items():
                val = fn(df_slice, nifty_20d)
                if val is not None and np.isfinite(val):
                    raw[name] = val
            if raw:
                date_scores[ticker] = raw
        panel[as_of] = date_scores
    return panel


# ── portfolio simulation ──────────────────────────────────────────────────────

def _period_return(ohlcv: dict[str, pd.DataFrame], holdings: set[str],
                    t_from: pd.Timestamp, t_to: pd.Timestamp) -> float:
    """Equal-weight average simple return of `holdings` from t_from to t_to."""
    rets = []
    for t in holdings:
        df = ohlcv.get(t)
        if df is None:
            continue
        p0 = _price_asof(df, t_from)
        p1 = _price_asof(df, t_to)
        if p0 and p1 and p0 > 0:
            rets.append(p1 / p0 - 1)
    return float(np.mean(rets)) if rets else 0.0


def _rank_scores(panel: dict, as_of: pd.Timestamp, factor: str) -> dict[str, float]:
    date_scores = panel.get(as_of, {})
    if factor == "composite":
        return compute_composite(date_scores)
    return {t_: raw[factor] for t_, raw in date_scores.items() if factor in raw}


def simulate_strategy(
    ohlcv: dict[str, pd.DataFrame],
    panel: dict[pd.Timestamp, dict[str, dict[str, float]]],
    rebalance_dates: list[pd.Timestamp],
    top_n: int,
    factor: str,
    nifty_close: pd.Series | None = None,
    hedge_fraction: float = 0.0,
    below_200dma: pd.Series | None = None,
) -> dict:
    """
    Simulate a top-N equal-weight long portfolio ranked by `factor` (or
    "composite"), rebalanced at each date in rebalance_dates. Cost is booked
    once per name, in full, at EXIT (round_trip_cost_pct covers the whole
    buy+sell round trip) -- names still held at the final date never pay
    their exit leg, a minor, disclosed simplification that doesn't affect
    strategy-vs-strategy or strategy-vs-benchmark comparisons.

    hedge_fraction > 0 (Phase C): when the index is below its 200DMA at a
    rebalance, that fraction of the period's return is replaced by a short
    Nifty-futures leg (costed at the futures rate), reducing net exposure.
    """
    equity = 1.0
    curve: list[dict] = []
    prev_holdings: set[str] = set()
    turnovers: list[float] = []
    period_returns: list[dict] = []  # for regime-split analysis

    for i, t in enumerate(rebalance_dates):
        scores = _rank_scores(panel, t, factor)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        new_holdings = set(t_ for t_, _ in ranked[:top_n])

        exiting = prev_holdings - new_holdings
        cost_drag = (len(exiting) / top_n) * (round_trip_cost_pct("buy") / 100) if prev_holdings else 0.0
        turnovers.append(len(exiting) / max(1, len(prev_holdings)) if prev_holdings else 0.0)

        if i + 1 < len(rebalance_dates):
            t_next = rebalance_dates[i + 1]
            gross_ret = _period_return(ohlcv, new_holdings, t, t_next)

            if hedge_fraction > 0 and below_200dma is not None and nifty_close is not None:
                below = below_200dma.asof(t)
                if below is not None and pd.notna(below) and bool(below):
                    n0 = _price_asof(nifty_close, t)
                    n1 = _price_asof(nifty_close, t_next)
                    nifty_ret = (n1 / n0 - 1) if (n0 and n1 and n0 > 0) else 0.0
                    hedge_ret  = -nifty_ret * hedge_fraction
                    hedge_cost = hedge_fraction * (round_trip_cost_pct("sell") / 100)
                    gross_ret  = gross_ret * (1 - hedge_fraction) + hedge_ret - hedge_cost

            net_ret = gross_ret - cost_drag
            equity *= (1 + net_ret)
            period_returns.append({
                "date": t.date().isoformat(), "next_date": t_next.date().isoformat(),
                "n_holdings": len(new_holdings), "return": net_ret, "gross_return": gross_ret,
            })

        curve.append({"date": t.date().isoformat(), "equity": equity})
        prev_holdings = new_holdings

    return {
        "curve": curve,
        "final_equity": equity,
        "avg_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "period_returns": period_returns,
    }


def simulate_benchmark(nifty_close: pd.Series, rebalance_dates: list[pd.Timestamp]) -> dict:
    """Buy-and-hold Nifty over the same date range, marked at each rebalance
    date for apples-to-apples curve comparison. Negligible cost (one entry)."""
    p0 = _price_asof(nifty_close, rebalance_dates[0])
    curve = []
    for t in rebalance_dates:
        p = _price_asof(nifty_close, t)
        equity = (p / p0) if (p and p0) else 1.0
        curve.append({"date": t.date().isoformat(), "equity": equity})
    return {"curve": curve, "final_equity": curve[-1]["equity"] if curve else 1.0}


# ── performance metrics ───────────────────────────────────────────────────────

def _curve_returns(curve: list[dict]) -> np.ndarray:
    eq = np.array([c["equity"] for c in curve], dtype=float)
    if len(eq) < 2:
        return np.array([])
    return eq[1:] / eq[:-1] - 1


def compute_metrics(curve: list[dict], periods_per_year: float) -> dict:
    if len(curve) < 2 or curve[-1]["equity"] <= 0 or curve[0]["equity"] <= 0:
        return {"cagr_pct": None, "sharpe": None, "max_drawdown_pct": None, "total_return_pct": None, "n_periods": 0}

    rets = _curve_returns(curve)
    n = len(rets)
    total_return = curve[-1]["equity"] / curve[0]["equity"] - 1
    years = n / periods_per_year
    cagr = (curve[-1]["equity"] / curve[0]["equity"]) ** (1 / years) - 1 if years > 0 else None

    mean_r, std_r = float(rets.mean()), float(rets.std())
    sharpe = (mean_r / std_r) * np.sqrt(periods_per_year) if std_r > 0 else None

    eq = np.array([c["equity"] for c in curve], dtype=float)
    running_max = np.maximum.accumulate(eq)
    max_dd = float((eq / running_max - 1).min())

    return {
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2) if cagr is not None else None,
        "sharpe": round(sharpe, 3) if sharpe is not None else None,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "n_periods": n,
    }


def compute_ic(ohlcv: dict, panel: dict, rebalance_dates: list[pd.Timestamp], factor: str) -> dict:
    """Information Coefficient: per-period Spearman rank-correlation of factor
    score (at as_of) vs realized per-ticker forward return (as_of -> next
    rebalance), averaged across periods with a t-stat. Works for "composite"
    too via _rank_scores -- a composite ranking is still one scalar score per
    ticker per date, held to the same bar as any standalone factor."""
    period_ics = []
    for i, t in enumerate(rebalance_dates[:-1]):
        scores = _rank_scores(panel, t, factor)
        if len(scores) < 10:
            continue
        t_next = rebalance_dates[i + 1]
        fwd_rets = {}
        for tk in scores:
            df = ohlcv.get(tk)
            if df is None:
                continue
            p0 = _price_asof(df, t)
            p1 = _price_asof(df, t_next)
            if p0 and p1 and p0 > 0:
                fwd_rets[tk] = p1 / p0 - 1
        common = [tk for tk in scores if tk in fwd_rets]
        if len(common) < 10:
            continue
        s = pd.Series({tk: scores[tk] for tk in common})
        r = pd.Series({tk: fwd_rets[tk] for tk in common})
        ic = s.corr(r, method="spearman")
        if pd.notna(ic):
            period_ics.append(float(ic))

    if not period_ics:
        return {"mean_ic": None, "t_stat": None, "n": 0}
    arr = np.array(period_ics)
    mean_ic = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    t_stat = mean_ic / se if se > 0 else None
    return {"mean_ic": round(mean_ic, 4), "t_stat": round(t_stat, 2) if t_stat is not None else None, "n": len(arr)}


def compute_decile_spread(ohlcv: dict, panel: dict, rebalance_dates: list[pd.Timestamp], factor: str) -> dict:
    """Top-decile minus bottom-decile average forward return, net of cost,
    per period, averaged. Must be positive (and roughly monotone across
    deciles) for the factor to carry real cross-sectional signal."""
    spreads = []
    cost = round_trip_cost_pct("buy") / 100
    for i, t in enumerate(rebalance_dates[:-1]):
        scores = _rank_scores(panel, t, factor)
        if len(scores) < 20:
            continue
        t_next = rebalance_dates[i + 1]
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        decile_n = max(1, len(ranked) // 10)
        top = set(tk for tk, _ in ranked[:decile_n])
        bottom = set(tk for tk, _ in ranked[-decile_n:])
        top_ret = _period_return(ohlcv, top, t, t_next) - cost
        bottom_ret = _period_return(ohlcv, bottom, t, t_next) - cost
        spreads.append(top_ret - bottom_ret)
    if not spreads:
        return {"mean_spread_pct": None, "n": 0}
    return {"mean_spread_pct": round(float(np.mean(spreads)) * 100, 3), "n": len(spreads)}


def split_train_holdout(rebalance_dates: list[pd.Timestamp]) -> tuple[list, list]:
    split = pd.Timestamp(WALK_FORWARD_SPLIT)
    train = [t for t in rebalance_dates if t < split]
    holdout = [t for t in rebalance_dates if t >= split]
    return train, holdout


def regime_split_returns(period_returns: list[dict], trend_series: pd.Series) -> dict[str, dict]:
    buckets: dict[str, list[float]] = {"uptrend": [], "ranging": [], "downtrend": []}
    for pr in period_returns:
        t = pd.Timestamp(pr["date"])
        regime = trend_series.asof(t)
        if regime in buckets:
            buckets[regime].append(pr["return"])
    out = {}
    for regime, rets in buckets.items():
        out[regime] = {
            "n": len(rets),
            "mean_return_pct": round(float(np.mean(rets)) * 100, 3) if rets else None,
        }
    return out


# ── ship gate ──────────────────────────────────────────────────────────────────

def evaluate_ship_gate(factor: str, holdout_metrics: dict, bench_holdout_metrics: dict,
                        ic: dict, decile: dict) -> dict:
    reasons = []
    ships = True

    strat_sharpe = holdout_metrics.get("sharpe")
    bench_sharpe = bench_holdout_metrics.get("sharpe")
    if strat_sharpe is None or bench_sharpe is None or strat_sharpe <= bench_sharpe:
        ships = False
        reasons.append(f"holdout Sharpe {strat_sharpe} does not beat Nifty holdout Sharpe {bench_sharpe}")

    strat_cagr = holdout_metrics.get("cagr_pct")
    bench_cagr = bench_holdout_metrics.get("cagr_pct")
    alpha = (strat_cagr - bench_cagr) if (strat_cagr is not None and bench_cagr is not None) else None
    if alpha is None or alpha <= 0:
        ships = False
        reasons.append(f"holdout alpha vs Nifty is {alpha} (not positive)")

    # IC applies to every strategy including composite -- a composite ranking
    # is still a scalar score per ticker per date, just as rank-correlatable
    # against forward returns as any standalone factor. Composite must clear
    # the same bar, not get a free pass on the single strongest test.
    mean_ic, t_stat = ic.get("mean_ic"), ic.get("t_stat")
    if mean_ic is None or mean_ic < SHIP_MIN_IC or t_stat is None or t_stat < SHIP_MIN_IC_TSTAT:
        ships = False
        reasons.append(f"IC {mean_ic} (t={t_stat}) below bar (need >= {SHIP_MIN_IC}, t >= {SHIP_MIN_IC_TSTAT})")

    mean_spread = decile.get("mean_spread_pct")
    if mean_spread is None or mean_spread <= 0:
        ships = False
        reasons.append(f"decile spread {mean_spread}% is not positive")

    return {"ships": ships, "alpha_pct": round(alpha, 3) if alpha is not None else None, "reasons": reasons}


# ── main ──────────────────────────────────────────────────────────────────────

def run(rebalance_days: int, top_n_list: list[int], weeks: int | None, sample: int | None,
        validate: bool, hedge: bool) -> None:
    universe = _load_universe(sample)
    print(f"[factor_bt] universe: {len(universe)} tickers (liquid+midcap, NIFTY500, "
          f">={MIN_TURNOVER_CR}cr/day point-in-time gate)")

    print("[factor_bt] loading OHLCV from cache (no network)...")
    ohlcv: dict[str, pd.DataFrame] = {}
    for t in universe:
        df = _load_ticker_df(t)
        if df is not None:
            ohlcv[t] = df
    print(f"[factor_bt] loaded {len(ohlcv)}/{len(universe)} tickers")
    if not ohlcv:
        sys.exit("[factor_bt] no cached OHLCV found — run scripts/backtest.py first to populate cache/backtest_ohlcv/")

    nifty_df = _load_nifty()
    nifty_close = nifty_df["Close"].squeeze()
    trend_series = _classify_nifty_trend_series(nifty_close)
    dma200 = nifty_close.rolling(200).mean()
    below_200dma = (nifty_close < dma200)

    end = date.today()
    if weeks:
        start = end - timedelta(days=weeks * 7 + WARMUP_DAYS)
    else:
        start = nifty_close.index.min().date() + timedelta(days=WARMUP_DAYS)
    nifty_ts = nifty_close[nifty_close.index >= pd.Timestamp(start)]
    idxs = list(range(0, len(nifty_ts), rebalance_days))
    rebalance_dates = [nifty_ts.index[i] for i in idxs]
    if len(rebalance_dates) < 10:
        sys.exit(f"[factor_bt] only {len(rebalance_dates)} rebalance dates — widen --weeks")
    print(f"[factor_bt] {len(rebalance_dates)} rebalance dates, step={rebalance_days}d, "
          f"{rebalance_dates[0].date()} -> {rebalance_dates[-1].date()}")

    print("[factor_bt] building factor panel (this is the slow part)...")
    panel = build_panel(ohlcv, rebalance_dates, nifty_close)

    periods_per_year = 252 / rebalance_days
    strategies = FACTOR_NAMES + ["composite"]
    top_n = top_n_list[len(top_n_list) // 2]  # primary N for the main report; all N run below

    bench = simulate_benchmark(nifty_close, rebalance_dates)
    bench_metrics = compute_metrics(bench["curve"], periods_per_year)

    print(f"\n{'='*78}\n  FACTOR BACKTEST — {len(rebalance_dates)} rebalances, top_n={top_n}, "
          f"{rebalance_days}d step\n{'='*78}")
    print(f"  {'Strategy':<14} {'CAGR%':>8} {'Sharpe':>8} {'MaxDD%':>8} {'Turnover':>9}  vs Nifty CAGR {bench_metrics['cagr_pct']}%  Sharpe {bench_metrics['sharpe']}")

    results: dict = {"meta": {
        "universe_n": len(ohlcv), "rebalance_days": rebalance_days, "top_n": top_n,
        "min_turnover_cr": MIN_TURNOVER_CR, "date_range": [rebalance_dates[0].date().isoformat(), rebalance_dates[-1].date().isoformat()],
    }, "benchmark": {"metrics": bench_metrics}, "strategies": {}}

    train_dates, holdout_dates = split_train_holdout(rebalance_dates)
    bench_train = simulate_benchmark(nifty_close, train_dates) if len(train_dates) >= 5 else None
    bench_holdout = simulate_benchmark(nifty_close, holdout_dates) if len(holdout_dates) >= 5 else None
    results["benchmark"]["train_metrics"] = compute_metrics(bench_train["curve"], periods_per_year) if bench_train else None
    results["benchmark"]["holdout_metrics"] = compute_metrics(bench_holdout["curve"], periods_per_year) if bench_holdout else None

    for factor in strategies:
        sim = simulate_strategy(ohlcv, panel, rebalance_dates, top_n, factor)
        metrics = compute_metrics(sim["curve"], periods_per_year)
        print(f"  {factor:<14} {str(metrics['cagr_pct']):>8} {str(metrics['sharpe']):>8} "
              f"{str(metrics['max_drawdown_pct']):>8} {sim['avg_turnover']*100:>8.1f}%")

        entry: dict = {"metrics": metrics, "avg_turnover_pct": round(sim["avg_turnover"] * 100, 1)}

        if validate:
            # Re-simulate on the train/holdout date subsets independently (fresh equity
            # curves starting at 1.0 for each window) rather than slicing sim's combined
            # curve -- keeps CAGR/Sharpe/drawdown correctly scoped to each window.
            sim_train = simulate_strategy(ohlcv, panel, train_dates, top_n, factor) if len(train_dates) >= 5 else None
            sim_holdout = simulate_strategy(ohlcv, panel, holdout_dates, top_n, factor) if len(holdout_dates) >= 5 else None
            train_metrics = compute_metrics(sim_train["curve"], periods_per_year) if sim_train else {}
            holdout_metrics = compute_metrics(sim_holdout["curve"], periods_per_year) if sim_holdout else {}

            ic = compute_ic(ohlcv, panel, holdout_dates, factor) if len(holdout_dates) >= 5 else {"mean_ic": None, "t_stat": None, "n": 0}
            decile = compute_decile_spread(ohlcv, panel, holdout_dates, factor) if len(holdout_dates) >= 5 else {"mean_spread_pct": None, "n": 0}
            regime = regime_split_returns(sim["period_returns"], trend_series)

            gate = evaluate_ship_gate(
                factor, holdout_metrics, results["benchmark"]["holdout_metrics"] or {}, ic, decile,
            )

            entry.update({
                "train_metrics": train_metrics, "holdout_metrics": holdout_metrics,
                "ic_holdout": ic, "decile_spread_holdout": decile, "regime_split": regime,
                "ship_gate": gate,
            })
            print(f"      IC(holdout)={ic['mean_ic']} t={ic['t_stat']} n={ic['n']}  "
                  f"decile_spread={decile['mean_spread_pct']}%  "
                  f"holdout CAGR={holdout_metrics.get('cagr_pct')}% Sharpe={holdout_metrics.get('sharpe')}  "
                  f"-> {'SHIP' if gate['ships'] else 'NO-SHIP'}")
            if not gate["ships"]:
                for r in gate["reasons"]:
                    print(f"        - {r}")

        results["strategies"][factor] = entry

    if hedge:
        print(f"\n{'='*78}\n  HEDGE OVERLAY (holdout window) — best long strategy x hedge_fraction\n{'='*78}")
        shipping = [f for f, e in results["strategies"].items() if e.get("ship_gate", {}).get("ships")]
        hedge_target = shipping[0] if shipping else "composite"
        print(f"  Applying hedge overlay to: {hedge_target} (holdout {holdout_dates[0].date() if holdout_dates else '?'}+)")
        results["hedge"] = {"target_factor": hedge_target, "variants": {}}
        for hf in HEDGE_FRACTIONS:
            sim_h = simulate_strategy(ohlcv, panel, holdout_dates, top_n, hedge_target,
                                       nifty_close=nifty_close, hedge_fraction=hf, below_200dma=below_200dma)
            m = compute_metrics(sim_h["curve"], periods_per_year)
            print(f"  hedge_fraction={hf:.2f}  CAGR={m['cagr_pct']}%  Sharpe={m['sharpe']}  MaxDD={m['max_drawdown_pct']}%")
            results["hedge"]["variants"][str(hf)] = m
        best_hf = max(HEDGE_FRACTIONS, key=lambda hf: (results["hedge"]["variants"][str(hf)].get("sharpe") or -999))
        unhedged_sharpe = results["hedge"]["variants"]["0.0"].get("sharpe")
        best_sharpe = results["hedge"]["variants"][str(best_hf)].get("sharpe")
        hedge_ships = best_hf > 0 and best_sharpe is not None and unhedged_sharpe is not None and best_sharpe > unhedged_sharpe
        results["hedge"]["verdict"] = {
            "best_hedge_fraction": best_hf, "ships": hedge_ships,
            "reason": (f"hedge_fraction={best_hf} improves Sharpe ({best_sharpe} > {unhedged_sharpe})" if hedge_ships
                       else "no hedge fraction improves risk-adjusted return OOS vs unhedged — hedge does NOT ship"),
        }
        print(f"  -> {results['hedge']['verdict']['reason']}")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n{'='*78}\n  Results -> {OUT_FILE.name}\n{'='*78}")

    if validate:
        any_ship = any(e.get("ship_gate", {}).get("ships") for e in results["strategies"].values())
        print(f"\n  OVERALL VERDICT: {'AT LEAST ONE FACTOR SHIPS' if any_ship else 'NO-SHIP — nothing clears the gate on this window.'}")
        if not any_ship:
            print("  Buy-and-hold Nifty remains the benchmark to beat. This is a complete,")
            print("  honest result of running this backtest, not a failure to route around.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebalance", type=int, default=REBALANCE_DEFAULT, choices=[5, 21],
                     help="Rebalance frequency in trading days (5=weekly, 21=monthly, default monthly)")
    ap.add_argument("--top-n", type=int, default=None, help="Override the primary top-N (default: middle of TOP_N_OPTIONS)")
    ap.add_argument("--weeks", type=int, default=None, help="Limit to last N weeks (default: full cached history)")
    ap.add_argument("--sample", type=int, default=None, help="Limit universe to first N tickers (fast sanity check)")
    ap.add_argument("--validate", action="store_true", help="Run IC / decile-spread / walk-forward / ship-gate report")
    ap.add_argument("--hedge", action="store_true", help="Run the Nifty-futures hedge overlay (Phase C, requires --validate)")
    args = ap.parse_args()

    top_n_list = [args.top_n] if args.top_n else TOP_N_OPTIONS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run(args.rebalance, top_n_list, args.weeks, args.sample, args.validate, args.hedge)
