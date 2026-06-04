"""
Market-wide context fetcher: Nifty structure, sector heatmap, beta, ATR.

Two exported functions:
  fetch_market_wide_context()         — runs in Phase 1, parallel with other fetchers
  enrich_candidate_context(tickers, base_ctx) — runs in Phase 3, after scoring
"""

from __future__ import annotations

import warnings

import pandas as pd
import yfinance as yf

SECTOR_INDICES = [
    "^CNXAUTO", "^CNXBANK", "^CNXIT", "^CNXPHARMA", "^CNXFMCG",
    "^CNXMETAL", "^CNXREALTY", "^CNXENERGY", "^CNXINFRA",
    "^CNXPSUBANK", "^CNXMEDIA",
]

_NIFTY = "^NSEI"
_EMA_PERIODS = (20, 50, 200)


def _ema(series: pd.Series, span: int) -> float:
    return series.ewm(span=span, adjust=False).mean().iloc[-1]


def _classify_trend(close: pd.Series) -> dict:
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)
    price = float(close.iloc[-1])

    if ema20 > ema50 > ema200:
        trend = "uptrend"
    elif ema20 < ema50 < ema200:
        trend = "downtrend"
    else:
        trend = "ranging"

    return {
        "trend": trend,
        "ema20": round(ema20, 2),
        "ema50": round(ema50, 2),
        "ema200": round(ema200, 2),
        "current_price": round(price, 2),
    }


def _compute_nifty_regime(nifty_df: pd.DataFrame) -> str:
    """
    Returns 'low_vol', 'normal', or 'high_vol' based on where today's Nifty ATR-14%
    sits within its own 252-day history.
      < 30th percentile → low_vol  (signals more reliable — breakouts are clean)
      > 70th percentile → high_vol (signals noisier — score all candidates -1)
    """
    if len(nifty_df) < 50:
        return "normal"
    high  = nifty_df["High"].squeeze()
    low   = nifty_df["Low"].squeeze()
    close = nifty_df["Close"].squeeze()
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_pct = (tr.ewm(alpha=1 / 14, adjust=False).mean() / close * 100).dropna()
    if atr_pct.empty:
        return "normal"
    current = float(atr_pct.iloc[-1])
    percentile = float((atr_pct < current).mean() * 100)
    if percentile < 30:
        return "low_vol"
    if percentile > 70:
        return "high_vol"
    return "normal"


def fetch_market_wide_context() -> dict:
    """
    Phase 1: fetch Nifty structure + sector heatmap + store Nifty returns for beta.
    Returns dict with keys: nifty_structure, sector_heatmap, nifty_returns.
    Safe to call with no arguments; returns empty defaults on failure.
    """
    result = {
        "nifty_structure": {"trend": "ranging", "ema20": 0, "ema50": 0, "ema200": 0, "current_price": 0},
        "sector_heatmap": {},
        "nifty_returns": pd.Series(dtype=float),
        "nifty_regime": "normal",
    }

    # Nifty 50 — 200d history for EMA + beta baseline
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nifty_df = yf.download(_NIFTY, period="252d", auto_adjust=True, progress=False)
        if not nifty_df.empty:
            close = nifty_df["Close"].squeeze()
            result["nifty_structure"] = _classify_trend(close)
            result["nifty_returns"]   = close.pct_change().dropna()
            result["nifty_regime"]    = _compute_nifty_regime(nifty_df)
    except Exception as exc:
        print(f"[market_context] Nifty fetch error: {exc}")

    # Sector heatmap — 5-day % change for 11 sector indices
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sector_raw = yf.download(
                SECTOR_INDICES, period="10d", auto_adjust=True, progress=False
            )
        if not sector_raw.empty:
            # yf returns multi-level columns when downloading multiple tickers
            closes = sector_raw["Close"] if "Close" in sector_raw.columns else sector_raw
            if isinstance(closes, pd.Series):
                closes = closes.to_frame()
            heatmap = {}
            for col in closes.columns:
                series = closes[col].dropna()
                if len(series) >= 2:
                    pct = (series.iloc[-1] / series.iloc[-6] - 1) * 100 if len(series) >= 6 else (series.iloc[-1] / series.iloc[0] - 1) * 100
                    heatmap[str(col)] = round(float(pct), 2)
            result["sector_heatmap"] = heatmap
    except Exception as exc:
        print(f"[market_context] Sector heatmap error: {exc}")

    return result


def _compute_beta(stock_returns: pd.Series, nifty_returns: pd.Series) -> float:
    aligned = pd.concat([stock_returns, nifty_returns], axis=1).dropna()
    if len(aligned) < 30:
        return float("nan")
    cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    var = aligned.iloc[:, 1].var()
    if var == 0:
        return float("nan")
    return round(cov / var, 3)


def _compute_atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    try:
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
        close_val = float(close.iloc[-1])
        if close_val <= 0:
            return float("nan")
        return round(float(atr) / close_val * 100, 3)
    except Exception:
        return float("nan")


def enrich_candidate_context(tickers: list[str], base_ctx: dict) -> dict:
    """
    Phase 3: download 90d OHLCV, compute beta, ATR%, and deterministic technicals
    (RSI, MACD, Wyckoff phase, entry/stop/target levels) for each candidate ticker.
    Returns base_ctx merged with: ohlcv_90d, beta, atr_pct, technicals.
    """
    from src.technicals import enrich_with_technicals  # avoid circular at module level
    ctx = dict(base_ctx)
    ctx["ohlcv_90d"] = {}
    ctx["beta"] = {}
    ctx["atr_pct"] = {}
    ctx["technicals"] = {}

    if not tickers:
        return ctx

    nifty_returns: pd.Series = base_ctx.get("nifty_returns", pd.Series(dtype=float))

    # Nifty 20-day cumulative return — used by enrich_with_technicals for RS signal
    nifty_20d_return: float | None = None
    if not nifty_returns.empty and len(nifty_returns) >= 20:
        nifty_20d_return = float((1 + nifty_returns.iloc[-20:]).prod() - 1)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                tickers, period="200d", auto_adjust=True, progress=False, group_by="ticker"
            )
        if raw.empty:
            return ctx
    except Exception as exc:
        print(f"[market_context] candidate OHLCV fetch error: {exc}")
        return ctx

    # Normalise: yfinance 1.x always returns MultiIndex (ticker, field) with group_by="ticker".
    # Slice per-ticker so every df in raw has flat columns (Open, High, Low, Close, Volume).
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0).unique().tolist()
        raw = {t: raw[t] for t in tickers if t in level0}
    else:
        # Older yfinance: flat columns for single ticker
        if len(tickers) == 1:
            raw = {tickers[0]: raw}
        else:
            raw = {}

    for ticker, df in raw.items():
        if df.empty:
            continue
        df = df.dropna(how="all")
        df = df[df["Close"].notna()]   # drop incomplete rows (market not yet open for the day)
        if df.empty:
            continue

        # Standardise column names (yfinance may use 'Adj Close' or 'Close')
        df = df.rename(columns={"Adj Close": "Close"}) if "Adj Close" in df.columns else df
        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(df.columns):
            continue

        ctx["ohlcv_90d"][ticker] = df

        # Beta
        stock_returns = df["Close"].pct_change().dropna()
        if not nifty_returns.empty:
            ctx["beta"][ticker] = _compute_beta(stock_returns, nifty_returns)

        # ATR%
        atr_pct = _compute_atr_pct(df)
        ctx["atr_pct"][ticker] = atr_pct

        # Deterministic technicals: RSI, MACD, Wyckoff phase, entry/stop/target
        close_val = float(df["Close"].squeeze().iloc[-1])
        atr_abs = atr_pct / 100 * close_val if pd.notna(atr_pct) else 0
        try:
            ctx["technicals"][ticker] = enrich_with_technicals(
                df, close_val, atr_abs, nifty_20d_return=nifty_20d_return
            )
        except Exception as exc:
            print(f"[market_context] technicals error for {ticker}: {exc}")
            ctx["technicals"][ticker] = {}

        # Weekly trend: resample daily → weekly closes, compute EMA10 vs EMA20.
        # Requires ≥20 weekly bars (~100 trading days). With 200d download we get ~28 weeks.
        try:
            weekly = df["Close"].squeeze().resample("W").last().dropna()
            if len(weekly) >= 20:
                w_ema10 = float(weekly.ewm(span=10, adjust=False).mean().iloc[-1])
                w_ema20 = float(weekly.ewm(span=20, adjust=False).mean().iloc[-1])
                ctx["technicals"][ticker]["weekly_trend_aligned"] = w_ema10 > w_ema20
            else:
                ctx["technicals"][ticker]["weekly_trend_aligned"] = False
        except Exception:
            ctx["technicals"][ticker]["weekly_trend_aligned"] = False

    print(f"[market_context] enriched {len(ctx['ohlcv_90d'])} candidates with OHLCV/beta/ATR/technicals")
    return ctx
