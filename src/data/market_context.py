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
    }

    # Nifty 50 — 200d history for EMA + beta baseline
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nifty_df = yf.download(_NIFTY, period="252d", auto_adjust=True, progress=False)
        if not nifty_df.empty:
            close = nifty_df["Close"].squeeze()
            result["nifty_structure"] = _classify_trend(close)
            result["nifty_returns"] = close.pct_change().dropna()
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
        import pandas_ta as ta  # deferred import — optional dep
        atr_series = ta.atr(df["High"], df["Low"], df["Close"], length=period)
        if atr_series is None or atr_series.empty:
            return float("nan")
        atr_val = float(atr_series.dropna().iloc[-1])
        close_val = float(df["Close"].iloc[-1])
        if close_val <= 0:
            return float("nan")
        return round(atr_val / close_val * 100, 3)
    except Exception:
        # Fallback: manual ATR (Wilder)
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
    Phase 3: download 90d OHLCV, compute beta and ATR% for each candidate ticker.
    Returns base_ctx merged with: ohlcv_90d, beta, atr_pct.
    """
    ctx = dict(base_ctx)  # shallow copy to avoid mutating original
    ctx["ohlcv_90d"] = {}
    ctx["beta"] = {}
    ctx["atr_pct"] = {}

    if not tickers:
        return ctx

    nifty_returns: pd.Series = base_ctx.get("nifty_returns", pd.Series(dtype=float))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                tickers, period="90d", auto_adjust=True, progress=False, group_by="ticker"
            )
        if raw.empty:
            return ctx
    except Exception as exc:
        print(f"[market_context] candidate OHLCV fetch error: {exc}")
        return ctx

    # Normalise: single ticker returns a flat DataFrame; multi-ticker returns MultiIndex
    if len(tickers) == 1:
        ticker = tickers[0]
        raw = {ticker: raw}
    else:
        raw = {t: raw[t] for t in tickers if t in raw.columns.get_level_values(0)}

    for ticker, df in raw.items():
        if df.empty:
            continue
        df = df.dropna(how="all")
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
        ctx["atr_pct"][ticker] = _compute_atr_pct(df)

    print(f"[market_context] enriched {len(ctx['ohlcv_90d'])} candidates with OHLCV/beta/ATR")
    return ctx
