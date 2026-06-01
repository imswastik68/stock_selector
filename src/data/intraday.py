"""
Mid-day intraday volume check for watchlist candidates.

Runs at 12:30 PM IST on yesterday's top candidates (20 tickers max).
Downloads 5m OHLCV only for those tickers — not the full 2369-ticker universe.

Signal: intraday_surge = projected full-day volume >= 2.5x 30d avg
                         AND current price > yesterday's close
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone, timedelta

import pandas as pd
import yfinance as yf

_NSE_OPEN_MINUTE  = 9 * 60 + 15   # 9:15 AM IST
_NSE_CLOSE_MINUTE = 15 * 60 + 30  # 3:30 PM IST
_SESSION_MINUTES  = _NSE_CLOSE_MINUTE - _NSE_OPEN_MINUTE  # 375 min

_SURGE_THRESHOLD  = 2.5   # projected volume vs 30d avg
_MIN_ELAPSED_MIN  = 30    # don't run within first 30 min (opening volatility)


def _ist_elapsed_minutes() -> int:
    """Minutes elapsed since NSE open (9:15 AM IST). Capped at session length."""
    IST = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(IST)
    current = now.hour * 60 + now.minute
    return max(0, min(current - _NSE_OPEN_MINUTE, _SESSION_MINUTES))


def _extract_ticker_df(raw: pd.DataFrame, ticker: str, n_tickers: int) -> pd.DataFrame:
    """Pull per-ticker DataFrame from a multi or single-ticker yfinance result."""
    if isinstance(raw.columns, pd.MultiIndex):
        if ticker not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        return raw[ticker].dropna(how="all")
    return raw.dropna(how="all") if n_tickers == 1 else pd.DataFrame()


def fetch_intraday_signals(tickers: list[str]) -> dict[str, dict]:
    """
    For each ticker fetch today's 5m bars + 35d daily OHLCV.
    Returns {ticker: {
        volume_so_far, projected_volume, volume_ratio_projected,
        price_current, prev_close, price_above_prev_close, intraday_surge
    }}.
    """
    if not tickers:
        return {}

    elapsed = _ist_elapsed_minutes()
    if elapsed < _MIN_ELAPSED_MIN:
        print(f"[intraday] only {elapsed} min into session — skipping (need {_MIN_ELAPSED_MIN}+)")
        return {}

    fraction = elapsed / _SESSION_MINUTES  # fraction of session elapsed

    print(f"[intraday] {elapsed} min elapsed ({fraction*100:.0f}% of session) — "
          f"checking {len(tickers)} candidates...")

    # 5m bars for today
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df5 = yf.download(
                tickers, period="1d", interval="5m",
                auto_adjust=True, progress=False, group_by="ticker",
            )
    except Exception as exc:
        print(f"[intraday] 5m download error: {exc}")
        return {}

    # Daily bars for 30d volume avg + yesterday's close
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dfd = yf.download(
                tickers, period="35d", interval="1d",
                auto_adjust=True, progress=False, group_by="ticker",
            )
    except Exception as exc:
        print(f"[intraday] daily download error: {exc}")
        return {}

    results: dict[str, dict] = {}
    n = len(tickers)

    for ticker in tickers:
        try:
            t5  = _extract_ticker_df(df5, ticker, n)
            td  = _extract_ticker_df(dfd, ticker, n)

            if t5.empty or td.empty or len(td) < 2:
                continue

            volume_so_far     = int(t5["Volume"].sum())
            projected_volume  = volume_so_far / fraction if fraction > 0 else 0
            avg_30d           = float(td["Volume"].iloc[:-1].tail(30).mean())
            vol_ratio_proj    = round(projected_volume / avg_30d, 2) if avg_30d > 0 else 0

            price_current = float(t5["Close"].iloc[-1])
            prev_close    = float(td["Close"].iloc[-2])   # yesterday's close
            price_above   = price_current > prev_close
            pct_vs_prev   = round((price_current / prev_close - 1) * 100, 2) if prev_close else 0

            results[ticker] = {
                "volume_so_far":          volume_so_far,
                "projected_volume":       round(projected_volume),
                "volume_ratio_projected": vol_ratio_proj,
                "price_current":          round(price_current, 2),
                "prev_close":             round(prev_close, 2),
                "pct_vs_prev_close":      pct_vs_prev,
                "price_above_prev_close": price_above,
                "intraday_surge":         vol_ratio_proj >= _SURGE_THRESHOLD and price_above,
            }
        except Exception:
            continue

    confirmed = sum(1 for v in results.values() if v["intraday_surge"])
    print(f"[intraday] {len(results)} tickers checked, {confirmed} with intraday_surge "
          f"(proj >= {_SURGE_THRESHOLD}x AND price > prev close)")
    return results
