"""
Short-horizon mean-reversion scanner (reversal_oversold_v2) -- SOTA Round
Phase 2. Deliberately the OPPOSITE bet to the trend/momentum cluster (buys
sharp oversold dips, not strength) -- a genuinely orthogonal edge, not
another way of saying "price is going up."

Backtested SHIP verdict (scripts/backtest_events.py:reversal_v2_verdict,
outputs/event_backtest.json): n=6843, ret_lift=+2.102 -- the single
strongest measured lift in this codebase. 5/5 chronological splits
positive (incl. the most recent), survives 2x transaction costs, and BOTH
exit models agree (standard ATR-template +2.102, 5-day time-exit +0.892).

Pre-declared condition (module constants below are the SINGLE SOURCE OF
TRUTH -- scripts/backtest_events.py imports them, never hardcodes its own
copy, so live and backtest can't silently drift): 3-day return <=
RET_3D_THRESHOLD AND RSI(2) < RSI2_THRESHOLD. Deliberately NO 200DMA filter
-- the with-filter primary (Alpha Round) failed its own 70/30 holdout;
only this unfiltered variant, re-tested under a stricter multi-split gate,
shipped. See tests/test_reversal.py's parity test against
collect_reversal_diag_events.
"""

from __future__ import annotations

import logging
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

from src.technicals import _rsi_series

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
NIFTY500_CSV = DATA_DIR / "nifty500.csv"

RET_3D_THRESHOLD = -0.07   # 3-day return <= -7%
RSI2_THRESHOLD = 10.0      # RSI(2) < 10
RSI2_PERIOD = 2
# Mirrors scripts/mine_big_movers.py's MIN_TURNOVER_CR exactly -- that's the
# liquidity gate the backtest applied (via _turnover_ok) to every evaluated
# event. src/ cannot import from scripts/ (repo convention), so this is a
# deliberate, commented duplication -- keep the two values in sync by hand.
MIN_TURNOVER_CR = 2.0
BATCH_SIZE = 100


def _load_universe() -> list[str]:
    """NIFTY500 only -- matches scripts/factor_backtest.py's _load_universe,
    the exact universe reversal_oversold_v2 was backtested against (NOT
    src.data.breakouts's broader NIFTY500+SME live universe)."""
    tickers: list[str] = []
    if NIFTY500_CSV.exists():
        df = pd.read_csv(NIFTY500_CSV)
        col = next((c for c in df.columns if "symbol" in c.lower()), df.columns[0])
        tickers = [f"{s.strip().upper()}.NS" for s in df[col].dropna()]
    if not tickers:
        tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
    return list(dict.fromkeys(tickers))


def _fires(close: pd.Series) -> tuple[bool, float | None, float | None]:
    """The pre-declared condition, evaluated on the LAST bar of `close` only
    -- shared by both the live per-ticker check and (indirectly, via the
    module constants) the backtest collector. Returns (fires, ret_3d, rsi2)."""
    if len(close) < 4:
        return False, None, None
    ret_3d = float(close.iloc[-1] / close.iloc[-4] - 1)
    rsi2 = float(_rsi_series(close, period=RSI2_PERIOD).iloc[-1])
    fires = ret_3d <= RET_3D_THRESHOLD and rsi2 < RSI2_THRESHOLD
    return fires, ret_3d, rsi2


def _turnover_ok(df: pd.DataFrame) -> bool:
    """Mirrors scripts/mine_big_movers.py's _turnover_ok exactly (median
    close*volume over the trailing 30 bars >= MIN_TURNOVER_CR, and today's
    bar isn't a zero-range halt/no-trade print)."""
    hist = df.tail(30)
    if len(hist) < 10:
        return False
    turnover_cr = float((hist["Close"] * hist["Volume"]).median()) / 1e7
    last = hist.iloc[-1]
    return turnover_cr >= MIN_TURNOVER_CR and float(last["High"]) != float(last["Low"])


def _parse_batch(df: pd.DataFrame, tickers: list[str]) -> list[dict]:
    results = []
    for ticker in tickers:
        try:
            if isinstance(df.columns, pd.MultiIndex):
                if ticker not in df.columns.get_level_values(0):
                    continue
                t_df = df.xs(ticker, axis=1, level=0).dropna(how="all")
            else:
                t_df = df.dropna(how="all")

            if t_df.empty or len(t_df) < 30:
                continue

            close = t_df["Close"].dropna()
            if len(close) < 30:
                continue

            fires, ret_3d, rsi2 = _fires(close)
            if not fires:
                continue
            if not _turnover_ok(t_df):
                continue

            results.append({
                "ticker": ticker,
                "ret_3d_pct": round(ret_3d * 100, 2),
                "rsi2": rsi2,
                "today_close": round(float(close.iloc[-1]), 2),
            })
        except Exception:
            continue
    return results


def fetch_reversal_signals() -> list[dict]:
    """
    Return NIFTY500 stocks currently firing the reversal_oversold_v2
    condition. Batch yfinance downloads, same caching pattern as
    src.data.breakouts.fetch_breakouts (same-day cache, pre-market reuses
    yesterday's data).
    """
    import os
    from src.cache import load_today, load_latest, save_today
    cached = load_today("reversal")
    if cached is not None:
        return cached
    if os.environ.get("SCAN_MODE") == "pre_market":
        cached = load_latest("reversal")
        if cached is not None:
            return cached

    universe = _load_universe()
    print(f"[reversal] scanning {len(universe)} tickers in batches of {BATCH_SIZE}...")

    end = date.today()
    start = end - timedelta(days=90)  # 3mo -- RSI(2)+3d return need ~30 bars, no 200DMA here

    results = []
    batches = [universe[i:i + BATCH_SIZE] for i in range(0, len(universe), BATCH_SIZE)]

    for i, batch in enumerate(batches):
        print(f"[reversal] batch {i + 1}/{len(batches)} ({len(batch)} tickers)...")
        try:
            df = yf.download(
                batch, start=start, end=end, progress=False,
                auto_adjust=True, group_by="ticker", threads=True,
            )
            results.extend(_parse_batch(df, batch))
        except Exception as exc:
            print(f"[reversal] batch {i + 1} failed: {exc}")

    print(f"[reversal] {len(results)} oversold reversal candidate(s) found")
    save_today("reversal", results)
    return results
