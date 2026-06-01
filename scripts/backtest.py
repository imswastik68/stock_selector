"""
Backtest the stock scanner's signal quality.

Strategy: for each trading week over the past 6 months, apply the scorer
to the signals available on Friday close and measure the price change over
the next 5 and 10 trading days vs Nifty 50.

Usage:
    python scripts/backtest.py
    python scripts/backtest.py --weeks 12   # last 12 weeks only

Output: per-week hit table + summary hit rate and average return.
"""

from __future__ import annotations

import argparse
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent.parent

_NIFTY = "^NSEI"
_ARCHIVE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "text/html,*/*",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_universe() -> list[str]:
    tickers: list[str] = []
    for csv_path in [ROOT / "data" / "nifty500.csv", ROOT / "data" / "sme_list.csv"]:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            col = next((c for c in df.columns if "symbol" in c.lower()), df.columns[0])
            tickers += [f"{s.strip().upper()}.NS" for s in df[col].dropna()]
    if not tickers:
        print("[backtest] WARNING: no universe CSVs — using a small seed list")
        tickers = [
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
            "SBIN.NS", "AXISBANK.NS", "WIPRO.NS", "SUNPHARMA.NS", "HCLTECH.NS",
            "MARUTI.NS", "TITAN.NS", "BAJFINANCE.NS", "NESTLEIND.NS", "LT.NS",
            "ASIANPAINT.NS", "ULTRACEMCO.NS", "ITC.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
        ]
    return list(dict.fromkeys(tickers))


def _fetch_history(tickers: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
    """Batch-download OHLCV and return per-ticker DataFrames."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            tickers,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    if raw.empty:
        return {}
    if isinstance(raw.columns, pd.MultiIndex):
        out = {}
        for t in tickers:
            if t in raw.columns.get_level_values(0):
                df = raw[t].dropna(how="all")
                if not df.empty:
                    out[t] = df
        return out
    # Single-ticker case
    if len(tickers) == 1:
        return {tickers[0]: raw.dropna(how="all")} if not raw.empty else {}
    return {}


def _compute_signal_score(df: pd.DataFrame, as_of: pd.Timestamp) -> float:
    """
    Simplified signal score for a stock at a specific date.
    Uses only OHLCV-derivable signals (volume surge + near 52w high + EMA trend).
    Returns float score 0-6.
    """
    hist = df[df.index <= as_of].tail(40)
    if len(hist) < 20:
        return 0.0

    closes = hist["Close"].squeeze()
    volumes = hist["Volume"].squeeze()

    score = 0.0

    # Volume surge: today vs 30d avg
    today_vol = float(volumes.iloc[-1])
    avg_30d = float(volumes.iloc[:-1].mean()) if len(volumes) > 1 else today_vol
    if avg_30d > 0 and today_vol / avg_30d >= 2.5:
        score += 3.0

    # Near 52w high (within 5%)
    high_52w = float(closes.max())
    today_close = float(closes.iloc[-1])
    if high_52w > 0 and (high_52w - today_close) / high_52w <= 0.05:
        score += 2.0

    # EMA trend: EMA20 > EMA50 (bullish structure)
    ema20 = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(closes.ewm(span=50, adjust=False).mean().iloc[-1])
    if ema20 > ema50:
        score += 1.0

    return score


def _forward_return(df: pd.DataFrame, signal_date: pd.Timestamp, days: int) -> float | None:
    """Return % gain from signal_date close to +days trading days later."""
    future = df[df.index > signal_date]
    if len(future) < days:
        return None
    entry = float(df[df.index <= signal_date]["Close"].squeeze().iloc[-1])
    exit_ = float(future.iloc[days - 1]["Close"].squeeze())
    if entry <= 0:
        return None
    return round((exit_ / entry - 1) * 100, 2)


# ── main backtest ─────────────────────────────────────────────────────────────

def run_backtest(weeks: int = 26) -> None:
    today = date.today()
    start = today - timedelta(days=weeks * 7 + 60)  # 60 extra days for history
    end = today

    universe = _load_universe()
    print(f"[backtest] universe: {len(universe)} tickers | period: {start} to {end}")

    # Download all OHLCV in one shot
    print("[backtest] downloading history (this may take a minute)...")
    BATCH = 100
    all_hist: dict[str, pd.DataFrame] = {}
    for i in range(0, len(universe), BATCH):
        batch = universe[i:i + BATCH]
        print(f"[backtest] batch {i // BATCH + 1}/{(len(universe) + BATCH - 1) // BATCH}...")
        all_hist.update(_fetch_history(batch, start, end))

    # Nifty benchmark
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nifty_raw = yf.download(_NIFTY, start=start.isoformat(), end=end.isoformat(),
                                auto_adjust=True, progress=False)
    nifty_close = nifty_raw["Close"].squeeze() if not nifty_raw.empty else pd.Series(dtype=float)

    # Find all Friday signal dates within the backtest window
    signal_date = start + timedelta(days=(4 - start.weekday()) % 7)  # first Friday
    rows: list[dict] = []

    while signal_date + timedelta(days=14) <= today:
        # Convert to Timestamp for DataFrame indexing
        ts = pd.Timestamp(signal_date)

        # Find qualifying signals on this date
        picks: list[tuple[str, float]] = []
        for ticker, df in all_hist.items():
            available = df[df.index <= ts]
            if available.empty:
                continue
            score = _compute_signal_score(df, ts)
            if score >= 4.0:
                picks.append((ticker, score))

        picks.sort(key=lambda x: x[1], reverse=True)
        top5 = picks[:5]

        # Nifty benchmark returns
        nifty_available = nifty_close[nifty_close.index <= ts]
        nifty_future = nifty_close[nifty_close.index > ts]
        nifty_5d = nifty_10d = None
        if not nifty_available.empty and len(nifty_future) >= 5:
            n_entry = float(nifty_available.iloc[-1])
            nifty_5d  = round((float(nifty_future.iloc[4]) / n_entry - 1) * 100, 2)
        if not nifty_available.empty and len(nifty_future) >= 10:
            n_entry = float(nifty_available.iloc[-1])
            nifty_10d = round((float(nifty_future.iloc[9]) / n_entry - 1) * 100, 2)

        for ticker, score in top5:
            df = all_hist[ticker]
            ret5  = _forward_return(df, ts, 5)
            ret10 = _forward_return(df, ts, 10)
            rows.append({
                "signal_date": signal_date.isoformat(),
                "ticker": ticker,
                "score": score,
                "ret_5d_%": ret5,
                "ret_10d_%": ret10,
                "nifty_5d_%": nifty_5d,
                "nifty_10d_%": nifty_10d,
                "beat_nifty_5d": (ret5 is not None and nifty_5d is not None and ret5 > nifty_5d),
                "beat_nifty_10d": (ret10 is not None and nifty_10d is not None and ret10 > nifty_10d),
            })

        signal_date += timedelta(days=7)

    if not rows:
        print("[backtest] no qualifying picks found in the period.")
        return

    results = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS — last", weeks, "weeks")
    print("=" * 70)
    print(results.to_string(index=False))

    # Summary
    valid5  = results.dropna(subset=["ret_5d_%"])
    valid10 = results.dropna(subset=["ret_10d_%"])

    print("\n── Summary ──")
    print(f"  Total signals : {len(results)}")
    if not valid5.empty:
        hit5 = valid5["beat_nifty_5d"].mean() * 100
        avg5 = valid5["ret_5d_%"].mean()
        print(f"  5d  hit rate  : {hit5:.1f}%  (avg return: {avg5:+.2f}%)")
    if not valid10.empty:
        hit10 = valid10["beat_nifty_10d"].mean() * 100
        avg10 = valid10["ret_10d_%"].mean()
        print(f"  10d hit rate  : {hit10:.1f}%  (avg return: {avg10:+.2f}%)")

    # Save
    out_path = ROOT / "outputs" / "backtest_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\n[backtest] full results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest NSE/BSE signal quality")
    parser.add_argument("--weeks", type=int, default=26, help="Number of weeks to backtest (default: 26)")
    args = parser.parse_args()
    run_backtest(weeks=args.weeks)
