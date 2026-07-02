"""52-week low breakdown detector with volume confirmation — mirror of breakouts.py.

Sells were previously an incidental by-product of buy-oriented scans (a stock only
became a sell pick if its Wyckoff phase happened to classify DISTRIBUTION/MARKDOWN
after surfacing on a volume/breakout scan). This is the missing dedicated bearish
scan source.
"""

import logging
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
NIFTY500_CSV = DATA_DIR / "nifty500.csv"
SME_CSV = DATA_DIR / "sme_list.csv"

LOW_PROXIMITY_PCT = 5.0   # within 5% of 52w low
MIN_VOLUME_RATIO = 1.2
CONSOLIDATION_WEEKS = 3
BATCH_SIZE = 100


def _load_universe() -> list[str]:
    tickers: list[str] = []
    for csv_path in [NIFTY500_CSV, SME_CSV]:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            col = next((c for c in df.columns if "symbol" in c.lower()), df.columns[0])
            tickers += [f"{s.strip().upper()}.NS" for s in df[col].dropna()]
    if not tickers:
        tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
    return list(dict.fromkeys(tickers))


def _check_consolidation(closes: pd.Series, weeks: int = 3) -> bool:
    window = closes.iloc[-(weeks * 5 + 1):-1]
    if len(window) < 5:
        return False
    price_range_pct = (window.max() - window.min()) / window.mean() * 100
    return price_range_pct < 5.0


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

            if t_df.empty or len(t_df) < 50:
                continue

            closes = t_df["Close"].dropna()
            volumes = t_df["Volume"].dropna()

            if len(closes) < 50:
                continue

            # Use prior 52w low (excluding today) so we can detect actual breakdowns
            prior_52w_low = float(closes.iloc[:-1].tail(252).min())
            today_close = float(closes.iloc[-1])
            proximity_pct = (today_close - prior_52w_low) / prior_52w_low * 100

            if proximity_pct > LOW_PROXIMITY_PCT:
                continue

            today_vol = float(volumes.iloc[-1])
            avg_30d = float(volumes.iloc[:-1].tail(30).mean())
            volume_ratio = today_vol / avg_30d if avg_30d > 0 else 0

            if volume_ratio < MIN_VOLUME_RATIO:
                continue

            results.append({
                "ticker": ticker,
                "52w_low": round(prior_52w_low, 2),
                "today_close": round(today_close, 2),
                "proximity_pct": round(proximity_pct, 2),
                "actual_breakdown": today_close < prior_52w_low,  # price broke through prior 52w low
                "volume_ratio": round(volume_ratio, 2),
                "consolidation_breakdown": _check_consolidation(closes, CONSOLIDATION_WEEKS),
            })
        except Exception:
            continue

    return results


def fetch_breakdowns() -> list[dict]:
    """
    Return stocks within LOW_PROXIMITY_PCT of 52-week low with volume confirmation.
    Uses batch yfinance downloads for speed.
    Results are cached to disk for the calendar day — subsequent same-day calls are instant.
    """
    import os
    from src.cache import load_today, load_latest, save_today
    cached = load_today("breakdowns")
    if cached is not None:
        return cached
    if os.environ.get("SCAN_MODE") == "pre_market":
        cached = load_latest("breakdowns")
        if cached is not None:
            return cached  # pre-market: reuse yesterday's data, no new download

    universe = _load_universe()
    print(f"[breakdowns] scanning {len(universe)} tickers in batches of {BATCH_SIZE}...")

    end = date.today()
    start = end - timedelta(days=395)  # 52 weeks + buffer

    results = []
    batches = [universe[i:i + BATCH_SIZE] for i in range(0, len(universe), BATCH_SIZE)]

    for i, batch in enumerate(batches):
        print(f"[breakdowns] batch {i + 1}/{len(batches)} ({len(batch)} tickers)...")
        try:
            df = yf.download(
                batch,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
            )
            results.extend(_parse_batch(df, batch))
        except Exception as exc:
            print(f"[breakdowns] batch {i + 1} failed: {exc}")

    results.sort(key=lambda x: x["volume_ratio"], reverse=True)
    print(f"[breakdowns] {len(results)} breakdown candidates found")
    save_today("breakdowns", results)
    return results
