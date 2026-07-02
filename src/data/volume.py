"""Volume anomaly scanner: today's volume vs 30-day average."""

import logging
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
NIFTY500_CSV = DATA_DIR / "nifty500.csv"
SME_CSV = DATA_DIR / "sme_list.csv"

VOLUME_SURGE_THRESHOLD = 2.5  # was 5.0 — too restrictive, genuine moves start at 2-3x
HEAVY_SELLING_VOL_RATIO = 1.5   # lower bar than volume_surge — distribution doesn't need a 2.5x spike
HEAVY_SELLING_PCT_CHANGE = -4.0
SMALL_CAP_THRESHOLD_CR = 500
BATCH_SIZE = 100  # yfinance handles ~100 tickers per batch comfortably


def _load_universe() -> list[str]:
    tickers: list[str] = []
    for csv_path in [NIFTY500_CSV, SME_CSV]:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            col = next((c for c in df.columns if "symbol" in c.lower()), df.columns[0])
            tickers += [f"{s.strip().upper()}.NS" for s in df[col].dropna()]
    if not tickers:
        tickers = [
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
            "SBIN.NS", "AXISBANK.NS", "WIPRO.NS", "SUNPHARMA.NS", "HCLTECH.NS",
        ]
        print("[volume] WARNING: no universe CSVs found, using fallback ticker list")
    return list(dict.fromkeys(tickers))


def _batch_download(tickers: list[str], lookback_days: int = 36) -> pd.DataFrame:
    """Download OHLCV for a list of tickers in one request. Returns a MultiIndex DataFrame."""
    end = date.today()
    start = end - timedelta(days=lookback_days)
    df = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
    )
    return df


def _parse_batch(df: pd.DataFrame, tickers: list[str]) -> list[dict]:
    results = []

    for ticker in tickers:
        try:
            if isinstance(df.columns, pd.MultiIndex):
                # Multi-ticker download: columns are (ticker, field) with group_by="ticker"
                if ticker not in df.columns.get_level_values(0):
                    continue
                t_df = df.xs(ticker, axis=1, level=0).dropna(how="all")
            else:
                # Single-ticker fallback (shouldn't happen in batch, but be safe)
                t_df = df.dropna(how="all")

            if t_df.empty or len(t_df) < 5:
                continue

            volumes = t_df["Volume"].dropna()
            closes = t_df["Close"].dropna()

            if len(volumes) < 2:
                continue

            today_vol = float(volumes.iloc[-1])
            avg_30d = float(volumes.iloc[:-1].tail(30).mean())
            if avg_30d == 0:
                continue

            volume_ratio = today_vol / avg_30d
            today_close = float(closes.iloc[-1])
            prev_close = float(closes.iloc[-2])
            pct_change = (today_close - prev_close) / prev_close * 100 if prev_close else 0.0

            results.append({
                "ticker": ticker,
                "today_volume": today_vol,
                "avg_30d_volume": avg_30d,
                "volume_ratio": round(volume_ratio, 2),
                "today_close": today_close,
                "pct_change": round(pct_change, 2),
                "market_cap_cr": None,  # fetched separately only for surge/heavy-selling candidates
                "is_volume_surge": volume_ratio >= VOLUME_SURGE_THRESHOLD,
                "is_small_cap": False,  # enriched below for surge/heavy-selling candidates only
                "is_distribution": volume_ratio >= VOLUME_SURGE_THRESHOLD and today_close < prev_close,
                "is_heavy_selling": volume_ratio >= HEAVY_SELLING_VOL_RATIO and pct_change <= HEAVY_SELLING_PCT_CHANGE,
            })
        except Exception:
            continue

    return results


def _enrich_market_cap(candidates: list[dict]) -> None:
    """Fetch market cap only for the (small) set of surge candidates."""
    for c in candidates:
        try:
            info = yf.Ticker(c["ticker"]).fast_info
            mkt_cap = getattr(info, "market_cap", None)
            if mkt_cap:
                c["market_cap_cr"] = round(mkt_cap / 1e7, 1)
                c["is_small_cap"] = c["market_cap_cr"] < SMALL_CAP_THRESHOLD_CR
        except Exception:
            pass


def fetch_volume_gainers() -> list[dict]:
    """
    Return stocks with volume_ratio >= VOLUME_SURGE_THRESHOLD (bullish surge) UNION
    is_heavy_selling (bearish — lower volume bar + a real down day, feeds the short
    pipeline), sorted by volume_ratio descending. Function name kept for compatibility
    with existing callers even though the result set is no longer purely bullish.
    Uses batch yfinance downloads for speed (~10× faster than per-ticker calls).
    Results are cached to disk for the calendar day — subsequent same-day calls are instant.
    """
    import os
    from src.cache import load_today, load_latest, save_today
    cached = load_today("volume_gainers")
    if cached is not None:
        return cached
    if os.environ.get("SCAN_MODE") == "pre_market":
        cached = load_latest("volume_gainers")
        if cached is not None:
            return cached  # pre-market: reuse yesterday's data, no new download

    universe = _load_universe()
    print(f"[volume] scanning {len(universe)} tickers in batches of {BATCH_SIZE}...")

    all_records: list[dict] = []
    batches = [universe[i:i + BATCH_SIZE] for i in range(0, len(universe), BATCH_SIZE)]

    for i, batch in enumerate(batches):
        print(f"[volume] batch {i + 1}/{len(batches)} ({len(batch)} tickers)...")
        try:
            df = _batch_download(batch)
            records = _parse_batch(df, batch)
            all_records.extend(records)
        except Exception as exc:
            print(f"[volume] batch {i + 1} failed: {exc}")

    surge_candidates = [r for r in all_records if r["is_volume_surge"] or r["is_heavy_selling"]]
    _enrich_market_cap(surge_candidates)

    surge_candidates.sort(key=lambda x: x["volume_ratio"], reverse=True)
    n_selling = sum(1 for r in surge_candidates if r["is_heavy_selling"])
    print(f"[volume] {len(all_records)} tickers processed, {len(surge_candidates)} candidates "
          f"({len(surge_candidates) - n_selling} surge, {n_selling} heavy-selling)")
    save_today("volume_gainers", surge_candidates)
    return surge_candidates
