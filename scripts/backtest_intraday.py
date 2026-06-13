"""
Intraday signal sanity check — 60-day 5m bars (free yfinance limit).

Tests:
  1. intraday_surge: projected volume >= 2.5x AND price > prev close
     Measures same-day return from signal bar to session close.
  2. ORB (Opening Range Breakout): breakout above/below first 30-min high/low
     with volume confirmation.
  3. VWAP reclaim: price crosses above VWAP after being below (bullish signal).

NOTE: Only ~60 days of 5m history available free. Small sample — treat as
directional check, not statistically robust validation.

Usage:
  python scripts/backtest_intraday.py
  python scripts/backtest_intraday.py --tickers RELIANCE.NS TCS.NS INFY.NS
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
    import yfinance as yf
except ImportError:
    sys.exit("Run: pip install yfinance pandas")

OUTPUTS = ROOT / "outputs"
_NSE_OPEN  = pd.Timedelta(hours=9, minutes=15)   # 9:15 AM IST
_NSE_CLOSE = pd.Timedelta(hours=15, minutes=30)  # 3:30 PM IST
_IST = timezone(timedelta(hours=5, minutes=30))


def _default_tickers() -> list[str]:
    """Top-50 liquid NIFTY stocks for intraday testing."""
    csv = ROOT / "data" / "nifty500.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        col = next((c for c in df.columns if "symbol" in c.lower()), df.columns[0])
        tickers = [f"{s.strip().upper()}.NS" for s in df[col].dropna()]
        return tickers[:50]
    return [
        "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
        "SBIN.NS","AXISBANK.NS","WIPRO.NS","SUNPHARMA.NS","HCLTECH.NS",
        "MARUTI.NS","TITAN.NS","BAJFINANCE.NS","NESTLEIND.NS","LT.NS",
        "ASIANPAINT.NS","ULTRACEMCO.NS","ITC.NS","BHARTIARTL.NS","KOTAKBANK.NS",
    ]


def _fetch_5m(tickers: list[str]) -> dict[str, pd.DataFrame]:
    print(f"[intraday_bt] downloading 5m bars for {len(tickers)} tickers (~60d limit)...")
    result: dict[str, pd.DataFrame] = {}
    batch_size = 20
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(
                    batch, period="60d", interval="5m",
                    auto_adjust=True, progress=False, group_by="ticker",
                )
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                for t in batch:
                    if t in raw.columns.get_level_values(0):
                        df = raw[t].dropna(how="all")
                        if not df.empty:
                            result[t] = df
            elif len(batch) == 1:
                df = raw.dropna(how="all")
                if not df.empty:
                    result[batch[0]] = df
        except Exception as exc:
            print(f"[intraday_bt] batch error: {exc}")
    print(f"[intraday_bt] got 5m data for {len(result)}/{len(tickers)} tickers")
    return result


def _fetch_daily(tickers: list[str]) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(tickers, period="90d", interval="1d",
                          auto_adjust=True, progress=False, group_by="ticker")
    if raw.empty:
        return result
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if t in raw.columns.get_level_values(0):
                df = raw[t].dropna(how="all")
                df = df[df["Close"].notna()]
                if not df.empty:
                    result[t] = df
    elif len(tickers) == 1:
        df = raw.dropna(how="all")
        df = df[df["Close"].notna()]
        if not df.empty:
            result[tickers[0]] = df
    return result


def _to_ist(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("Asia/Kolkata")


def _analyze_day(day_5m: pd.DataFrame, prev_close: float,
                 avg_30d_vol: float) -> dict:
    """
    Analyze one trading day's 5m bars for intraday signals and outcomes.
    Returns signal flags + day return.
    """
    if day_5m.empty or prev_close <= 0:
        return {}

    day_5m = day_5m.copy()
    day_5m.index = day_5m.index.map(_to_ist)

    # Filter to NSE session (9:15 – 15:30)
    times = day_5m.index.map(lambda t: pd.Timedelta(hours=t.hour, minutes=t.minute))
    session = day_5m[(times >= _NSE_OPEN) & (times <= _NSE_CLOSE)]
    if session.empty or len(session) < 6:
        return {}

    # Daily volume
    total_vol = float(session["Volume"].sum())
    vol_proj_ratio = (total_vol / avg_30d_vol) if avg_30d_vol > 0 else 0

    open_price  = float(session["Open"].iloc[0])
    close_price = float(session["Close"].iloc[-1])
    day_return  = (close_price - prev_close) / prev_close * 100

    # ── Signal 1: intraday_surge ──────────────────────────────────────────────
    # Detect if volume surge + price > prev close at the mid-session mark (~bar 25, ~2h in)
    mid_bar = min(25, len(session) // 2)
    vol_so_far = float(session["Volume"].iloc[:mid_bar].sum())
    fraction   = mid_bar / 75  # 75 × 5m bars = 375 min session
    proj_vol   = vol_so_far / fraction if fraction > 0 else 0
    proj_ratio = proj_vol / avg_30d_vol if avg_30d_vol > 0 else 0
    mid_price  = float(session["Close"].iloc[mid_bar-1])
    surge_signal = bool(proj_ratio >= 2.5 and mid_price > prev_close)
    # Return from mid-session signal bar to close
    surge_ret = (close_price - mid_price) / mid_price * 100 if surge_signal else None

    # ── Signal 2: ORB (Opening Range Breakout) ────────────────────────────────
    # Opening range = first 6 bars (30 min)
    orb_bars = session.iloc[:6]
    orb_high = float(orb_bars["High"].max())
    orb_low  = float(orb_bars["Low"].min())
    orb_vol  = float(orb_bars["Volume"].sum())
    orb_avg_vol = avg_30d_vol / 75 * 6  # expected 30-min vol
    orb_vol_ok  = orb_vol > orb_avg_vol * 1.5

    orb_bull = orb_sell = False
    orb_bull_ret = orb_sell_ret = None

    rest = session.iloc[6:]
    for i, (idx, row) in enumerate(rest.iterrows()):
        if not orb_bull and float(row["Close"]) > orb_high and orb_vol_ok:
            orb_bull = True
            orb_entry = float(row["Close"])
            # Return from breakout bar to day close
            future_bars = rest.iloc[i+1:]
            if not future_bars.empty:
                orb_bull_ret = (float(future_bars["Close"].iloc[-1]) - orb_entry) / orb_entry * 100
            break

    for i, (idx, row) in enumerate(rest.iterrows()):
        if not orb_sell and float(row["Close"]) < orb_low and orb_vol_ok:
            orb_sell = True
            orb_entry = float(row["Close"])
            future_bars = rest.iloc[i+1:]
            if not future_bars.empty:
                orb_sell_ret = (orb_entry - float(future_bars["Close"].iloc[-1])) / orb_entry * 100
            break

    # ── Signal 3: VWAP reclaim ────────────────────────────────────────────────
    # VWAP: cumulative (price × vol) / cumulative vol
    hlc3 = (session["High"] + session["Low"] + session["Close"]) / 3
    cum_pv = (hlc3 * session["Volume"]).cumsum()
    cum_v  = session["Volume"].cumsum()
    vwap   = cum_pv / cum_v.replace(0, float("nan"))

    prices = session["Close"].values
    vwap_v = vwap.values
    vwap_reclaim = False
    vwap_ret     = None

    for i in range(1, len(session) - 1):
        if (prices[i-1] < vwap_v[i-1] and prices[i] > vwap_v[i]
                and float(session["Volume"].iloc[i]) > avg_30d_vol / 75 * 1.5):
            vwap_reclaim = True
            entry = prices[i]
            vwap_ret = (prices[-1] - entry) / entry * 100
            break

    return {
        "open":    open_price,
        "close":   close_price,
        "prev_close": prev_close,
        "day_return": round(day_return, 2),
        "vol_proj_ratio": round(vol_proj_ratio, 2),
        # surge
        "surge_signal": surge_signal,
        "surge_ret":    round(surge_ret, 2) if surge_ret is not None else None,
        # ORB
        "orb_bull":      orb_bull,
        "orb_bull_ret":  round(orb_bull_ret, 2) if orb_bull_ret is not None else None,
        "orb_sell":      orb_sell,
        "orb_sell_ret":  round(orb_sell_ret, 2) if orb_sell_ret is not None else None,
        # VWAP
        "vwap_reclaim":  vwap_reclaim,
        "vwap_ret":      round(vwap_ret, 2) if vwap_ret is not None else None,
    }


def run_intraday_backtest(tickers: list[str]) -> None:
    bars_5m = _fetch_5m(tickers)
    daily   = _fetch_daily(tickers)

    all_days: list[dict] = []

    for ticker in tickers:
        df5  = bars_5m.get(ticker)
        dfd  = daily.get(ticker)
        if df5 is None or dfd is None or len(dfd) < 5:
            continue

        avg_30d_vol = float(dfd["Volume"].iloc[:-1].tail(30).mean())

        # Localize 5m index to IST and split by date
        idx_ist = df5.index.map(_to_ist)
        dates   = pd.Series(idx_ist).dt.date.unique()

        for d in dates:
            ts_d = pd.Timestamp(d)
            # Previous close from daily
            prev_daily = dfd[dfd.index.date < d]
            if prev_daily.empty:
                continue
            prev_close = float(prev_daily["Close"].iloc[-1])

            # 5m bars for this day
            mask   = pd.Series(idx_ist).dt.date == d
            day_5m = df5.iloc[mask.values]
            if day_5m.empty:
                continue

            result = _analyze_day(day_5m, prev_close, avg_30d_vol)
            if not result:
                continue
            result["ticker"] = ticker
            result["date"]   = d.isoformat()
            all_days.append(result)

    if not all_days:
        print("[intraday_bt] no days analyzed")
        return

    df = pd.DataFrame(all_days)
    total_days = len(df)

    # ── per-signal stats ──────────────────────────────────────────────────────
    def _stats(mask: pd.Series, ret_col: str, name: str):
        subset = df[mask & df[ret_col].notna()]
        n = len(subset)
        if n == 0:
            return
        avg = float(subset[ret_col].mean())
        pos = float((subset[ret_col] > 0).mean() * 100)
        print(f"  {name:<25}  n={n:>4}  ({n/total_days*100:.0f}% of days)  "
              f"avg_ret={avg:>+5.2f}%  positive={pos:.0f}%")

    print(f"\n{'='*60}")
    print(f"  INTRADAY SIGNAL STATS — {total_days} day-ticker observations")
    print(f"  ⚠ Small sample (~60d). Directional check only.")
    print(f"{'='*60}")
    _stats(df["surge_signal"], "surge_ret",    "intraday_surge (mid-session)")
    _stats(df["orb_bull"],     "orb_bull_ret", "ORB breakout (bull)")
    _stats(df["orb_sell"],     "orb_sell_ret", "ORB breakdown (sell)")
    _stats(df["vwap_reclaim"], "vwap_ret",     "VWAP reclaim (bull)")

    # Baseline: average day return
    avg_day = float(df["day_return"].mean())
    print(f"\n  Baseline avg day return: {avg_day:+.2f}%")
    print(f"{'='*60}")

    # Save
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS / "backtest_intraday.json"
    out_path.write_text(json.dumps({
        "meta": {
            "tickers": len(tickers),
            "tickers_with_data": len(bars_5m),
            "day_ticker_obs": total_days,
            "note": "5m bars, ~60d free yfinance limit. Small sample.",
        },
        "signal_stats": {
            "surge": {
                "n":         int(df["surge_signal"].sum()),
                "avg_ret":   round(float(df.loc[df["surge_signal"], "surge_ret"].dropna().mean()), 2) if df["surge_signal"].any() else None,
                "pct_pos":   round(float((df.loc[df["surge_signal"], "surge_ret"].dropna() > 0).mean() * 100), 1) if df["surge_signal"].any() else None,
            },
            "orb_bull": {
                "n":       int(df["orb_bull"].sum()),
                "avg_ret": round(float(df.loc[df["orb_bull"], "orb_bull_ret"].dropna().mean()), 2) if df["orb_bull"].any() else None,
                "pct_pos": round(float((df.loc[df["orb_bull"], "orb_bull_ret"].dropna() > 0).mean() * 100), 1) if df["orb_bull"].any() else None,
            },
            "vwap_reclaim": {
                "n":       int(df["vwap_reclaim"].sum()),
                "avg_ret": round(float(df.loc[df["vwap_reclaim"], "vwap_ret"].dropna().mean()), 2) if df["vwap_reclaim"].any() else None,
                "pct_pos": round(float((df.loc[df["vwap_reclaim"], "vwap_ret"].dropna() > 0).mean() * 100), 1) if df["vwap_reclaim"].any() else None,
            },
        },
    }, indent=2))
    print(f"\n  Results → {out_path.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=None,
                    help="Specific tickers to test (default: top-50 NIFTY)")
    args = ap.parse_args()
    tickers = args.tickers if args.tickers else _default_tickers()
    run_intraday_backtest(tickers)
