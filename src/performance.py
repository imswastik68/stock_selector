"""
Signal performance tracker.

Records daily pick outcomes (T1 hit / SL hit / still open) by comparing
next-day OHLC against the entry's stop_loss and target_1 levels.

Data stored in outputs/performance.json:
  {
    "YYYY-MM-DD": {
      "TICKER.NS": {
        "direction": "buy",
        "entry": 245.0,
        "stop_loss": 238.0,
        "target_1": 262.0,
        "outcome": "t1_hit" | "sl_hit" | "open" | "unknown",
        "outcome_date": "YYYY-MM-DD"
      }, ...
    }, ...
  }

Running hit-rate stats computed over last 30 days.
"""

from __future__ import annotations

import json
import warnings
from datetime import date, timedelta
from pathlib import Path

import yfinance as yf

_PERF_FILE = Path(__file__).parent.parent / "outputs" / "performance.json"


def _parse_price(s) -> float | None:
    try:
        return float(str(s).replace("₹", "").replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _load_perf() -> dict:
    if _PERF_FILE.exists():
        try:
            return json.loads(_PERF_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_perf(data: dict) -> None:
    _PERF_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PERF_FILE.write_text(json.dumps(data, indent=2, default=str))


def record_picks(watchlist_data: dict) -> None:
    """
    Record today's buy/sell picks into performance.json.
    Call this after save_output() each EOD run.
    """
    scan_date = watchlist_data.get("scan_date", date.today().isoformat())
    perf = _load_perf()

    if scan_date in perf:
        return  # already recorded today

    picks = {}
    for entry in watchlist_data.get("buy_watchlist", []):
        t = entry.get("ticker", "")
        sl = _parse_price(entry.get("stop_loss"))
        t1 = _parse_price(entry.get("target_1"))
        price = entry.get("today_close")
        if t and sl and t1:
            picks[t] = {
                "direction": "buy",
                "entry": price,
                "stop_loss": sl,
                "target_1": t1,
                "outcome": "open",
                "outcome_date": None,
            }
    for entry in watchlist_data.get("sell_watchlist", []):
        t = entry.get("ticker", "")
        sl = _parse_price(entry.get("stop_loss"))
        t1 = _parse_price(entry.get("target_1"))
        price = entry.get("today_close")
        if t and sl and t1:
            picks[t] = {
                "direction": "sell",
                "entry": price,
                "stop_loss": sl,
                "target_1": t1,
                "outcome": "open",
                "outcome_date": None,
            }

    if picks:
        perf[scan_date] = picks
        _save_perf(perf)
        print(f"[performance] recorded {len(picks)} picks for {scan_date}")


def evaluate_prior_picks(lookback_days: int = 7) -> dict:
    """
    For each open pick from the last `lookback_days` days, fetch next-day OHLC
    and update outcome to 'sl_hit', 't1_hit', or keep 'open'.
    Returns updated performance dict.
    """
    perf = _load_perf()
    today = date.today()
    cutoff = (today - timedelta(days=lookback_days)).isoformat()

    open_by_ticker: dict[str, list[tuple[str, str, dict]]] = {}
    for scan_date, picks in perf.items():
        if scan_date < cutoff:
            continue
        for ticker, pick in picks.items():
            if pick.get("outcome") == "open":
                open_by_ticker.setdefault(ticker, []).append((scan_date, ticker, pick))

    if not open_by_ticker:
        return perf

    tickers = list(open_by_ticker.keys())
    print(f"[performance] evaluating {len(tickers)} open picks...")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                tickers, period="10d", interval="1d",
                auto_adjust=True, progress=False, group_by="ticker",
            )
    except Exception as exc:
        print(f"[performance] OHLC fetch error: {exc}")
        return perf

    import pandas as pd
    updated = 0
    for ticker, entries in open_by_ticker.items():
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker not in raw.columns.get_level_values(0):
                    continue
                df = raw[ticker].dropna(how="all")
            elif len(tickers) == 1:
                df = raw.dropna(how="all")
            else:
                continue

            if df.empty or len(df) < 2:
                continue

            for scan_date, t, pick in entries:
                pick_date = scan_date  # we check bars AFTER pick_date
                pick_dt = pd.Timestamp(pick_date)

                # Get bars after the pick date
                future = df[df.index > pick_dt]
                if future.empty:
                    continue

                direction = pick["direction"]
                sl = pick["stop_loss"]
                t1 = pick["target_1"]
                outcome = "open"
                outcome_date = None

                for bar_date, row in future.iterrows():
                    high = float(row["High"])
                    low  = float(row["Low"])
                    bar_str = str(bar_date.date())

                    if direction == "buy":
                        if low <= sl:
                            outcome = "sl_hit"
                            outcome_date = bar_str
                            break
                        if high >= t1:
                            outcome = "t1_hit"
                            outcome_date = bar_str
                            break
                    else:  # sell
                        if high >= sl:
                            outcome = "sl_hit"
                            outcome_date = bar_str
                            break
                        if low <= t1:
                            outcome = "t1_hit"
                            outcome_date = bar_str
                            break

                pick["outcome"] = outcome
                pick["outcome_date"] = outcome_date
                if outcome != "open":
                    updated += 1
        except Exception:
            continue

    if updated:
        _save_perf(perf)
        print(f"[performance] updated {updated} pick outcomes")

    return perf


def performance_summary(lookback_days: int = 30) -> dict:
    """
    Compute hit-rate stats over last `lookback_days` days.
    Returns {total, t1_hit, sl_hit, open, win_rate_pct}.
    """
    perf = evaluate_prior_picks(lookback_days)
    today = date.today()
    cutoff = (today - timedelta(days=lookback_days)).isoformat()

    total = t1 = sl = open_count = 0
    for scan_date, picks in perf.items():
        if scan_date < cutoff:
            continue
        for pick in picks.values():
            total += 1
            outcome = pick.get("outcome", "open")
            if outcome == "t1_hit":
                t1 += 1
            elif outcome == "sl_hit":
                sl += 1
            else:
                open_count += 1

    decided = t1 + sl
    win_rate = round(t1 / decided * 100, 1) if decided > 0 else None

    return {
        "total_picks": total,
        "t1_hit": t1,
        "sl_hit": sl,
        "open": open_count,
        "win_rate_pct": win_rate,
        "lookback_days": lookback_days,
    }
