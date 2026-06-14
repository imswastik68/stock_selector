"""
Signal performance tracker.

Records daily pick outcomes (T1/T2 hit / SL hit / still open) using the same
exit engine as live trading (src.trade_sim.simulate_raw + WINNER_POLICY), so
tracked win-rate always reflects the chosen exit policy.

Data stored in outputs/performance.json:
  {
    "YYYY-MM-DD": {
      "TICKER.NS": {
        "direction": "buy",
        "entry": 245.0,
        "stop_loss": 238.0,
        "target_1": 262.0,
        "target_2": 280.0,          # stored so policies using T2 work correctly
        "outcome": "t1_hit" | "t2_hit" | "sl_hit" | "timeout" | "open",
        "outcome_date": "YYYY-MM-DD"
      }, ...
    }, ...
  }

NOTE: This tracks signal hit-rate over *all* generated picks (advisory).
      src/portfolio.py tracks the capital-constrained live book (actual P&L).

Running hit-rate stats computed over last 30 days.
"""

from __future__ import annotations

import json
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.trade_sim import WINNER_POLICY, simulate_raw

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
    for direction, key in [("buy", "buy_watchlist"), ("sell", "sell_watchlist")]:
        for entry in watchlist_data.get(key, []):
            t  = entry.get("ticker", "")
            sl = _parse_price(entry.get("stop_loss"))
            t1 = _parse_price(entry.get("target_1"))
            t2 = _parse_price(entry.get("target_2"))
            price = entry.get("today_close")
            if t and sl and t1:
                picks[t] = {
                    "direction":  direction,
                    "entry":      price,
                    "stop_loss":  sl,
                    "target_1":   t1,
                    "target_2":   t2,
                    "outcome":    "open",
                    "outcome_date": None,
                }

    if picks:
        perf[scan_date] = picks
        _save_perf(perf)
        print(f"[performance] recorded {len(picks)} picks for {scan_date}")


def evaluate_prior_picks(lookback_days: int = 7) -> dict:
    """
    For each open pick from the last `lookback_days` days, fetch OHLCV and
    update outcome via simulate_raw(exit_policy=WINNER_POLICY).
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
    print(f"[performance] evaluating {len(tickers)} open picks via {WINNER_POLICY} policy...")

    # 30d window: long enough for time_10d policy + buffer
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                tickers, period="30d", interval="1d",
                auto_adjust=True, progress=False, group_by="ticker",
            )
    except Exception as exc:
        print(f"[performance] OHLC fetch error: {exc}")
        return perf

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
                pick_dt = pd.Timestamp(scan_date)
                # Subtract 1 business day so simulate_raw checks entry on the scan date bar
                as_of = pick_dt - pd.tseries.offsets.BusinessDay(1)

                entry_price = float(pick["entry"] or 0)
                sl  = float(pick["stop_loss"])
                t1  = float(pick["target_1"])
                t2  = float(pick["target_2"]) if pick.get("target_2") else None
                direction = pick["direction"]

                if entry_price <= 0:
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sim = simulate_raw(
                        entry_price, entry_price, entry_price,
                        sl, t1, direction, as_of, df,
                        t2=t2, exit_policy=WINNER_POLICY,
                    )

                if not sim.get("triggered"):
                    continue  # keep "open" — entry didn't fill

                pick["outcome"]      = sim["outcome"]
                pick["outcome_date"] = sim.get("exit_date")
                if sim["outcome"] != "open":
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
    t2_hit and t1_hit both count as wins; timeout is neutral.
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
            if outcome in ("t1_hit", "t2_hit"):
                t1 += 1
            elif outcome == "sl_hit":
                sl += 1
            else:
                open_count += 1

    decided = t1 + sl
    win_rate = round(t1 / decided * 100, 1) if decided > 0 else None

    return {
        "total_picks":    total,
        "t1_hit":         t1,
        "sl_hit":         sl,
        "open":           open_count,
        "win_rate_pct":   win_rate,
        "lookback_days":  lookback_days,
        "exit_policy":    WINNER_POLICY,
    }
