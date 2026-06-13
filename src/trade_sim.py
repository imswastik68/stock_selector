"""
Shared trade simulator — single source of truth for SL/T1 mechanics.

Used by scripts/analyse_picks.py and scripts/backtest.py.

simulate_trade(pick, df) — simulate one pick from a structured dict.
simulate_raw(entry_lo, entry_hi, entry_mid, sl, t1, direction, as_of_date, df)
  — simulate from raw levels (used by backtest which builds them from technicals).
"""

from __future__ import annotations

import pandas as pd


def simulate_raw(
    entry_lo: float,
    entry_hi: float,
    entry_mid: float,
    sl: float,
    t1: float,
    direction: str,
    as_of_date: pd.Timestamp,
    df: pd.DataFrame,
) -> dict:
    """
    Core simulator. df must be full OHLCV; as_of_date is the signal date.

    Entry rule (first 2 bars after as_of_date):
      BUY:  bar_low <= entry_hi → entry triggered at entry_mid (or entry_hi if opened above)
      SELL: bar_high >= entry_lo → entry triggered at entry_mid (or entry_lo if opened below)

    Exit rule (bar by bar from entry):
      BUY:  gap-open ≤ SL → exit at open; low ≤ SL and high ≥ T1 → SL (conservative);
            low ≤ SL → SL; high ≥ T1 → T1.
      SELL: mirror logic.
    """
    future = df[df.index > as_of_date].copy()
    if future.empty:
        return {"triggered": False, "outcome": "no_data"}

    today_close = float(future["Close"].iloc[-1])

    # ── 1. Entry check (first 2 bars) ────────────────────────────────────────
    entry_window = future.iloc[:2]
    triggered    = False
    entry_price  = None
    entry_idx    = None

    for idx, row in entry_window.iterrows():
        bar_low  = float(row["Low"])
        bar_high = float(row["High"])
        bar_open = float(row["Open"])

        if direction == "buy":
            if bar_low <= entry_hi:
                triggered   = True
                entry_price = entry_mid if bar_open <= entry_hi else entry_hi
                entry_idx   = idx
                break
        else:
            if bar_high >= entry_lo:
                triggered   = True
                entry_price = entry_mid if bar_open >= entry_lo else entry_lo
                entry_idx   = idx
                break

    if not triggered:
        gap = (today_close - entry_mid) / entry_mid * 100 if entry_mid else 0
        return {
            "triggered":     False,
            "outcome":       "not_triggered",
            "entry_mid":     entry_mid,
            "current_price": today_close,
            "gap_pct":       round(gap, 2),
        }

    # ── 2. Scan forward for T1 / SL ──────────────────────────────────────────
    post_entry = future[future.index >= entry_idx]
    outcome    = "open"
    exit_price = today_close
    exit_idx   = future.index[-1]

    for idx, row in post_entry.iterrows():
        bar_low  = float(row["Low"])
        bar_high = float(row["High"])
        bar_open = float(row["Open"])

        if direction == "buy":
            if bar_open <= sl:
                outcome = "sl_hit"; exit_price = bar_open; exit_idx = idx; break
            sl_hit = bar_low  <= sl
            t1_hit = bar_high >= t1
            if sl_hit and t1_hit:
                outcome = "sl_hit"; exit_price = sl;       exit_idx = idx; break
            if sl_hit:
                outcome = "sl_hit"; exit_price = sl;       exit_idx = idx; break
            if t1_hit:
                outcome = "t1_hit"; exit_price = t1;       exit_idx = idx; break
        else:
            if bar_open >= sl:
                outcome = "sl_hit"; exit_price = bar_open; exit_idx = idx; break
            sl_hit = bar_high >= sl
            t1_hit = bar_low  <= t1
            if sl_hit and t1_hit:
                outcome = "sl_hit"; exit_price = sl;       exit_idx = idx; break
            if sl_hit:
                outcome = "sl_hit"; exit_price = sl;       exit_idx = idx; break
            if t1_hit:
                outcome = "t1_hit"; exit_price = t1;       exit_idx = idx; break

    days_held = max(1, (exit_idx - entry_idx).days)

    # ── 3. Returns ───────────────────────────────────────────────────────────
    if direction == "buy":
        return_pct    = (exit_price  - entry_price) / entry_price * 100
        t1_return_pct = (t1          - entry_price) / entry_price * 100
        sl_risk_pct   = (entry_price - sl)          / entry_price * 100
    else:
        return_pct    = (entry_price - exit_price)  / entry_price * 100
        t1_return_pct = (entry_price - t1)          / entry_price * 100
        sl_risk_pct   = (sl          - entry_price) / entry_price * 100

    lows  = post_entry["Low"].astype(float).values
    highs = post_entry["High"].astype(float).values
    if direction == "buy":
        mae = (min(lows)  - entry_price) / entry_price * 100
        mfe = (max(highs) - entry_price) / entry_price * 100
    else:
        mae = (entry_price - max(highs)) / entry_price * 100
        mfe = (entry_price - min(lows))  / entry_price * 100

    week_return = None
    if len(post_entry) >= 5:
        wc = float(post_entry.iloc[4]["Close"])
        week_return = round(((wc - entry_price) / entry_price * 100)
                            if direction == "buy"
                            else ((entry_price - wc) / entry_price * 100), 2)

    # Fixed-period returns for backtest analysis
    fwd: dict[str, float | None] = {}
    for n in (5, 10, 20):
        if len(post_entry) >= n:
            pc = float(post_entry.iloc[n - 1]["Close"])
            fwd[f"fwd_{n}d_pct"] = round((pc - entry_price) / entry_price * 100
                                         if direction == "buy"
                                         else (entry_price - pc) / entry_price * 100, 2)
        else:
            fwd[f"fwd_{n}d_pct"] = None

    return {
        "triggered":     True,
        "outcome":       outcome,
        "entry_price":   round(entry_price, 2),
        "exit_price":    round(exit_price, 2),
        "current_price": round(today_close, 2),
        "return_pct":    round(return_pct, 2),
        "t1_return_pct": round(t1_return_pct, 2),
        "sl_risk_pct":   round(sl_risk_pct, 2),
        "week_return":   week_return,
        "days_held":     days_held,
        "mae_pct":       round(mae, 2),
        "mfe_pct":       round(mfe, 2),
        **fwd,
    }


def simulate_trade(pick: dict, df: pd.DataFrame) -> dict:
    """Convenience wrapper for structured pick dicts (from telegram_picks.json)."""
    rec_date  = pd.Timestamp(pick["date"])
    entry_lo  = pick.get("entry_lo")  or pick["entry_mid"]
    entry_hi  = pick.get("entry_hi")  or pick["entry_mid"]
    entry_mid = pick["entry_mid"]
    sl        = pick["sl"]
    t1        = pick["t1"]
    direction = pick["direction"]

    # analyse_picks slices df from rec_date onwards; we need to pass as_of = day before
    # so simulate_raw can use df[index > as_of] and still include rec_date bars.
    # Simplest: pass as_of = rec_date - 1 business day, keeping rec_date in future.
    df_from = df[df.index >= rec_date].copy()
    if df_from.empty:
        return {"triggered": False, "outcome": "no_data"}

    # Inject a synthetic "yesterday" row so simulate_raw's future slice works correctly
    fake_yesterday = rec_date - pd.tseries.offsets.BusinessDay(1)
    return simulate_raw(entry_lo, entry_hi, entry_mid, sl, t1, direction,
                        fake_yesterday, df_from)
