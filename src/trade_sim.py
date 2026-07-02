"""
Shared trade simulator — single source of truth for SL/T1 mechanics.

Exit policies (exit_policy kwarg to simulate_raw):
  static          — first SL-or-T1 touch (default, original behavior)
  time_10d        — static + force-exit at close after 10 bars if still open
  breakeven_1r    — once price reaches entry+1R, raise SL→entry; target still T1
  partial_t1_be   — book 50% at T1, move SL→breakeven, ride remainder to T2
  trail_atr3      — chandelier trailing (3×ATR behind watermark); no fixed T1
  partial_t1_trail— book 50% at T1, then trail remainder 3×ATR (requires atr kwarg)

WINNER_POLICY: set after running scripts/backtest_exits.py
"""

from __future__ import annotations

import pandas as pd

# Applied in performance.py tracker and Telegram exit-plan line.
# Run scripts/backtest_exits.py then update this constant.
WINNER_POLICY: str = "static"

# Human-readable exit plan shown in Telegram alerts. Update alongside WINNER_POLICY.
EXIT_PLAN_HINT: dict[str, str] = {
    "static":           "first touch: exit at SL or T1",
    "time_10d":         "first touch + force-exit day 10 if still open",
    "breakeven_1r":     "SL → breakeven after +1R; target T1",
    "partial_t1_be":    "book 50% @ T1 · SL → breakeven · ride to T2",
    "trail_atr3":       "chandelier trail 3×ATR, no fixed target",
    "partial_t1_trail": "book 50% @ T1 · trail rest 3×ATR",
}


def simulate_raw(
    entry_lo: float,
    entry_hi: float,
    entry_mid: float,
    sl: float,
    t1: float,
    direction: str,
    as_of_date: pd.Timestamp,
    df: pd.DataFrame,
    *,
    t2: float | None = None,
    atr: float | None = None,
    exit_policy: str = "static",
    cost_pct: float = 0.0,
) -> dict:
    """
    Core simulator. df must be full OHLCV; as_of_date is the signal date.

    Entry rule (first 2 bars after as_of_date):
      BUY:  bar_low <= entry_hi → triggered at entry_mid (or entry_hi if gap-up open)
      SELL: bar_high >= entry_lo → triggered at entry_mid (or entry_lo if gap-down open)

    Exit policies: see module docstring. Default 'static' = original behavior.

    cost_pct: round-trip transaction cost (% of notional, see src/costs.py), deducted
    from return_pct when triggered. Default 0.0 = no behavior change vs pre-cost-model
    callers. return_pct_gross is always reported alongside net return_pct. fwd_*, mae/mfe,
    t1_return_pct, sl_risk_pct stay gross — they're geometry/diagnostics, not realized P&L.
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

    # ── 2. Exit scan ─────────────────────────────────────────────────────────
    post_entry = future[future.index >= entry_idx]
    outcome    = "open"
    exit_price = today_close
    exit_idx   = future.index[-1]

    # Policy state
    dynamic_sl   = sl               # may ratchet for trailing / breakeven policies
    partial_done = False            # True once 50% booked at T1 (partial policies)
    partial_price: float | None = None
    watermark    = entry_price      # running high (buy) / low (sell) for trailing

    R = abs(entry_price - sl)       # 1 risk unit

    for bar_n, (idx, row) in enumerate(post_entry.iterrows()):
        bar_low   = float(row["Low"])
        bar_high  = float(row["High"])
        bar_open  = float(row["Open"])
        bar_close = float(row["Close"])

        # ── time stop ─────────────────────────────────────────────────────
        if exit_policy == "time_10d" and bar_n >= 10:
            outcome = "timeout"; exit_price = bar_close; exit_idx = idx; break

        # ── trailing stop ratchet (before exit checks) ────────────────────
        if exit_policy in ("trail_atr3", "partial_t1_trail") and atr and atr > 0:
            if direction == "buy":
                watermark  = max(watermark, bar_high)
                dynamic_sl = max(dynamic_sl, watermark - 3 * atr)
            else:
                watermark  = min(watermark, bar_low)
                dynamic_sl = min(dynamic_sl, watermark + 3 * atr)

        # ── breakeven SL ratchet (before exit checks) ─────────────────────
        if exit_policy == "breakeven_1r":
            if direction == "buy" and bar_high >= entry_price + R:
                dynamic_sl = max(dynamic_sl, entry_price)
            elif direction == "sell" and bar_low <= entry_price - R:
                dynamic_sl = min(dynamic_sl, entry_price)

        # ── gap-open SL (highest priority) ───────────────────────────────
        if direction == "buy" and bar_open <= dynamic_sl:
            outcome = "sl_hit"; exit_price = bar_open; exit_idx = idx; break
        if direction == "sell" and bar_open >= dynamic_sl:
            outcome = "sl_hit"; exit_price = bar_open; exit_idx = idx; break

        # ── partial T1 booking (first time only) ──────────────────────────
        if not partial_done and exit_policy in ("partial_t1_be", "partial_t1_trail"):
            hit = (direction == "buy" and bar_high >= t1) or \
                  (direction == "sell" and bar_low  <= t1)
            if hit:
                partial_done  = True
                partial_price = t1
                if exit_policy == "partial_t1_be":
                    dynamic_sl = entry_price   # SL → breakeven for remainder
                continue   # 50% booked; remainder continues next bar

        # ── active target for remainder ───────────────────────────────────
        if exit_policy in ("trail_atr3", "partial_t1_trail"):
            active_target: float | None = None          # trailing only, no fixed target
        elif exit_policy == "partial_t1_be" and partial_done:
            active_target = t2                           # None → no upper target
        else:
            active_target = t1

        # ── normal SL / target exit ───────────────────────────────────────
        if direction == "buy":
            sl_hit = bar_low  <= dynamic_sl
            t1_hit = active_target is not None and bar_high >= active_target
            if sl_hit and t1_hit:
                outcome = "sl_hit"; exit_price = dynamic_sl; exit_idx = idx; break
            if sl_hit:
                outcome = "sl_hit"; exit_price = dynamic_sl; exit_idx = idx; break
            if t1_hit:
                lbl = "t2_hit" if (exit_policy == "partial_t1_be" and partial_done and t2) else "t1_hit"
                outcome = lbl; exit_price = active_target; exit_idx = idx; break
        else:
            sl_hit = bar_high >= dynamic_sl
            t1_hit = active_target is not None and bar_low  <= active_target
            if sl_hit and t1_hit:
                outcome = "sl_hit"; exit_price = dynamic_sl; exit_idx = idx; break
            if sl_hit:
                outcome = "sl_hit"; exit_price = dynamic_sl; exit_idx = idx; break
            if t1_hit:
                lbl = "t2_hit" if (exit_policy == "partial_t1_be" and partial_done and t2) else "t1_hit"
                outcome = lbl; exit_price = active_target; exit_idx = idx; break

    days_held = max(1, (exit_idx - entry_idx).days)

    # ── 3. Returns ───────────────────────────────────────────────────────────
    if direction == "buy":
        t1_return_pct = (t1          - entry_price) / entry_price * 100
        sl_risk_pct   = (entry_price - sl)          / entry_price * 100
    else:
        t1_return_pct = (entry_price - t1)          / entry_price * 100
        sl_risk_pct   = (sl          - entry_price) / entry_price * 100

    # Blended return for partial policies (50% leg1 at T1 + 50% leg2 at final exit)
    if partial_done and partial_price is not None:
        if direction == "buy":
            leg1 = (partial_price - entry_price) / entry_price * 100
            leg2 = (exit_price    - entry_price) / entry_price * 100
        else:
            leg1 = (entry_price - partial_price) / entry_price * 100
            leg2 = (entry_price - exit_price)    / entry_price * 100
        return_pct = 0.5 * leg1 + 0.5 * leg2
    else:
        if direction == "buy":
            return_pct = (exit_price - entry_price) / entry_price * 100
        else:
            return_pct = (entry_price - exit_price) / entry_price * 100

    return_pct_gross = return_pct
    return_pct = return_pct - cost_pct

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
        "exit_date":     exit_idx.date().isoformat() if hasattr(exit_idx, "date") else str(exit_idx),
        "entry_price":   round(entry_price, 2),
        "exit_price":    round(exit_price, 2),
        "current_price": round(today_close, 2),
        "return_pct":       round(return_pct, 2),
        "return_pct_gross": round(return_pct_gross, 2),
        "cost_pct":         round(cost_pct, 2),
        "t1_return_pct": round(t1_return_pct, 2),
        "sl_risk_pct":   round(sl_risk_pct, 2),
        "week_return":   week_return,
        "days_held":     days_held,
        "mae_pct":       round(mae, 2),
        "mfe_pct":       round(mfe, 2),
        **fwd,
    }


def simulate_trade(pick: dict, df: pd.DataFrame, exit_policy: str = "static") -> dict:
    """Convenience wrapper for structured pick dicts (from telegram_picks.json)."""
    rec_date  = pd.Timestamp(pick["date"])
    entry_lo  = pick.get("entry_lo")  or pick["entry_mid"]
    entry_hi  = pick.get("entry_hi")  or pick["entry_mid"]
    entry_mid = pick["entry_mid"]
    sl        = pick["sl"]
    t1        = pick["t1"]
    t2        = pick.get("t2")
    direction = pick["direction"]

    df_from = df[df.index >= rec_date].copy()
    if df_from.empty:
        return {"triggered": False, "outcome": "no_data"}

    fake_yesterday = rec_date - pd.tseries.offsets.BusinessDay(1)
    return simulate_raw(entry_lo, entry_hi, entry_mid, sl, t1, direction,
                        fake_yesterday, df_from, t2=t2, exit_policy=exit_policy)
