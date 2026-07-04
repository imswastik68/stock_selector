"""
Circuit-breaker threshold replay (Phase 5).

src/risk.py's drawdown_state thresholds (10%/15%/8%) were picked from
convention, never tested against this system's own trade history. This
script replays outputs/backtest_trades.csv (scripts/backtest.py's TA
backtest, closed trades with dates/returns/scores) chronologically through a
simulated capital-constrained book, and compares terminal equity + max
drawdown across a threshold grid (plus loss-streak cooldown on/off) to see
whether the live thresholds actually earn their keep.

Approximations (documented, not hidden):
  1. Equity only moves on REALIZED exits -- open positions are marked at
     COST, not mark-to-market. This makes replay drawdown a lagging
     (understated) proxy for true intra-trade drawdown -- a conservative
     bias that makes it HARDER, not easier, for the circuit breaker to show
     value here.
  2. exit_date = as_of + days_held (calendar days). days_held is measured
     from the actual entry bar (1-2 bars after as_of) to the exit bar
     (src/trade_sim.py:194), so this is off by a day or two vs as_of. Fine
     for a portfolio-level threshold comparison, not a trade-level P&L audit.
  3. sl reconstructed from close & atr_pct at risk_mult=2 (matches
     src/technicals.py:compute_entry_levels) since backtest_trades.csv
     doesn't store sl directly.

IMPORTANT (found by actually running this against the full dataset, not
assumed up front): drawdown_state's hysteresis has TWO ways out of "halted"
-- the book's own drawdown recovering to <=reset_pct, OR nifty_above_200dma
confirming a market recovery (src/risk.py:drawdown_state). main.py passes a
real value for the second escape valve every day; the first version of this
replay never did (always passed the default None), disabling that escape
route entirely. Combined with the "stop taking new trades while halted"
rule -- which removes the book's own ability to generate fresh gains to
climb back to its peak -- that produced a single ~4-year halt episode that
never reset for ANY threshold grid tested, an obviously non-representative
result. Fixed: nifty_close_series() below reconstructs the same
rolling-200-day-mean gate scripts/factor_backtest.py already uses for its
own below_200dma calculation, so the replay has the same two escape routes
the live system does.

Usage:
  python scripts/backtest_circuit.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
except ImportError:
    sys.exit("Run: pip install pandas")

from src.risk import RISK_CONFIG, drawdown_state, size_position

NIFTY_CSV = ROOT / "cache" / "backtest_nifty.csv"

TRADES_CSV = ROOT / "outputs" / "backtest_trades.csv"
OUT_FILE   = ROOT / "outputs" / "circuit_backtest.json"

CAPITAL         = RISK_CONFIG["capital"]
MAX_POSITIONS   = RISK_CONFIG["max_positions"]
MAX_NEW_PER_DAY = 3

LOSS_STREAK_N             = 5
LOSS_STREAK_COOLDOWN_DAYS = 7

# (name, reduced_pct, halted_pct, reset_pct) -- None triple = no circuit at all
GRID: list[tuple[str, float | None, float | None, float | None]] = [
    ("8/12",         8.0,  12.0,  6.0),
    ("10/15 (live)", 10.0, 15.0,  8.0),
    ("12/20",        12.0, 20.0, 10.0),
    ("no_circuit",   None, None, None),
]


def load_trades() -> pd.DataFrame:
    if not TRADES_CSV.exists():
        sys.exit(f"Not found: {TRADES_CSV}. Run scripts/backtest.py first.")
    df = pd.read_csv(TRADES_CSV)
    df = df[df["triggered"] == True]  # noqa: E712
    df = df[df["outcome"].isin(["sl_hit", "t1_hit", "t2_hit", "timeout"])]
    df = df[df["return_pct"].notna() & df["close"].notna() & df["atr_pct"].notna()
            & df["days_held"].notna()]
    df = df.copy()
    df["as_of"] = pd.to_datetime(df["as_of"])
    df["exit_date"] = df["as_of"] + pd.to_timedelta(df["days_held"].clip(lower=1), unit="D")
    df["sl"] = df.apply(
        lambda r: r["close"] - 2 * r["close"] * r["atr_pct"] / 100 if r["direction"] == "buy"
        else r["close"] + 2 * r["close"] * r["atr_pct"] / 100,
        axis=1,
    )
    df = df.sort_values("as_of").reset_index(drop=True)
    return df


def nifty_below_200dma_series() -> pd.Series | None:
    """Rolling-200-day-mean gate, same convention as
    scripts/factor_backtest.py's below_200dma. Returns a bool Series indexed
    by date (True = Nifty below its 200DMA that day), or None if the cache
    is missing (replay then falls back to no market-recovery escape route,
    same as before this fix -- disclosed, not silently degraded)."""
    if not NIFTY_CSV.exists():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_csv(NIFTY_CSV, index_col=0)
    # strip the yfinance multi-index CSV header artifact (mirrors
    # scripts/mine_big_movers.py:_load_nifty / scripts/backtest_events.py:trading_days)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df[df["Close"].notna()]
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    close = df["Close"]
    dma200 = close.rolling(200).mean()
    return close < dma200


def replay(trades: pd.DataFrame, *, reduced_pct: float | None, halted_pct: float | None,
           reset_pct: float | None, loss_streak_on: bool,
           nifty_below_200dma: pd.Series | None = None) -> dict:
    """
    Day-by-day replay:
      1. realize exits due today (cash += notional * (1 + return_pct/100))
      2. snapshot equity (open positions marked at cost -- see module docstring)
      3. compute circuit state from the snapshot history so far, WITH the
         same nifty_above_200dma escape valve the live system gets (see
         module docstring -- omitting this produced a single multi-year halt
         episode that never reset, on every threshold grid tested)
      4. if not halted/cooldown, take up to MAX_NEW_PER_DAY new trades (sorted
         by -score, ticker for determinism), sized via size_position with
         risk_multiplier=0.5 when reduced, respecting MAX_POSITIONS and cash
    """
    no_circuit = reduced_pct is None
    cash = CAPITAL
    open_positions: list[dict] = []
    equity_history: list[dict] = []
    closed_outcomes: list[str] = []  # 'win' | 'loss', chronological
    last_loss_date = None
    n_trades_taken = 0
    n_days_halted = 0
    n_days_reduced = 0

    all_dates = sorted(set(trades["as_of"]) | set(trades["exit_date"]))
    trades_by_day = {d: g for d, g in trades.groupby("as_of")}

    for today in all_dates:
        still_open = []
        for pos in open_positions:
            if pos["exit_date"] <= today:
                cash += pos["notional"] * (1 + pos["return_pct"] / 100)
                is_win = pos["return_pct"] > 0
                closed_outcomes.append("win" if is_win else "loss")
                if not is_win:
                    last_loss_date = today
            else:
                still_open.append(pos)
        open_positions = still_open

        equity = cash + sum(p["notional"] for p in open_positions)
        equity_history.append({"date": today.date().isoformat(), "equity": equity})

        if no_circuit:
            state = "normal"
        else:
            nifty_above_200dma = None
            if nifty_below_200dma is not None:
                below = nifty_below_200dma.asof(today)
                if below is not None and pd.notna(below):
                    nifty_above_200dma = not bool(below)
            ds = drawdown_state(equity_history, nifty_above_200dma=nifty_above_200dma,
                                 reduced_pct=reduced_pct, halted_pct=halted_pct, reset_pct=reset_pct)
            state = ds["state"]
        if state == "halted":
            n_days_halted += 1
        elif state == "reduced":
            n_days_reduced += 1

        cooldown = False
        if loss_streak_on and len(closed_outcomes) >= LOSS_STREAK_N and last_loss_date is not None:
            trailing = closed_outcomes[-LOSS_STREAK_N:]
            if all(o == "loss" for o in trailing):
                if (today - last_loss_date).days < LOSS_STREAK_COOLDOWN_DAYS:
                    cooldown = True

        if state != "halted" and not cooldown and today in trades_by_day:
            todays = trades_by_day[today].sort_values(["score", "ticker"], ascending=[False, True])
            taken_today = 0
            for _, row in todays.iterrows():
                if taken_today >= MAX_NEW_PER_DAY or len(open_positions) >= MAX_POSITIONS:
                    break
                risk_mult = 0.5 if state == "reduced" else 1.0
                pos_info = size_position(float(row["close"]), float(row["sl"]), capital=CAPITAL,
                                          score=row["score"], risk_multiplier=risk_mult)
                notional = pos_info["notional"]
                if notional <= 0 or notional > cash:
                    continue
                cash -= notional
                open_positions.append({
                    "ticker": row["ticker"], "notional": notional,
                    "exit_date": row["exit_date"], "return_pct": row["return_pct"],
                })
                taken_today += 1
                n_trades_taken += 1

    terminal_equity = cash + sum(p["notional"] for p in open_positions)
    eq_series = pd.Series([e["equity"] for e in equity_history])
    peak = eq_series.cummax()
    max_dd_pct = float(((eq_series - peak) / peak.replace(0, pd.NA) * 100).min()) if len(eq_series) else 0.0

    return {
        "terminal_equity": round(terminal_equity, 2),
        "total_return_pct": round((terminal_equity / CAPITAL - 1) * 100, 2),
        "max_dd_pct": round(max_dd_pct, 2) if max_dd_pct == max_dd_pct else 0.0,  # NaN guard
        "n_trades_taken": n_trades_taken,
        "n_days_halted": n_days_halted,
        "n_days_reduced": n_days_reduced,
        "n_days": len(equity_history),
    }


def main() -> None:
    trades = load_trades()
    print(f"[backtest_circuit] {len(trades)} closed trades, "
          f"{trades['as_of'].min().date()} -> {trades['as_of'].max().date()}")

    nifty_below_200dma = nifty_below_200dma_series()
    if nifty_below_200dma is None:
        print("[backtest_circuit] WARNING: cache/backtest_nifty.csv not found -- "
              "replaying WITHOUT the nifty-recovery escape valve from a halt "
              "(results will be pessimistic about halt duration)")

    results: dict[str, dict] = {}
    print(f"\n{'Config':<16}{'LossStreak':<12}{'Return%':>10}{'MaxDD%':>10}{'Trades':>9}{'Halted':>8}{'Reduced':>9}")
    for name, reduced_pct, halted_pct, reset_pct in GRID:
        for loss_streak_on in (True, False):
            if name == "no_circuit" and not loss_streak_on:
                key = "no_circuit_no_loss_streak"
            elif name == "no_circuit":
                key = "no_circuit_with_loss_streak"
            else:
                key = f"{name}{'_ls' if loss_streak_on else ''}"
            r = replay(trades, reduced_pct=reduced_pct, halted_pct=halted_pct,
                       reset_pct=reset_pct, loss_streak_on=loss_streak_on,
                       nifty_below_200dma=nifty_below_200dma)
            results[key] = r
            print(f"{name:<16}{str(loss_streak_on):<12}{r['total_return_pct']:>9.2f}%"
                  f"{r['max_dd_pct']:>9.2f}%{r['n_trades_taken']:>9}{r['n_days_halted']:>8}{r['n_days_reduced']:>9}")

    live_key = "10/15 (live)_ls"
    live = results[live_key]
    best_alt_key, best_alt = None, None
    for key, r in results.items():
        if key == live_key:
            continue
        if r["total_return_pct"] > live["total_return_pct"] and r["max_dd_pct"] > live["max_dd_pct"] - 1.0:
            if best_alt is None or r["total_return_pct"] > best_alt["total_return_pct"]:
                best_alt_key, best_alt = key, r

    if best_alt is not None:
        verdict = {
            "keep_live_thresholds": False,
            "reason": f"{best_alt_key} improves return ({best_alt['total_return_pct']}% > "
                      f"{live['total_return_pct']}%) without worsening MaxDD by >1pp "
                      f"({best_alt['max_dd_pct']}% vs {live['max_dd_pct']}%)",
        }
    else:
        verdict = {
            "keep_live_thresholds": True,
            "reason": "no alternative config improves BOTH terminal return and MaxDD "
                      "(>1pp better) over the live 10/15/8 thresholds -- status-quo bias "
                      "for a safety device is deliberate",
        }

    print(f"\n[backtest_circuit] verdict: {'KEEP live 10/15/8' if verdict['keep_live_thresholds'] else 'CHANGE thresholds'}")
    print(f"  {verdict['reason']}")

    out = {"results": results, "verdict": verdict}
    OUT_FILE.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[backtest_circuit] saved -> {OUT_FILE.name}")


if __name__ == "__main__":
    main()
