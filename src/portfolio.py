"""
Capital-constrained live portfolio simulator.

Maintains persistent state in outputs/portfolio.json:
  {
    "cash":          float,        # ₹ available to deploy
    "equity":        float,        # cash + mark-to-market value of holdings
    "realized_pnl":  float,        # cumulative closed P&L
    "holdings": {
      "TICKER.NS": {
        "qty":          int,
        "entry":        float,
        "sl":           float,
        "t1":           float,
        "t2":           float | null,
        "direction":    "buy" | "sell",
        "opened":       "YYYY-MM-DD",
        "exit_policy":  str,
        "last_price":   float | null,   # updated by mark_to_market
        "unrealized_pnl": float | null,
      }, ...
    },
    "closed": [
      {"ticker", "qty", "entry", "exit_price", "direction",
       "opened", "closed", "pnl", "return_pct", "outcome"}
    ],
    "equity_history": [{"date": "YYYY-MM-DD", "equity": float}, ...]  # capped, see _MAX_EQUITY_HISTORY
  }

NOTE: This tracks the capital-constrained live book (actual P&L + cash lifecycle).
      src/performance.py tracks signal hit-rate over all generated picks (advisory).
      These are intentionally separate concerns.

Workflow per EOD run (called in main.py):
  1. process_exits(today_bars)   — realize P&L, free cash, move to closed
  2. open_positions(alloc_picks) — deploy cash into new allocate=True picks
  3. mark_to_market(today_closes)— update unrealized P&L + equity
"""

from __future__ import annotations

import json
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

from src.trade_sim import WINNER_POLICY, simulate_trade
from src.costs import round_trip_cost_pct

_STATE_FILE = Path(__file__).parent.parent / "outputs" / "portfolio.json"
_INITIAL_CAPITAL = 100_000.0  # default; overridden by RISK_CAPITAL env if set
_MAX_EQUITY_HISTORY = 400  # ~1.5y of daily snapshots -- plenty for drawdown_state's peak-tracking


def _load() -> dict:
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text())
        except Exception:
            pass
    # First-run default
    import os
    capital = float(os.environ.get("RISK_CAPITAL", str(_INITIAL_CAPITAL)))
    return {
        "cash":         capital,
        "equity":       capital,
        "realized_pnl": 0.0,
        "holdings":     {},
        "closed":       [],
        "equity_history": [],
    }


def _save(state: dict) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def process_exits(today_bars: dict[str, pd.DataFrame]) -> list[dict]:
    """
    For each open holding, run simulate_trade against today_bars.
    On exit: realize P&L, free cash, move to closed list.
    Returns list of closed trade dicts.
    """
    state   = _load()
    closed  = []
    today   = date.today().isoformat()

    for ticker, holding in list(state["holdings"].items()):
        df = today_bars.get(ticker)
        if df is None or df.empty:
            continue

        pick = {
            "date":      holding["opened"],
            "direction": holding["direction"],
            "entry_mid": holding["entry"],
            "entry_lo":  holding.get("entry_lo", holding["entry"]),
            "entry_hi":  holding.get("entry_hi", holding["entry"]),
            "sl":        holding["sl"],
            "t1":        holding["t1"],
            "t2":        holding.get("t2"),
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = simulate_trade(pick, df, exit_policy=holding.get("exit_policy", WINNER_POLICY))

        if not sim.get("triggered") or sim.get("outcome") == "open":
            continue  # still open

        exit_price = float(sim["exit_price"])
        qty        = holding["qty"]
        entry      = holding["entry"]
        direction  = holding["direction"]

        if direction == "buy":
            pnl = (exit_price - entry) * qty
            ret_pct = (exit_price - entry) / entry * 100
        else:
            pnl = (entry - exit_price) * qty
            ret_pct = (entry - exit_price) / entry * 100

        cost_pct = round_trip_cost_pct(direction)   # % of notional, see src/costs.py
        cost_amt = qty * entry * cost_pct / 100
        pnl -= cost_amt
        ret_pct -= cost_pct

        state["cash"]         += qty * entry + pnl
        state["realized_pnl"] = round(state["realized_pnl"] + pnl, 2)

        closed_rec = {
            "ticker":     ticker,
            "qty":        qty,
            "entry":      entry,
            "exit_price": round(exit_price, 2),
            "direction":  direction,
            "opened":     holding["opened"],
            "closed":     today,
            "pnl":        round(pnl, 2),
            "return_pct": round(ret_pct, 2),
            "cost_pct":   round(cost_pct, 2),
            "outcome":    sim.get("outcome"),
        }
        state["closed"].append(closed_rec)
        closed.append(closed_rec)
        del state["holdings"][ticker]
        print(f"[portfolio] EXIT  {ticker:<20} {sim['outcome']:<10} "
              f"pnl=₹{pnl:+,.0f} ({ret_pct:+.1f}%)  cash→₹{state['cash']:,.0f}")

    _save(state)
    return closed


def open_positions(allocated_picks: list[dict]) -> list[str]:
    """
    For allocate=True picks not already held, deploy cash if sufficient.
    Returns list of tickers opened.
    """
    state  = _load()
    opened = []
    today  = date.today().isoformat()

    for pick in allocated_picks:
        if not pick.get("allocate"):
            continue
        ticker = pick.get("ticker", "")
        if not ticker or ticker in state["holdings"]:
            continue  # already held

        pos     = pick.get("position") or {}
        qty     = pos.get("shares", 0)
        notional = pos.get("notional", 0.0)
        entry   = pick.get("today_close") or 0.0
        sl      = None
        t1      = None
        t2      = None

        # Parse levels from formatted strings
        def _p(s) -> float | None:
            try:
                return float(str(s).replace("₹", "").replace(",", "").strip())
            except (ValueError, TypeError):
                return None

        sl  = _p(pick.get("stop_loss"))
        t1  = _p(pick.get("target_1"))
        t2  = _p(pick.get("target_2"))

        if qty <= 0 or notional <= 0 or not entry or not sl or not t1:
            continue
        if state["cash"] < notional:
            print(f"[portfolio] SKIP  {ticker:<20} insufficient cash "
                  f"(need ₹{notional:,.0f}, have ₹{state['cash']:,.0f})")
            continue

        state["cash"] -= notional
        state["holdings"][ticker] = {
            "qty":        qty,
            "entry":      float(entry),
            "entry_lo":   float(entry),
            "entry_hi":   float(entry),
            "sl":         float(sl),
            "t1":         float(t1),
            "t2":         float(t2) if t2 else None,
            "direction":  pick.get("direction", "buy"),
            "opened":     today,
            "exit_policy": WINNER_POLICY,
            "last_price":  float(entry),
            "unrealized_pnl": 0.0,
        }
        opened.append(ticker)
        print(f"[portfolio] OPEN  {ticker:<20} qty={qty}  notional=₹{notional:,.0f}  "
              f"cash→₹{state['cash']:,.0f}")

    _save(state)
    return opened


def mark_to_market(today_closes: dict[str, float]) -> dict:
    """
    Update unrealized P&L for all holdings using today's closes.
    Recomputes equity = cash + Σ(qty × last_price).
    Returns summary dict.
    """
    state       = _load()
    total_mkt   = 0.0
    total_unreal = 0.0

    for ticker, holding in state["holdings"].items():
        price = today_closes.get(ticker)
        if price is None:
            price = holding.get("last_price") or holding["entry"]
        entry     = holding["entry"]
        qty       = holding["qty"]
        direction = holding["direction"]

        if direction == "buy":
            unreal = (float(price) - entry) * qty
        else:
            unreal = (entry - float(price)) * qty

        holding["last_price"]     = round(float(price), 2)
        holding["unrealized_pnl"] = round(unreal, 2)
        total_mkt    += entry * qty + unreal
        total_unreal += unreal

    state["equity"] = round(state["cash"] + total_mkt, 2)

    # Daily equity snapshot for src/risk.py's drawdown_state circuit breaker.
    # One entry per calendar date -- re-running the same day (e.g. a retry)
    # updates today's entry rather than appending a duplicate.
    today_iso = date.today().isoformat()
    history = state.setdefault("equity_history", [])
    if history and history[-1]["date"] == today_iso:
        history[-1]["equity"] = state["equity"]
    else:
        history.append({"date": today_iso, "equity": state["equity"]})
    if len(history) > _MAX_EQUITY_HISTORY:
        del history[:-_MAX_EQUITY_HISTORY]

    _save(state)

    return {
        "cash":            round(state["cash"], 2),
        "equity":          state["equity"],
        "realized_pnl":    state["realized_pnl"],
        "unrealized_pnl":  round(total_unreal, 2),
        "n_holdings":      len(state["holdings"]),
        "holdings":        list(state["holdings"].keys()),
    }


def equity_history() -> list[dict]:
    """Chronological [{"date", "equity"}, ...] snapshots for drawdown_state()."""
    return _load().get("equity_history", [])


def summary() -> dict:
    """Return current portfolio state without modifying it."""
    state = _load()
    unreal = sum(
        h.get("unrealized_pnl", 0.0) for h in state["holdings"].values()
    )
    return {
        "cash":           round(state["cash"], 2),
        "equity":         round(state["equity"], 2),
        "realized_pnl":   round(state["realized_pnl"], 2),
        "unrealized_pnl": round(unreal, 2),
        "n_holdings":     len(state["holdings"]),
        "holdings":       list(state["holdings"].keys()),
    }
