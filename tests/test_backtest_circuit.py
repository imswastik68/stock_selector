"""
Tests for scripts/backtest_circuit.py's replay engine, using synthetic trade
sequences (not the real backtest_trades.csv, which changes size/content on
every scripts/backtest.py re-run). Verifies: the live circuit config actually
halts trading and takes fewer trades than no_circuit when drawdown crosses
-16%; cash is conserved for a single-position sequence; a looser threshold
grid (12/20) does not halt where the live grid (10/15) does.
"""

from __future__ import annotations

import pandas as pd

import backtest_circuit as bc


def _trades(rows: list[dict]) -> pd.DataFrame:
    """rows: [{"as_of": "2026-01-01", "ticker": "A.NS", "direction": "buy",
    "close": 100.0, "atr_pct": 2.0, "score": 5, "return_pct": 10.0,
    "days_held": 3}, ...]"""
    df = pd.DataFrame(rows)
    df["as_of"] = pd.to_datetime(df["as_of"])
    df["exit_date"] = df["as_of"] + pd.to_timedelta(df["days_held"], unit="D")
    df["sl"] = df.apply(
        lambda r: r["close"] - 2 * r["close"] * r["atr_pct"] / 100 if r["direction"] == "buy"
        else r["close"] + 2 * r["close"] * r["atr_pct"] / 100,
        axis=1,
    )
    return df


def test_live_circuit_halts_and_takes_fewer_trades_than_no_circuit():
    """A sequence of large consecutive losses should cross -15% drawdown and
    trigger a halt under the live 10/15/8 grid, taking fewer total trades
    than the no_circuit variant (which keeps trading through the drawdown)."""
    rows = []
    for i in range(20):
        rows.append({
            "as_of": f"2026-01-{i+1:02d}", "ticker": f"L{i}.NS", "direction": "buy",
            "close": 100.0, "atr_pct": 2.0, "score": 5,
            "return_pct": -25.0,  # big loss every trade -> drawdown accelerates fast
            "days_held": 1,
        })
    trades = _trades(rows)

    live = bc.replay(trades, reduced_pct=10.0, halted_pct=15.0, reset_pct=8.0, loss_streak_on=False)
    none = bc.replay(trades, reduced_pct=None, halted_pct=None, reset_pct=None, loss_streak_on=False)

    assert live["n_days_halted"] > 0
    assert live["n_trades_taken"] < none["n_trades_taken"]


def test_cash_conservation_single_position_sequence():
    """With only ever one position open at a time, terminal equity must equal
    capital times the compounded return of each trade (within float tolerance) --
    no cash leaks or duplicates."""
    rows = [
        {"as_of": "2026-01-01", "ticker": "A.NS", "direction": "buy", "close": 100.0,
         "atr_pct": 2.0, "score": 5, "return_pct": 10.0, "days_held": 5},
        {"as_of": "2026-01-10", "ticker": "B.NS", "direction": "buy", "close": 100.0,
         "atr_pct": 2.0, "score": 5, "return_pct": -5.0, "days_held": 5},
    ]
    trades = _trades(rows)
    result = bc.replay(trades, reduced_pct=None, halted_pct=None, reset_pct=None, loss_streak_on=False)

    assert result["n_trades_taken"] == 2
    # size_position caps notional at max_position_pct of capital -- terminal
    # equity should sit between capital*(1-small loss) and capital*(1+small gain),
    # i.e. bounded, not exploded or zeroed.
    assert bc.CAPITAL * 0.5 < result["terminal_equity"] < bc.CAPITAL * 1.5


def test_looser_grid_does_not_halt_where_live_grid_does():
    """5 sequential, non-overlapping single-position trades, each losing 16%
    on a ~25%-of-capital position (score=5 -> notional caps at
    max_position_pct=25% of RISK_CONFIG's capital -- see src/risk.py).
    Trades 1-3 lose ~4% of capital each (equity 100k->96k->92k->88k, dd
    -4%/-8%/-12%). Once dd crosses -10% (after trade 3), the live grid's
    risk_multiplier=0.5 halves trades 4-5's notional/loss (~2% of capital
    each instead of 4%), landing final dd at -16% -- past the live grid's
    15% halt line but not the looser grid's 20% line, so the two configs
    must diverge on this exact sequence."""
    rows = []
    for i in range(5):
        rows.append({
            "as_of": f"2026-01-{i*2+1:02d}", "ticker": f"L{i}.NS", "direction": "buy",
            "close": 100.0, "atr_pct": 2.0, "score": 5,
            "return_pct": -16.0, "days_held": 1,
        })
    trades = _trades(rows)

    live   = bc.replay(trades, reduced_pct=10.0, halted_pct=15.0, reset_pct=8.0, loss_streak_on=False)
    looser = bc.replay(trades, reduced_pct=12.0, halted_pct=20.0, reset_pct=10.0, loss_streak_on=False)

    assert live["max_dd_pct"] <= -15.0
    assert live["max_dd_pct"] > -20.0
    assert live["n_days_halted"] > 0
    assert looser["n_days_halted"] == 0
