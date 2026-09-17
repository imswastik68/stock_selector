"""MAE/MFE must cover the holding period only, not the rest of the dataframe.

`post_entry` runs from entry to the END of the supplied OHLCV, so scanning it
for min/max measured excursions that happened after the trade had already
exited. Symptoms seen 2026-09-17 on outputs/backtest_trades.csv: short trades
reporting mean mae_pct of -110% (median -44%), and losing trades appearing to
have been +10% in profit before reversing -- an artifact of post-exit bars, not
something the trade ever experienced.

return_pct/outcome were never affected (they break at the exit bar); this is a
diagnostics-only field, but it is the field an exit-policy investigation leans
on, so a wrong value argues for changes that the realised numbers do not
support.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.trade_sim import simulate_raw


def _bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """rows = (open, high, low, close), one per trading day."""
    idx = pd.date_range("2026-01-02", periods=len(rows), freq="B")
    return pd.DataFrame(
        {
            "Open":  [r[0] for r in rows],
            "High":  [r[1] for r in rows],
            "Low":   [r[2] for r in rows],
            "Close": [r[3] for r in rows],
            "Volume": [1_000_000] * len(rows),
        },
        index=idx,
    )


def test_long_mfe_ignores_rally_after_the_stop_was_hit():
    """Enter at 100, stop at 95. Price dips to 94 (stop hit on bar 2), then the
    name triples. MFE must reflect the trade, not the post-exit moonshot."""
    df = _bars([
        (100, 101, 99, 100),    # bar 1: entry zone touched, fills at 100
        (99,  100, 94, 95),     # bar 2: low 94 -> stop at 95 hit, trade over
        (95,  300, 95, 300),    # bar 3: +200% AFTER the exit
        (300, 400, 300, 400),   # bar 4
    ])
    out = simulate_raw(
        entry_lo=99.5, entry_hi=100.5, entry_mid=100.0,
        sl=95.0, t1=120.0, direction="buy",
        as_of_date=df.index[0] - pd.Timedelta(days=1), df=df,
    )
    assert out["outcome"] == "sl_hit"
    # bars 1-2 only: high 101 -> MFE +1%, low 94 -> MAE -6%
    assert out["mfe_pct"] == pytest.approx(1.0, abs=0.01)
    assert out["mae_pct"] == pytest.approx(-6.0, abs=0.01)


def test_short_mae_is_bounded_by_the_holding_period():
    """A short stopped out on bar 2 must not report the adverse excursion of a
    later 4x. This is the -110% MAE case from backtest_trades.csv."""
    df = _bars([
        (100, 101, 99,  100),   # entry fills at 100 (short)
        (101, 106, 100, 105),   # high 106 -> stop at 105 hit, trade over
        (105, 400, 105, 400),   # +300% AFTER the exit
    ])
    out = simulate_raw(
        entry_lo=99.5, entry_hi=100.5, entry_mid=100.0,
        sl=105.0, t1=90.0, direction="sell",
        as_of_date=df.index[0] - pd.Timedelta(days=1), df=df,
    )
    assert out["outcome"] == "sl_hit"
    # worst against a short over bars 1-2 is high 106 -> -6%, never -300%
    assert out["mae_pct"] == pytest.approx(-6.0, abs=0.01)
    assert out["mae_pct"] > -50, "MAE leaked post-exit bars"


def test_open_trade_still_measures_to_the_last_bar():
    """A trade that never exits keeps the full window -- the fix must not
    truncate positions that are genuinely still open."""
    df = _bars([
        (100, 101, 99,  100),
        (100, 110, 99,  108),
        (108, 115, 107, 112),
    ])
    out = simulate_raw(
        entry_lo=99.5, entry_hi=100.5, entry_mid=100.0,
        sl=80.0, t1=200.0, direction="buy",   # neither ever hit
        as_of_date=df.index[0] - pd.Timedelta(days=1), df=df,
    )
    assert out["outcome"] == "open"
    assert out["mfe_pct"] == pytest.approx(15.0, abs=0.01)   # high 115
    assert out["mae_pct"] == pytest.approx(-1.0, abs=0.01)   # low 99
