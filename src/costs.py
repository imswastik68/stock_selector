"""
Round-trip transaction cost model — STT + brokerage + exchange/DP charges +
slippage, as a percent of notional. No cost modeling existed anywhere before
this (grep-confirmed); every backtest/live return number was gross.

Defaults are round working numbers, not a broker-specific breakdown: equity
delivery carries STT (0.1% on the sell leg alone), exchange + DP charges,
and worse slippage on illiquid SMID names; stock futures carry no delivery
STT and tighter spreads on F&O-liquid names, hence the lower default.
"""

from __future__ import annotations

import os

EQUITY_DELIVERY_RT_PCT = float(os.environ.get("COST_EQUITY_RT_PCT", "0.30"))
FUTURES_RT_PCT         = float(os.environ.get("COST_FUTURES_RT_PCT", "0.15"))


def round_trip_cost_pct(direction: str) -> float:
    """
    Round-trip cost as a percent of notional.
    direction: 'buy' -> cash equity delivery, 'sell' -> stock future (F&O-gated, see Phase 3).
    """
    return FUTURES_RT_PCT if direction == "sell" else EQUITY_DELIVERY_RT_PCT
