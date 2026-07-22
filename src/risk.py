"""
Advisory position sizing and portfolio risk summary.

Default: capital ₹10,00,000, risk 1% per trade (₹10,000).
Env overrides: RISK_CAPITAL, RISK_PER_TRADE_PCT,
               RISK_MAX_POSITION_PCT, RISK_MAX_PORTFOLIO_PCT, RISK_MAX_POSITIONS

Example (entry ₹250, SL ₹238):
  risk/share = ₹12 → shares = floor(10000/12) = 833
  notional   = 833 × ₹250 = ₹2,08,250  (20.8% of capital, under 25% cap)
  risk_amount= 833 × ₹12  = ₹9,996     (≈ 1.0% of capital)
"""

from __future__ import annotations

import math
import os

RISK_CONFIG = {
    "capital":            float(os.environ.get("RISK_CAPITAL",            "1000000")),
    "risk_per_trade_pct": float(os.environ.get("RISK_PER_TRADE_PCT",      "0.01")),
    "max_position_pct":   float(os.environ.get("RISK_MAX_POSITION_PCT",   "0.25")),
    "max_portfolio_pct":  float(os.environ.get("RISK_MAX_PORTFOLIO_PCT",  "0.06")),
    "max_positions":      int(os.environ.get("RISK_MAX_POSITIONS",        "8")),
    "max_per_sector":     int(os.environ.get("RISK_MAX_PER_SECTOR",       "3")),
    # "other" (unmapped SME/unknown) gets a looser cap than named sectors, not
    # unlimited — nearly all SMID longs land here (sector_map.json only covers
    # ~175 tickers), so a 3-per-sector cap would strangle the long book.
    "max_other":          int(os.environ.get("RISK_MAX_OTHER",           "4")),
}


# Conviction scaling: only applied when score predicts win magnitude (run --score-magnitude first).
# Enabled 2026-07 (Phase 5): 156-week/185,673-closed-trade backtest shows a genuinely
# monotone relationship across all 3 metrics, not just avg_win — WR 41.5%->42.1%->42.3%,
# avg_win 12.126%->12.127%->12.492%, avg_return -0.540%->-0.500%->-0.281% across score
# tiers <5/5-7/>=8. The earlier 52-week bear-market-only window (commit 9f4b89e's
# analysis) only showed avg_win technically monotone while WR was flat/inverted --
# this is a materially stronger result on 12x the sample across 3 market regimes.
CONVICTION_SIZING: bool = True

# Risk % tiers by score (used only when CONVICTION_SIZING is True)
_SCORE_TIERS: list[tuple[int, float]] = [
    (8,  0.0125),  # score ≥ 8 → 1.25%
    (5,  0.0100),  # score 5–7 → 1.0%
    (0,  0.0075),  # score < 5 → 0.75%
]


def _score_to_risk_pct(score: int | float | None) -> float:
    if not CONVICTION_SIZING or score is None:
        return RISK_CONFIG["risk_per_trade_pct"]
    for threshold, pct in _SCORE_TIERS:
        if score >= threshold:
            return pct
    return RISK_CONFIG["risk_per_trade_pct"]


def size_position(
    entry: float,
    sl: float,
    *,
    capital: float | None = None,
    risk_pct: float | None = None,
    max_position_pct: float | None = None,
    score: int | float | None = None,
    risk_multiplier: float = 1.0,
) -> dict:
    """
    Fixed-fractional position sizing.

    shares = floor(capital × risk_pct × risk_multiplier / risk_per_share)
    Capped when notional > capital × max_position_pct.

    If CONVICTION_SIZING is True and score is provided, risk_pct is scaled
    by score tier (run --score-magnitude to verify monotone signal first).

    risk_multiplier (Phase 4): scales the resolved risk_pct, e.g. 0.5 when
    drawdown_state() reports "reduced". Default 1.0 -- no effect unless the
    caller passes something else (main.py, wired to the circuit breaker).

    Returns dict: shares, notional, risk_amount, risk_pct_actual, position_pct, capped
    """
    cap  = capital          if capital          is not None else RISK_CONFIG["capital"]
    rp   = (risk_pct if risk_pct is not None else _score_to_risk_pct(score)) * risk_multiplier
    maxp = max_position_pct if max_position_pct is not None else RISK_CONFIG["max_position_pct"]

    risk_per_share = abs(entry - sl)
    if risk_per_share <= 0 or entry <= 0:
        return {"shares": 0, "notional": 0.0, "risk_amount": 0.0,
                "risk_pct_actual": 0.0, "position_pct": 0.0, "capped": False}

    risk_budget  = cap * rp
    shares       = int(risk_budget / risk_per_share)
    capped       = False
    max_notional = cap * maxp

    if shares > 0 and shares * entry > max_notional:
        shares = int(max_notional / entry)
        capped = True

    if shares <= 0:
        return {"shares": 0, "notional": 0.0, "risk_amount": 0.0,
                "risk_pct_actual": 0.0, "position_pct": 0.0, "capped": capped}

    notional        = round(shares * entry, 2)
    risk_amount     = round(shares * risk_per_share, 2)
    risk_pct_actual = round(risk_amount / cap * 100, 2)
    position_pct    = round(notional   / cap * 100, 1)

    return {
        "shares":          shares,
        "notional":        notional,
        "risk_amount":     risk_amount,
        "risk_pct_actual": risk_pct_actual,
        "position_pct":    position_pct,
        "capped":          capped,
    }


def portfolio_summary(
    sized_picks: list[dict],
    *,
    capital: float | None = None,
    max_portfolio_risk_pct: float | None = None,
    max_positions: int | None = None,
    max_per_sector: int | None = None,
    max_other: int | None = None,
) -> dict:
    """
    Rank picks by score, allocate until risk budget, max_positions, or sector cap hit.
    Mutates each pick: adds pick["allocate"] = True/False, pick["drop_reason"] if dropped.

    Sector cap: pick["sector"] must be set (by main.py from build_sector_map);
    "other" (unmapped/SME) uses the looser max_other cap, not unlimited.

    Returns: capital, total_deployed, total_deployed_pct, total_risk,
             total_risk_pct, budget_left, n_allocated, n_dropped, n_sector_capped
    """
    cap        = capital               if capital               is not None else RISK_CONFIG["capital"]
    max_rp     = max_portfolio_risk_pct if max_portfolio_risk_pct is not None else RISK_CONFIG["max_portfolio_pct"]
    max_pos    = max_positions          if max_positions          is not None else RISK_CONFIG["max_positions"]
    max_sec    = max_per_sector         if max_per_sector         is not None else RISK_CONFIG["max_per_sector"]
    max_oth    = max_other              if max_other              is not None else RISK_CONFIG["max_other"]

    risk_budget    = cap * max_rp
    total_deployed = 0.0
    total_risk     = 0.0
    n_allocated    = 0
    n_dropped      = 0
    n_sector_capped = 0
    sector_counts: dict[str, int] = {}

    picks = sorted(sized_picks, key=lambda x: x.get("score", 0), reverse=True)

    for pick in picks:
        pos    = pick.get("position") or {}
        risk   = pos.get("risk_amount") or 0.0
        notl   = pos.get("notional")   or 0.0
        sector = pick.get("sector", "other") or "other"

        budget_ok   = (total_risk + risk) <= risk_budget
        position_ok = n_allocated < max_pos
        sector_ok   = sector_counts.get(sector, 0) < (max_oth if sector == "other" else max_sec)

        if budget_ok and position_ok and sector_ok and risk > 0:
            pick["allocate"] = True
            total_deployed  += notl
            total_risk         += risk
            n_allocated        += 1
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        else:
            pick["allocate"] = False
            if not sector_ok:
                pick["drop_reason"] = f"sector cap ({sector})"
                n_sector_capped += 1
            elif not budget_ok:
                pick["drop_reason"] = "budget full"
            elif not position_ok:
                pick["drop_reason"] = "max positions"
            else:
                pick["drop_reason"] = "no sizing"
            n_dropped += 1

    return {
        "capital":            cap,
        "total_deployed":     round(total_deployed, 2),
        "total_deployed_pct": round(total_deployed / cap * 100, 1) if cap else 0.0,
        "total_risk":         round(total_risk, 2),
        "total_risk_pct":     round(total_risk / cap * 100, 2) if cap else 0.0,
        "budget_left":        round(risk_budget - total_risk, 2),
        "n_allocated":        n_allocated,
        "n_dropped":          n_dropped,
        "n_sector_capped":    n_sector_capped,
    }


# ── circuit breaker (Phase 4) ─────────────────────────────────────────────────

DRAWDOWN_REDUCED_PCT = 10.0   # >10% from peak -> halve risk_per_trade_pct for new positions
DRAWDOWN_HALTED_PCT  = 15.0   # >15% from peak -> no new positions
DRAWDOWN_RESET_PCT   = 8.0    # once halted, stays halted until DD recovers to <=8% (or Nifty > 200DMA)


def drawdown_state(
    equity_history: list[dict],
    nifty_above_200dma: bool | None = None,
    *,
    reduced_pct: float = DRAWDOWN_REDUCED_PCT,
    halted_pct: float = DRAWDOWN_HALTED_PCT,
    reset_pct: float = DRAWDOWN_RESET_PCT,
) -> dict:
    """
    Book-level circuit breaker off a chronological equity curve (src.portfolio's
    daily snapshots: [{"date", "equity"}, ...]).

      normal:  drawdown from peak <= reduced_pct   -> full size
      reduced: reduced_pct < drawdown <= halted_pct -> halve risk_per_trade_pct for new entries
      halted:  drawdown > halted_pct                -> no new positions

    Hysteresis: once drawdown has breached -halted_pct, the state stays
    "halted" even as it recovers back through the reduced/halted band, until
    EITHER drawdown recovers to <=reset_pct OR nifty_above_200dma is True --
    otherwise the state would flap in and out of "halted" right around the
    halted_pct line every time the book ticks a fraction either side of it.

    reduced_pct/halted_pct/reset_pct default to the live thresholds (module
    constants above) -- kwargs exist so scripts/backtest_circuit.py (Phase 5)
    can replay alternative threshold grids without duplicating this function.
    """
    equities = [float(e["equity"]) for e in equity_history if e.get("equity") is not None]
    if not equities:
        return {"state": "normal", "drawdown_pct": 0.0, "peak_equity": None}

    peak = equities[0]
    dd_series = []
    for eq in equities:
        peak = max(peak, eq)
        dd_series.append((eq / peak - 1) * 100 if peak > 0 else 0.0)

    current_dd, current_peak = dd_series[-1], peak

    breached_halted_since_reset = False
    for dd in reversed(dd_series):
        if dd <= -halted_pct:
            breached_halted_since_reset = True
        if dd >= -reset_pct:
            break

    if breached_halted_since_reset and not nifty_above_200dma:
        state = "halted"
    elif current_dd <= -reduced_pct:
        state = "reduced"
    else:
        state = "normal"

    return {
        "state": state,
        "drawdown_pct": round(current_dd, 2),
        "peak_equity": round(current_peak, 2),
    }
