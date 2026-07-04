"""
Monthly momentum live track — parallel to the daily TA scanner, separate
capital, separate tracking (plan Phase 2).

Reuses scripts/factor_backtest.py's loaders/factors verbatim via sys.path
(same pattern tests/conftest.py already uses to `import factor_backtest`) --
duplicating that ~250 lines here would just drift out of sync with it.

GATE STATUS (read at runtime from outputs/factor_backtest.json, not
hardcoded): Phase 1's multi-split ship gate came back NO-SHIP for every
strategy tested (mom_12_1, mom_gated, hi_52w, low_vol, rs_quality,
composite) -- see the "Phase 1" commit. Per the plan's own fallback branch,
this script therefore runs PAPER-ONLY: it still builds and reports the book
every month (so a live paper track record accumulates), but does NOT deploy
real capital via src/portfolio.py. If a future re-run of factor_backtest.py
--validate ever flips mom_12_1 or mom_gated to "ships": true, this script
picks that up automatically on its next run and switches to LIVE sizing --
no code change needed, see _gate_status().

Book construction: point-in-time turnover gate (>=10cr/day) -> rank by
mom_12_1 -> top-20 equal-weight -> 100% cash if Nifty is below its 200DMA
(the same momentum-crash gate tested, and found not to help on its own, in
Phase 1 -- kept here anyway because a live paper track record is the only
way to see whether it earns its keep going forward, independent of the
backtest's verdict).

Cadence: intended to run once a month, on the first weekday of the
calendar month (approximation -- does not consult the NSE holiday
calendar, so if day 1 is a market holiday the "month" figures are marked
by whatever the cache's last row is, off by at most a day or two). The
monthly cron in .github/workflows/daily_scan.yml fires on days 1-3 of the
month; this script no-ops on the other two days. Use --force to run
regardless of the day check (also bypasses the idempotent
already-ran-this-month skip -- use for manual re-runs/testing only).

Usage:
  python scripts/factor_scan.py                 # normal monthly run
  python scripts/factor_scan.py --force          # ignore day-of-month + idempotency guards
  python scripts/factor_scan.py --sample 60      # fast sanity check, first 60 tickers
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("Run: pip install pandas numpy")

import factor_backtest as fb  # noqa: E402 (needs sys.path set up first)
from src.risk import RISK_CONFIG  # noqa: E402
from src.telegram_alert import send_telegram_alert  # noqa: E402

OUTPUTS = ROOT / "outputs"
FACTOR_PERF_FILE = OUTPUTS / "factor_performance.json"

TOP_N = 20
MOMENTUM_CAPITAL_PCT = min(0.20, max(0.10, float(os.environ.get("MOMENTUM_CAPITAL_PCT", "0.15"))))


# ── gate status (dynamic — reads Phase 1's own backtest output) ─────────────

def _gate_status() -> tuple[bool, str]:
    """
    True (LIVE) only if mom_12_1 or mom_gated actually cleared the
    multi-split ship gate in the most recent scripts/factor_backtest.py
    --validate run. Everything else, including no backtest output existing
    yet, defaults to PAPER-ONLY -- the safe default, not the exception.
    """
    bt_file = OUTPUTS / "factor_backtest.json"
    if not bt_file.exists():
        return False, "no factor_backtest.json found — defaulting to PAPER-ONLY"
    try:
        bt = json.loads(bt_file.read_text())
    except Exception:
        return False, "factor_backtest.json unreadable — defaulting to PAPER-ONLY"
    for name in ("mom_12_1", "mom_gated"):
        gate = bt.get("strategies", {}).get(name, {}).get("ship_gate_multi_split", {})
        if gate.get("ships"):
            return True, f"{name} passed the multi-split ship gate (outputs/factor_backtest.json)"
    return False, "no momentum strategy has passed the multi-split ship gate — PAPER-ONLY (outputs/factor_backtest.json)"


# ── book construction ────────────────────────────────────────────────────────

def build_book(sample: int | None = None):
    """Returns (ranked [(ticker, score), ...] top-N, prices {ticker: close},
    as_of Timestamp, below_200dma bool, nifty_close Series)."""
    universe = fb._load_universe(sample)
    ohlcv: dict[str, pd.DataFrame] = {}
    for t in universe:
        df = fb._load_ticker_df(t)
        if df is not None:
            ohlcv[t] = df
    if not ohlcv:
        sys.exit("[factor_scan] no cached OHLCV — run scripts/backtest.py first to populate cache/backtest_ohlcv/")

    nifty_df = fb._load_nifty()
    nifty_close = nifty_df["Close"].squeeze()
    dma200 = nifty_close.rolling(200).mean()
    below_200dma = bool(len(dma200.dropna()) and nifty_close.iloc[-1] < dma200.iloc[-1])
    as_of = nifty_close.index[-1]

    scores: dict[str, float] = {}
    prices: dict[str, float] = {}
    for ticker, full_df in ohlcv.items():
        if len(full_df) < 60 or not fb._turnover_ok(full_df):
            continue
        val = fb.factor_mom_12_1(full_df)
        if val is not None and pd.notna(val) and np.isfinite(val):
            scores[ticker] = float(val)
            prices[ticker] = float(full_df["Close"].iloc[-1])

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:TOP_N]
    return ranked, prices, as_of, below_200dma, nifty_close


def size_book(ranked: list[tuple[str, float]], prices: dict[str, float],
              below_200dma: bool, is_live: bool) -> dict:
    capital = RISK_CONFIG["capital"]
    momentum_capital = capital * MOMENTUM_CAPITAL_PCT

    if below_200dma:
        return {"status": "CASH — Nifty below 200DMA (momentum-crash gate)", "positions": []}
    if not ranked:
        return {"status": "CASH — no names cleared the turnover/history gate", "positions": []}

    per_name = momentum_capital / len(ranked)
    positions = []
    for ticker, score in ranked:
        price = prices.get(ticker, 0.0)
        qty = int(per_name / price) if price > 0 else 0
        if qty <= 0:
            continue
        positions.append({
            "ticker": ticker,
            "mom_12_1_score": round(score, 4),
            "price": round(price, 2),
            "qty": qty,
            "notional": round(qty * price, 2),
        })
    return {"status": "LIVE" if is_live else "PAPER", "positions": positions}


# ── month-over-month diff + performance tracking ────────────────────────────

def _month_files() -> list[Path]:
    return sorted(OUTPUTS.glob("factor_????-??.json"))


def _prior_book_data(month_key: str) -> dict | None:
    prior = [f for f in _month_files() if f.stem.split("_", 1)[1] < month_key]
    if not prior:
        return None
    try:
        return json.loads(prior[-1].read_text())
    except Exception:
        return None


def compute_diff(new_tickers: set[str], prior_tickers: set[str]) -> dict:
    return {
        "enter": sorted(new_tickers - prior_tickers),
        "exit": sorted(prior_tickers - new_tickers),
        "hold": sorted(new_tickers & prior_tickers),
    }


def _load_factor_perf() -> dict:
    if FACTOR_PERF_FILE.exists():
        try:
            return json.loads(FACTOR_PERF_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_factor_perf(perf: dict) -> None:
    FACTOR_PERF_FILE.parent.mkdir(parents=True, exist_ok=True)
    FACTOR_PERF_FILE.write_text(json.dumps(perf, indent=2, default=str))


def _update_factor_perf(month: str, book_ret_pct: float | None, nifty_ret_pct: float | None) -> dict:
    perf = _load_factor_perf()
    if month in perf:
        return perf  # already realized this month, don't double-count
    perf[month] = {
        "book_return_pct": book_ret_pct,
        "nifty_return_pct": nifty_ret_pct,
        "beat_nifty": (book_ret_pct is not None and nifty_ret_pct is not None
                       and book_ret_pct > nifty_ret_pct),
    }
    _save_factor_perf(perf)
    return perf


def promotion_signal(perf: dict) -> str:
    """3 consecutive months beating Nifty -> suggest scaling; trailing 6
    months all underperforming -> demote to paper. Advisory only -- printed
    and included in the Telegram alert, never auto-applied."""
    months = sorted(perf.keys())
    if len(months) >= 3 and all(perf[m]["beat_nifty"] for m in months[-3:]):
        return ("3 consecutive months beating Nifty -> consider scaling to 50% of capital "
                "(still requires a ship-gate PASS before sizing real money)")
    if len(months) >= 6 and not any(perf[m]["beat_nifty"] for m in months[-6:]):
        return "trailing 6 months all underperforming Nifty -> demote to paper / reduce further"
    return "no promotion/demotion trigger yet"


def _is_first_weekday_of_month(today: date) -> bool:
    """Approximation, not the NSE trading calendar (see module docstring)."""
    if today.weekday() >= 5:
        return False
    d = today.replace(day=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d == today


# ── main ──────────────────────────────────────────────────────────────────────

def main(sample: int | None = None, force: bool = False) -> int:
    today = date.today()
    if not force and not _is_first_weekday_of_month(today):
        print(f"[factor_scan] {today.isoformat()} is not the first weekday of the month "
              f"— skipping (use --force to override)")
        return 0

    month_key = today.strftime("%Y-%m")
    out_file = OUTPUTS / f"factor_{month_key}.json"
    if out_file.exists() and not force:
        print(f"[factor_scan] {out_file.name} already exists this month — idempotent skip "
              f"(use --force to recompute)")
        send_telegram_alert(json.loads(out_file.read_text()), mode="factor")
        return 0

    is_live, reason = _gate_status()
    print(f"[factor_scan] gate status: {'LIVE' if is_live else 'PAPER-ONLY'} — {reason}")

    ranked, prices, as_of, below_200dma, nifty_close = build_book(sample)
    print(f"[factor_scan] as_of={as_of.date()}  below_200dma={below_200dma}  "
          f"{len(ranked)} names ranked (turnover+history gated)")

    book = size_book(ranked, prices, below_200dma, is_live)
    new_tickers = {p["ticker"] for p in book["positions"]}

    prior_data = _prior_book_data(month_key)
    prior_tickers = {p["ticker"] for p in prior_data["book"]["positions"]} if prior_data else set()
    diff = compute_diff(new_tickers, prior_tickers)

    prior_month_realized = {"month": None, "book_return_pct": None, "nifty_return_pct": None}
    signal = "insufficient history for promotion/demotion evaluation"
    if prior_data:
        prior_positions = prior_data["book"]["positions"]
        rets = [
            prices[p["ticker"]] / p["price"] - 1
            for p in prior_positions
            if p["ticker"] in prices and p["price"] > 0
        ]
        book_ret_pct = round(float(np.mean(rets)) * 100, 2) if rets else None

        p0 = fb._price_asof(nifty_close, pd.Timestamp(prior_data["as_of"]))
        p1 = float(nifty_close.iloc[-1])
        nifty_ret_pct = round((p1 / p0 - 1) * 100, 2) if (p0 and p0 > 0) else None

        prior_month_realized = {
            "month": prior_data["month"], "book_return_pct": book_ret_pct, "nifty_return_pct": nifty_ret_pct,
        }
        if book_ret_pct is not None:
            perf = _update_factor_perf(prior_data["month"], book_ret_pct, nifty_ret_pct)
            signal = promotion_signal(perf)
            print(f"[factor_scan] {prior_data['month']} realized: book {book_ret_pct:+.2f}% "
                  f"vs Nifty {nifty_ret_pct:+.2f}%  -> {signal}")

    out_data = {
        "month": month_key,
        "as_of": as_of.date().isoformat(),
        "gate_status": {"is_live": is_live, "reason": reason},
        "below_200dma": below_200dma,
        "book": book,
        "diff": diff,
        "momentum_capital_pct": MOMENTUM_CAPITAL_PCT,
        "capital_allocated": round(RISK_CONFIG["capital"] * MOMENTUM_CAPITAL_PCT, 2),
        "prior_month_realized": prior_month_realized,
        "promotion_signal": signal,
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out_data, indent=2, default=str))
    print(f"[factor_scan] saved -> {out_file.name}")

    print(f"\n[factor_scan] === {book['status']} ({len(book['positions'])} names) ===")
    for i, p in enumerate(book["positions"], 1):
        print(f"  {i:>2}. {p['ticker']:<20} qty={p['qty']:<6} @₹{p['price']:.2f} "
              f"= ₹{p['notional']:,.0f}  mom_12_1={p['mom_12_1_score']:+.3f}")
    if diff["enter"]:
        print(f"[factor_scan] ENTER: {', '.join(diff['enter'])}")
    if diff["exit"]:
        print(f"[factor_scan] EXIT:  {', '.join(diff['exit'])}")

    send_telegram_alert(out_data, mode="factor")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="Limit universe to first N tickers (fast sanity check)")
    ap.add_argument("--force", action="store_true",
                     help="Ignore the day-of-month + already-ran-this-month guards (manual re-run/testing)")
    args = ap.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.exit(main(sample=args.sample, force=args.force))
