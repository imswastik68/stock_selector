"""Reset the paper book to a clean capital base.

Why this exists: outputs/portfolio.json was the old ~Rs 1L lineage (equity
~101k, cash Rs 11.4k) while CI runs RISK_CAPITAL=1000000, so
src/risk.py:size_position returned Rs 125k-150k notional per pick and
src/portfolio.py:open_positions skipped EVERY pick on `cash < notional`.
Zero positions opened between 2026-07-17 and 2026-09-17 while 57 buys were
emitted in September alone -- the live-proof book had stopped tracking the
system entirely.

The old state is archived, never deleted, so the closed-trade history stays
auditable. Open holdings are marked to their last known price and moved to
`closed` with outcome "reset" so the archived P&L stays complete.

Usage:
    python scripts/reset_portfolio.py                 # reset to RISK_CONFIG capital
    python scripts/reset_portfolio.py --capital 500000
    python scripts/reset_portfolio.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STATE = ROOT / "outputs" / "portfolio.json"
ARCHIVE_DIR = ROOT / "outputs" / "portfolio_archive"


def main() -> None:
    from src.risk import RISK_CONFIG

    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=RISK_CONFIG["capital"],
                    help="starting cash (default: RISK_CONFIG['capital'])")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    today = date.today().isoformat()
    old = json.loads(STATE.read_text()) if STATE.exists() else {}

    holdings = old.get("holdings", {})
    print(f"old book: equity Rs {old.get('equity', 0):,.0f}  "
          f"cash Rs {old.get('cash', 0):,.0f}  "
          f"{len(holdings)} open  {len(old.get('closed', []))} closed")
    for t, h in holdings.items():
        last = h.get("last_price") or h["entry"]
        print(f"  flatten {t:<18} qty={h['qty']:<5} entry={h['entry']:.2f} "
              f"last={last:.2f} unreal=Rs {h.get('unrealized_pnl', 0):+,.0f}")
    print(f"new book: cash = equity = Rs {args.capital:,.0f}")

    if args.dry_run:
        print("dry-run: nothing written")
        return

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    dest = ARCHIVE_DIR / f"portfolio_{today}.json"
    if STATE.exists():
        shutil.copy2(STATE, dest)
        print(f"archived -> {dest.relative_to(ROOT)}")

    # Flatten open holdings into `closed` at last mark so the archive is complete.
    closed = list(old.get("closed", []))
    for ticker, h in holdings.items():
        last = float(h.get("last_price") or h["entry"])
        entry, qty = float(h["entry"]), int(h["qty"])
        pnl = (last - entry) * qty if h["direction"] == "buy" else (entry - last) * qty
        closed.append({
            "ticker": ticker, "qty": qty, "entry": entry, "exit_price": last,
            "direction": h["direction"], "opened": h["opened"], "closed": today,
            "pnl": round(pnl, 2),
            "return_pct": round((last / entry - 1) * 100, 2) if entry else 0.0,
            "outcome": "reset",
        })

    state = {
        "cash": args.capital,
        "equity": args.capital,
        "realized_pnl": 0.0,
        "holdings": {},
        "closed": closed,
        "equity_history": [{"date": today, "equity": args.capital}],
        "reset_on": today,
        "reset_capital": args.capital,
    }
    STATE.write_text(json.dumps(state, indent=2, default=str))
    print(f"wrote {STATE.relative_to(ROOT)}: flat book, Rs {args.capital:,.0f} cash, "
          f"{len(closed)} closed trades retained")


if __name__ == "__main__":
    main()
