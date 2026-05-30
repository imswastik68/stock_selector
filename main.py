"""
Stock Selector — Daily Orchestrator
Runs at 6:30 PM IST (13:00 UTC) via GitHub Actions.

Pipeline: fetch data → score candidates → Claude synthesis → Telegram alert → save JSON
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.data.bulk_deals import fetch_bulk_deals
from src.data.breakouts import fetch_breakouts
from src.data.fo_ban import fetch_fo_ban_delta
from src.data.results import fetch_results_calendar
from src.data.volume import fetch_volume_gainers
from src.scorer import score_candidates
from src.agent import synthesize_watchlist
from src.telegram_alert import send_telegram_alert

OUTPUTS_DIR = Path(__file__).parent / "outputs"


def fetch_all_data() -> tuple[list, list, list, list, list]:
    """Fetch all five data sources concurrently."""
    print("[main] fetching market data...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            "bulk_deals": pool.submit(fetch_bulk_deals),
            "volume": pool.submit(fetch_volume_gainers),
            "fo_ban": pool.submit(fetch_fo_ban_delta),
            "results": pool.submit(fetch_results_calendar),
            "breakouts": pool.submit(fetch_breakouts),
        }

        results = {}
        for name, future in futures.items():
            try:
                results[name] = future.result()
            except Exception as exc:
                print(f"[main] {name} fetch error (continuing): {exc}")
                results[name] = []

    elapsed = time.time() - t0
    print(f"[main] all data fetched in {elapsed:.1f}s")
    return (
        results["bulk_deals"],
        results["volume"],
        results["fo_ban"],
        results["results"],
        results["breakouts"],
    )


def save_output(watchlist_data: dict) -> Path:
    """Save the watchlist JSON to outputs/YYYY-MM-DD.json."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    scan_date = watchlist_data.get("scan_date", date.today().isoformat())
    out_path = OUTPUTS_DIR / f"{scan_date}.json"
    out_path.write_text(json.dumps(watchlist_data, indent=2, default=str))
    print(f"[main] output saved to {out_path}")
    return out_path


def main() -> int:
    print(f"[main] === Stock Selector run: {date.today().isoformat()} ===")

    # 1. Fetch
    bulk_deals, volume_gainers, fo_ban_removed, results_calendar, breakouts = fetch_all_data()

    # Count total unique tickers across all sources for the report
    all_tickers: set[str] = set()
    all_tickers.update(d["ticker"] for d in bulk_deals)
    all_tickers.update(d["ticker"] for d in volume_gainers)
    all_tickers.update(fo_ban_removed)
    all_tickers.update(r["ticker"] for r in results_calendar)
    all_tickers.update(b["ticker"] for b in breakouts)
    total_scanned = len(all_tickers)

    # 2. Score
    candidates = score_candidates(
        bulk_deals, volume_gainers, fo_ban_removed, results_calendar, breakouts
    )

    if not candidates:
        print("[main] no qualifying candidates today")
        watchlist_data = {
            "watchlist": [],
            "scan_date": date.today().isoformat(),
            "total_candidates_scanned": total_scanned,
        }
        save_output(watchlist_data)
        send_telegram_alert(watchlist_data)
        return 0

    # 3. Claude synthesis (top 20 candidates → ranked 10)
    print(f"[main] sending {min(len(candidates), 20)} candidates to Claude for synthesis...")
    watchlist_data = synthesize_watchlist(candidates, total_scanned)

    # 4. Save
    save_output(watchlist_data)

    # 5. Telegram alert
    send_telegram_alert(watchlist_data)

    # Print summary to stdout for GitHub Actions logs
    watchlist = watchlist_data.get("watchlist", [])
    print(f"\n[main] === WATCHLIST ({len(watchlist)} picks) ===")
    for i, entry in enumerate(watchlist, 1):
        print(
            f"  {i:>2}. {entry['ticker']:<20} score={entry['score']}"
            f"  {entry['timeframe']}  target=+{entry['target_move_pct']}%"
            f"  risk={entry['risk']}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
