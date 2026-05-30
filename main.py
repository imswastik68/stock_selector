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
from src.data.market_context import enrich_candidate_context, fetch_market_wide_context
from src.data.results import fetch_results_calendar
from src.data.volume import fetch_volume_gainers
from src.scorer import score_candidates
from src.agent import synthesize_watchlist
from src.telegram_alert import send_telegram_alert

OUTPUTS_DIR = Path(__file__).parent / "outputs"


def fetch_all_data() -> tuple[list, list, list, list, list, dict]:
    """Fetch all data sources concurrently. Returns 6-tuple including market-wide context."""
    print("[main] fetching market data...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            "bulk_deals": pool.submit(fetch_bulk_deals),
            "volume": pool.submit(fetch_volume_gainers),
            "fo_ban": pool.submit(fetch_fo_ban_delta),
            "results": pool.submit(fetch_results_calendar),
            "breakouts": pool.submit(fetch_breakouts),
            "market_wide": pool.submit(fetch_market_wide_context),
        }

        results = {}
        for name, future in futures.items():
            try:
                results[name] = future.result()
            except Exception as exc:
                print(f"[main] {name} fetch error (continuing): {exc}")
                results[name] = [] if name != "market_wide" else {}

    elapsed = time.time() - t0
    print(f"[main] all data fetched in {elapsed:.1f}s")
    return (
        results["bulk_deals"],
        results["volume"],
        results["fo_ban"],
        results["results"],
        results["breakouts"],
        results["market_wide"],
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

    # 1. Fetch (parallel)
    bulk_deals, volume_gainers, fo_ban_removed, results_calendar, breakouts, market_wide_ctx = fetch_all_data()

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
            "buy_watchlist": [],
            "sell_watchlist": [],
            "phase_b_watchlist": [],
            "nifty_context": market_wide_ctx.get("nifty_structure", {}).get("trend", "ranging"),
            "scan_date": date.today().isoformat(),
            "total_screened": total_scanned,
            "data_quality_warnings": [],
        }
        save_output(watchlist_data)
        send_telegram_alert(watchlist_data)
        return 0

    # 3. Enrich market context for top 15 candidates (OHLCV, beta, ATR)
    # Capped at 15 to stay under Groq free tier 6K TPM limit (~5,500 tokens per prompt)
    candidate_tickers = [c["ticker"] for c in candidates[:15]]
    print(f"[main] enriching market context for {len(candidate_tickers)} candidates...")
    market_context = enrich_candidate_context(candidate_tickers, market_wide_ctx)
    # Stash auxiliary data so agent.py can embed it in the prompt
    market_context["bulk_deals"] = bulk_deals
    market_context["fo_ban_delta"] = fo_ban_removed
    market_context["results_calendar"] = results_calendar

    # 4. LLM synthesis (Wyckoff + SMC + VSA)
    print(f"[main] synthesising watchlist for {min(len(candidates), 15)} candidates...")
    watchlist_data = synthesize_watchlist(candidates, total_scanned, market_context=market_context)

    # 5. Save
    save_output(watchlist_data)

    # 6. Telegram alert
    send_telegram_alert(watchlist_data)

    # Print summary to stdout for GitHub Actions logs
    buy_list = watchlist_data.get("buy_watchlist", [])
    sell_list = watchlist_data.get("sell_watchlist", [])
    phase_b_list = watchlist_data.get("phase_b_watchlist", [])

    print(f"\n[main] === BUY WATCHLIST ({len(buy_list)} picks) ===")
    for i, entry in enumerate(buy_list, 1):
        print(
            f"  {i:>2}. {entry['ticker']:<22} score={entry['score']}"
            f"  {entry.get('wyckoff_phase','?'):<18} risk={entry['risk']}"
        )

    if sell_list:
        print(f"\n[main] === SELL WATCHLIST ({len(sell_list)} picks) ===")
        for i, entry in enumerate(sell_list, 1):
            print(
                f"  {i:>2}. {entry['ticker']:<22} score={entry['score']}"
                f"  {entry.get('wyckoff_phase','?'):<18} risk={entry['risk']}"
            )

    if phase_b_list:
        print(f"\n[main] === PHASE-B WATCH ({len(phase_b_list)} stocks) ===")
        for i, entry in enumerate(phase_b_list, 1):
            print(f"  {i:>2}. {entry['ticker']:<22} {entry.get('phase','?')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
