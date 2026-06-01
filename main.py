"""
Stock Selector — Daily Orchestrator
Runs at 6:30 PM IST (13:00 UTC) via GitHub Actions.

Pipeline: fetch data → score candidates → Claude synthesis → Telegram alert → save JSON
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone, timedelta
from pathlib import Path


def _detect_scan_mode() -> str:
    """
    Detect whether this is a pre-market or EOD scan.
    Pre-market: before 9:15 AM IST (uses yesterday's cache, no downloads).
    EOD: after 3:30 PM IST (full fresh download).
    Override via SCAN_MODE env var: 'pre_market' or 'eod'.
    """
    override = os.environ.get("SCAN_MODE", "").strip().lower()
    if override in ("pre_market", "mid_day", "eod"):
        return override
    IST = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(IST)
    if now.hour < 9 or (now.hour == 9 and now.minute < 15):
        return "pre_market"
    if now.hour == 12 or (now.hour == 13 and now.minute == 0):
        return "mid_day"
    return "eod"

from dotenv import load_dotenv

load_dotenv()

from src.data.bulk_deals import fetch_bulk_deals
from src.data.delivery import fetch_delivery_signals
from src.data.breakouts import fetch_breakouts
from src.data.fo_ban import fetch_fo_ban_delta
from src.data.market_context import enrich_candidate_context, fetch_market_wide_context
from src.data.options import fetch_options_signals
from src.data.promoter import fetch_promoter_signals
from src.data.results import fetch_results_calendar
from src.data.volume import fetch_volume_gainers
from src.scorer import score_candidates
from src.agent import synthesize_watchlist
from src.telegram_alert import send_telegram_alert

OUTPUTS_DIR = Path(__file__).parent / "outputs"


def fetch_all_data() -> tuple[list, list, list, list, list, dict, dict]:
    """Fetch all data sources concurrently. Returns 7-tuple including market-wide and delivery context."""
    print("[main] fetching market data...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=7) as pool:
        futures = {
            "bulk_deals": pool.submit(fetch_bulk_deals),
            "volume": pool.submit(fetch_volume_gainers),
            "fo_ban": pool.submit(fetch_fo_ban_delta),
            "results": pool.submit(fetch_results_calendar),
            "breakouts": pool.submit(fetch_breakouts),
            "market_wide": pool.submit(fetch_market_wide_context),
            "delivery": pool.submit(fetch_delivery_signals),
        }

        results = {}
        for name, future in futures.items():
            try:
                results[name] = future.result()
            except Exception as exc:
                print(f"[main] {name} fetch error (continuing): {exc}")
                results[name] = [] if name not in ("market_wide", "delivery") else {}

    elapsed = time.time() - t0
    print(f"[main] all data fetched in {elapsed:.1f}s")
    return (
        results["bulk_deals"],
        results["volume"],
        results["fo_ban"],
        results["results"],
        results["breakouts"],
        results["market_wide"],
        results["delivery"],
    )


def save_output(watchlist_data: dict) -> Path:
    """Save the watchlist JSON to outputs/YYYY-MM-DD.json."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    scan_date = watchlist_data.get("scan_date", date.today().isoformat())
    out_path = OUTPUTS_DIR / f"{scan_date}.json"
    out_path.write_text(json.dumps(watchlist_data, indent=2, default=str))
    print(f"[main] output saved to {out_path}")
    return out_path


def run_mid_day_scan() -> int:
    """
    Mid-day scan (12:30 PM IST): check intraday momentum for yesterday's candidates.
    No full universe download — only the ~20 tickers from the most recent output.
    """
    from src.data.intraday import fetch_intraday_signals

    IST = timezone(timedelta(hours=5, minutes=30))
    scan_time = datetime.now(IST).strftime("%H:%M")
    print(f"[main] === Mid-day scan: {date.today().isoformat()}  {scan_time} IST ===")

    # Load most recent watchlist output for candidate tickers
    outputs = sorted(OUTPUTS_DIR.glob("*.json"), reverse=True)
    if not outputs:
        print("[main] no prior output found — run EOD scan first")
        return 0

    prior = json.loads(outputs[0].read_text())
    all_entries = (
        prior.get("buy_watchlist", []) +
        prior.get("sell_watchlist", []) +
        prior.get("phase_b_watchlist", [])
    )
    tickers = [
        c["ticker"] for c in all_entries
        if not c["ticker"].startswith("[PENNY]")
    ][:20]

    if not tickers:
        print("[main] no candidates from prior run — nothing to check")
        return 0

    intraday = fetch_intraday_signals(tickers)
    confirmed = {t: v for t, v in intraday.items() if v["intraday_surge"]}

    print(f"\n[main] === MID-DAY CONFIRMED ({len(confirmed)} / {len(intraday)} checked) ===")
    for ticker, v in confirmed.items():
        pct = v.get("pct_vs_prev_close", 0)
        print(
            f"  {ticker:<22} proj={v['volume_ratio_projected']:.1f}x  "
            f"price={v['price_current']} ({pct:+.1f}% vs prev)"
        )

    send_telegram_alert(
        {
            "scan_date":          date.today().isoformat(),
            "scan_time":          scan_time,
            "intraday_confirmed": confirmed,
            "intraday_checked":   intraday,
        },
        mode="mid_day",
    )
    return 0


def main() -> int:
    scan_mode = _detect_scan_mode()
    os.environ["SCAN_MODE"] = scan_mode  # propagate to fetchers via env

    if scan_mode == "mid_day":
        return run_mid_day_scan()

    print(f"[main] === Stock Selector run: {date.today().isoformat()}  mode={scan_mode} ===")

    # 1. Fetch (parallel)
    bulk_deals, volume_gainers, fo_ban_removed, results_calendar, breakouts, market_wide_ctx, delivery_signals = fetch_all_data()

    # Count total unique tickers across all sources for the report
    all_tickers: set[str] = set()
    all_tickers.update(d["ticker"] for d in bulk_deals)
    all_tickers.update(d["ticker"] for d in volume_gainers)
    all_tickers.update(fo_ban_removed)
    all_tickers.update(r["ticker"] for r in results_calendar)
    all_tickers.update(b["ticker"] for b in breakouts)
    total_scanned = len(all_tickers)

    # 2. Initial score (without options/promoter — those need per-ticker API calls)
    nifty_regime = market_wide_ctx.get("nifty_regime", "normal")
    print(f"[main] nifty_regime={nifty_regime}")
    candidates = score_candidates(
        bulk_deals, volume_gainers, fo_ban_removed, results_calendar, breakouts,
        delivery_signals=delivery_signals,
        market_regime=nifty_regime,
    )

    # 3. Enrich top 20 candidates with options PCR + promoter buying
    #    Done AFTER scoring so we only call APIs for qualified stocks, not the whole universe
    if candidates:
        top_tickers = [c["ticker"] for c in candidates[:20]]
        print(f"[main] fetching options data for {len(top_tickers)} candidates...")
        prev_closes = {d["ticker"]: d.get("today_close", 0) for d in volume_gainers}
        options_signals = fetch_options_signals(top_tickers, prev_closes=prev_closes)

        print(f"[main] fetching promoter signals for {len(top_tickers)} candidates...")
        promoter_signals = fetch_promoter_signals(top_tickers, bulk_deals)

        # Re-score with new signals so they affect sorting
        candidates = score_candidates(
            bulk_deals, volume_gainers, fo_ban_removed, results_calendar, breakouts,
            promoter_signals=promoter_signals, options_signals=options_signals,
            delivery_signals=delivery_signals,
            market_regime=nifty_regime,
        )
    else:
        options_signals = {}
        promoter_signals = {}

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

    # 4. Enrich market context for top 20 candidates (OHLCV, beta, ATR, technicals)
    candidate_tickers = [c["ticker"] for c in candidates[:20]]
    print(f"[main] enriching market context for {len(candidate_tickers)} candidates...")
    market_context = enrich_candidate_context(candidate_tickers, market_wide_ctx)
    # Stash auxiliary data so agent.py can embed it in the prompt
    market_context["bulk_deals"] = bulk_deals
    market_context["fo_ban_delta"] = fo_ban_removed
    market_context["results_calendar"] = results_calendar

    # 5. Third re-score: incorporate RSI, RS vs Nifty, OBV, BB squeeze from technicals
    #    (only top 20 have technical data — rest get no delta, preserving their rank)
    # 2 technical signals after research review (rsi_extended, obv, bb_squeeze removed)
    tech_signals = {
        ticker: {
            "rsi_momentum": t.get("rsi_momentum", False),
            "rs_vs_nifty":  t.get("rs_vs_nifty", False),
        }
        for ticker, t in market_context.get("technicals", {}).items()
    }
    if tech_signals:
        print(f"[main] re-scoring with technical signals for {len(tech_signals)} candidates...")
        candidates = score_candidates(
            bulk_deals, volume_gainers, fo_ban_removed, results_calendar, breakouts,
            promoter_signals=promoter_signals,
            options_signals=options_signals,
            technical_signals=tech_signals,
            delivery_signals=delivery_signals,
            market_regime=nifty_regime,
        )

    # 6. LLM synthesis (Wyckoff + SMC + VSA)
    print(f"[main] synthesising watchlist for {min(len(candidates), 20)} candidates...")
    watchlist_data = synthesize_watchlist(candidates, total_scanned, market_context=market_context)

    # 7. Save
    save_output(watchlist_data)

    # 8. Telegram alert
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
