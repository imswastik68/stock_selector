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
from src.data.bse_announcements import fetch_bse_announcements
from src.data.delivery import fetch_delivery_signals
from src.data.breakouts import fetch_breakouts
from src.data.fii_dii import fetch_fii_dii_data
from src.data.fo_ban import fetch_fo_ban_delta
from src.data.gift_nifty import fetch_gift_nifty
from src.data.sector_rotation import fetch_hot_sector_tickers
from src.data.market_context import enrich_candidate_context, fetch_market_wide_context
from src.data.options import fetch_options_signals
from src.data.promoter import fetch_promoter_signals
from src.data.results import fetch_results_calendar
from src.data.volume import fetch_volume_gainers
from src.scorer import score_candidates
from src.agent import synthesize_watchlist
from src.performance import performance_summary, record_picks
from src.telegram_alert import send_telegram_alert

OUTPUTS_DIR = Path(__file__).parent / "outputs"


def fetch_all_data() -> tuple[list, list, list, list, list, dict, dict, set, list, dict, dict, set]:
    """Fetch all data sources concurrently. Returns 12-tuple."""
    print("[main] fetching market data...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            "bulk_deals": pool.submit(fetch_bulk_deals),
            "volume": pool.submit(fetch_volume_gainers),
            "fo_ban": pool.submit(fetch_fo_ban_delta),
            "results": pool.submit(fetch_results_calendar),
            "breakouts": pool.submit(fetch_breakouts),
            "market_wide": pool.submit(fetch_market_wide_context),
            "delivery": pool.submit(fetch_delivery_signals),
            "announcements": pool.submit(fetch_bse_announcements),
            "fii_dii": pool.submit(fetch_fii_dii_data),
            "gift_nifty": pool.submit(fetch_gift_nifty),
        }

        results = {}
        for name, future in futures.items():
            try:
                results[name] = future.result()
            except Exception as exc:
                print(f"[main] {name} fetch error (continuing): {exc}")
                results[name] = [] if name not in ("market_wide", "delivery", "fii_dii", "gift_nifty") else {}

    # Sector rotation depends on market_wide (sector heatmap); run sequentially after
    try:
        heatmap = results["market_wide"].get("sector_heatmap", {})
        hot_sector_tickers = fetch_hot_sector_tickers(heatmap)
    except Exception as exc:
        print(f"[main] sector_rotation fetch error (continuing): {exc}")
        hot_sector_tickers = set()

    elapsed = time.time() - t0
    print(f"[main] all data fetched in {elapsed:.1f}s")

    # fo_ban returns (removed_list, current_set); unpack here so callers get clean types
    fo_ban_raw = results["fo_ban"]
    if isinstance(fo_ban_raw, tuple):
        fo_ban_removed, fo_ban_current = fo_ban_raw
    else:
        fo_ban_removed, fo_ban_current = fo_ban_raw, set()

    return (
        results["bulk_deals"],
        results["volume"],
        fo_ban_removed,
        results["results"],
        results["breakouts"],
        results["market_wide"],
        results["delivery"],
        fo_ban_current,
        results["announcements"],
        results["fii_dii"],
        results["gift_nifty"],
        hot_sector_tickers,
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

    # Build stop_loss map from prior watchlist for SL hit detection
    def _parse_price_str(s) -> float | None:
        try:
            return float(str(s).replace("₹", "").replace(",", "").strip())
        except (ValueError, TypeError):
            return None

    sl_map: dict[str, dict] = {}
    for entry in prior.get("buy_watchlist", []):
        t = entry.get("ticker", "")
        sl = _parse_price_str(entry.get("stop_loss"))
        if t and sl is not None:
            sl_map[t] = {"stop_loss": sl, "direction": "buy"}
    for entry in prior.get("sell_watchlist", []):
        t = entry.get("ticker", "")
        sl = _parse_price_str(entry.get("stop_loss"))
        if t and sl is not None:
            sl_map[t] = {"stop_loss": sl, "direction": "sell"}

    intraday = fetch_intraday_signals(tickers)
    confirmed = {t: v for t, v in intraday.items() if v["intraday_surge"]}

    # Detect stop-loss hits
    sl_hits: dict[str, dict] = {}
    for ticker, v in intraday.items():
        if ticker not in sl_map:
            continue
        sl_info = sl_map[ticker]
        price = v.get("price_current")
        stop = sl_info["stop_loss"]
        direction = sl_info["direction"]
        hit = (direction == "buy" and price is not None and price <= stop) or \
              (direction == "sell" and price is not None and price >= stop)
        if hit:
            sl_hits[ticker] = {
                "price_current": price,
                "stop_loss": stop,
                "direction": direction,
                "pct_vs_stop": round((price / stop - 1) * 100, 1) if stop else 0,
            }

    print(f"\n[main] === MID-DAY CONFIRMED ({len(confirmed)} / {len(intraday)} checked) ===")
    for ticker, v in confirmed.items():
        pct = v.get("pct_vs_prev_close", 0)
        print(
            f"  {ticker:<22} proj={v['volume_ratio_projected']:.1f}x  "
            f"price={v['price_current']} ({pct:+.1f}% vs prev)"
        )
    if sl_hits:
        print(f"\n[main] === STOP-LOSS HITS ({len(sl_hits)}) ===")
        for ticker, v in sl_hits.items():
            print(f"  {ticker:<22} price={v['price_current']} SL={v['stop_loss']}")

    send_telegram_alert(
        {
            "scan_date":          date.today().isoformat(),
            "scan_time":          scan_time,
            "intraday_confirmed": confirmed,
            "intraday_checked":   intraday,
            "sl_hits":            sl_hits,
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
    bulk_deals, volume_gainers, fo_ban_removed, results_calendar, breakouts, market_wide_ctx, delivery_signals, fo_ban_current, announcements, fii_dii, gift_nifty, hot_sector_tickers = fetch_all_data()

    # Count total unique tickers across all sources for the report
    all_tickers: set[str] = set()
    all_tickers.update(d["ticker"] for d in bulk_deals)
    all_tickers.update(d["ticker"] for d in volume_gainers)
    all_tickers.update(fo_ban_removed)
    all_tickers.update(r["ticker"] for r in results_calendar)
    all_tickers.update(b["ticker"] for b in breakouts)
    all_tickers.update(a["ticker"] for a in announcements)
    total_scanned = len(all_tickers)

    # 2. Initial score (without options/promoter — those need per-ticker API calls)
    nifty_regime = market_wide_ctx.get("nifty_regime", "normal")
    print(f"[main] nifty_regime={nifty_regime}")
    candidates = score_candidates(
        bulk_deals, volume_gainers, fo_ban_removed, results_calendar, breakouts,
        delivery_signals=delivery_signals,
        fo_ban_current=fo_ban_current,
        market_regime=nifty_regime,
        announcements=announcements,
        hot_sector_tickers=hot_sector_tickers,
    )

    # 3. Enrich top 20 candidates with options PCR + promoter buying
    #    Done AFTER scoring so we only call APIs for qualified stocks, not the whole universe
    if candidates:
        top_tickers = [c["ticker"] for c in candidates[:20]]
        print(f"[main] fetching options data for {len(top_tickers)} candidates...")
        # Build prev_closes from volume data AND breakouts so candidates that appear
        # only in bulk deals / breakouts (not volume gainers) still get a price reference
        # for options long/short buildup detection.
        prev_closes = {b["ticker"]: b.get("today_close", 0) for b in breakouts}
        prev_closes.update({d["ticker"]: d.get("today_close", 0) for d in volume_gainers})
        options_signals = fetch_options_signals(top_tickers, prev_closes=prev_closes)

        print(f"[main] fetching promoter signals for {len(top_tickers)} candidates...")
        promoter_signals = fetch_promoter_signals(top_tickers, bulk_deals)

        # Re-score with new signals so they affect sorting
        candidates = score_candidates(
            bulk_deals, volume_gainers, fo_ban_removed, results_calendar, breakouts,
            promoter_signals=promoter_signals, options_signals=options_signals,
            delivery_signals=delivery_signals,
            fo_ban_current=fo_ban_current,
            market_regime=nifty_regime,
            announcements=announcements,
            hot_sector_tickers=hot_sector_tickers,
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
    market_context["fii_dii"] = fii_dii
    market_context["gift_nifty"] = gift_nifty

    # 5. Third re-score: incorporate RSI, RS vs Nifty, OBV, BB squeeze from technicals
    #    (only top 20 have technical data — rest get no delta, preserving their rank)
    # 2 technical signals after research review (rsi_extended, obv, bb_squeeze removed)
    tech_signals = {
        ticker: {
            "rsi_momentum":        t.get("rsi_momentum", False),
            "rs_vs_nifty":         t.get("rs_vs_nifty", False),
            "rsi_bearish_div":     t.get("rsi_bearish_div", False),
            "rsi_bullish_div":     t.get("rsi_bullish_div", False),
            "macd_bullish_cross":  t.get("macd_bullish_cross", False),
            "macd_bearish_cross":  t.get("macd_bearish_cross", False),
            "obv_accumulation":    t.get("obv_accumulation", False),
            "bb_squeeze_breakout": t.get("bb_squeeze_breakout", False),
            "bullish_candle":      t.get("bullish_candle", False),
            "bearish_candle":      t.get("bearish_candle", False),
            "weekly_trend_aligned":t.get("weekly_trend_aligned", False),
            "momentum_6m_strong":  t.get("momentum_6m_strong", False),
            "rs_quality_strong":   t.get("rs_quality_strong", False),
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
            fo_ban_current=fo_ban_current,
            market_regime=nifty_regime,
            announcements=announcements,
            hot_sector_tickers=hot_sector_tickers,
        )

    # 6. LLM synthesis (Wyckoff + SMC + VSA)
    print(f"[main] synthesising watchlist for {min(len(candidates), 20)} candidates...")
    watchlist_data = synthesize_watchlist(candidates, total_scanned, market_context=market_context)

    # 7. Save
    save_output(watchlist_data)

    # 7b. Performance tracking: record today's picks + evaluate prior
    try:
        record_picks(watchlist_data)
        perf = performance_summary(lookback_days=30)
        watchlist_data["performance"] = perf
        print(f"[main] performance: {perf}")
    except Exception as exc:
        print(f"[main] performance tracking error (non-fatal): {exc}")

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
