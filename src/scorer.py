"""
Deterministic multi-signal scoring engine.

Takes outputs from all data fetchers and returns a scored candidate list.
Claude is NOT involved here — all scoring is rule-based and auditable.
"""

from __future__ import annotations

SHORT_TERM_WEIGHTS = {
    "bulk_deal_fii_dii": 3,
    "volume_5x": 3,
    "actual_52w_breakout": 3,   # price broke through prior 52w high
    "delivery_surge": 2,        # DELIV_PER >= 50% on EQ series — institutional accumulation
    "rs_vs_nifty": 2,           # stock outperforming Nifty by 2%+ over 20d
    "options_pcr_fear": 1,      # PCR > 1.5: extreme fear = contrarian bullish (min OI required)
    "options_long_buildup": 1,  # price up + OI up = fresh longs (min OI required)
    "rsi_momentum": 1,          # RSI 50-75: momentum building
    "rsi_bullish_div": 1,       # price LL + RSI HL: sellers losing momentum
}

SWING_WEIGHTS = {
    "results_due": 1,          # upcoming results — informational flag only (PEAD needs actual beat)
    "promoter_buying": 3,      # sets timeframe → 5-7d
    "sector_rotation": 1,
    "consolidation_breakout": 3,
}

DISQUALIFIER_WEIGHTS = {
    "sebi_investigation": -10,
    "f_group": -5,
    "distribution_signal": -5,  # volume spike + price DOWN = institutional selling
    "rsi_bearish_div": -2,       # price HH + RSI LH: momentum fading (valid even above RSI 70)
    "thin_market": -4,           # avg daily turnover < ₹5cr: unreliable signals + exit risk
    "options_short_buildup": -2,  # price down + OI up = fresh shorts entering
    "options_pcr_greed": -1,      # PCR < 0.5: extreme complacency (min OI required)
}

MIN_SCORE = 3
MIN_SIGNALS = 1
PENNY_THRESHOLD = 10.0  # ₹10


def _build_signal_map(
    ticker: str,
    bulk_deals: list[dict],
    volume_data: dict | None,
    fo_ban_removed: list[str],
    results_calendar: list[dict],
    breakout_data: dict | None,
    promoter_data: dict | None = None,
    options_data: dict | None = None,
    technical_data: dict | None = None,
    delivery_data: dict | None = None,
) -> dict[str, bool]:
    """Build a boolean signal map for a single ticker."""
    signals: dict[str, bool] = {k: False for k in list(SHORT_TERM_WEIGHTS) + list(SWING_WEIGHTS) + list(DISQUALIFIER_WEIGHTS)}

    # --- SHORT-TERM signals ---

    # Bulk/block deal by known FII/DII
    ticker_deals = [d for d in bulk_deals if d["ticker"] == ticker and d["is_fii_dii"]]
    signals["bulk_deal_fii_dii"] = bool(ticker_deals)

    # Volume ≥ 2.5× 30-day average
    if volume_data:
        signals["volume_5x"]          = volume_data.get("is_volume_surge", False)
        signals["distribution_signal"] = volume_data.get("is_distribution", False)
        # Thin market: avg daily turnover < ₹5 crore — signal quality is unreliable
        avg_30d = volume_data.get("avg_30d_volume", 0) or 0
        price   = volume_data.get("today_close", 0) or 0
        signals["thin_market"] = bool(avg_30d > 0 and price > 0 and avg_30d * price < 5e7)

    # 52-week high: only actual breakout scores; proximity alone removed (redundant + noisy)
    if breakout_data:
        signals["actual_52w_breakout"]    = breakout_data.get("actual_breakout", False)
        signals["consolidation_breakout"] = breakout_data.get("consolidation_breakout", False)

    # Options signals — PCR only at extremes; long_buildup gated by min OI in options.py
    if options_data:
        signals["options_pcr_fear"]      = options_data.get("pcr_fear", False)
        signals["options_pcr_greed"]     = options_data.get("pcr_greed", False)
        signals["options_long_buildup"]  = options_data.get("long_buildup", False)
        signals["options_short_buildup"] = options_data.get("short_buildup", False)

    # Technical signals (pass 3 only — after enrich_candidate_context)
    if technical_data:
        signals["rsi_momentum"]   = technical_data.get("rsi_momentum", False)
        signals["rs_vs_nifty"]    = technical_data.get("rs_vs_nifty", False)
        signals["rsi_bearish_div"] = technical_data.get("rsi_bearish_div", False)
        signals["rsi_bullish_div"] = technical_data.get("rsi_bullish_div", False)

    # Delivery volume signal from NSE bhav copy
    if delivery_data:
        signals["delivery_surge"] = delivery_data.get("delivery_surge", False)

    # --- SWING signals ---

    # Upcoming results — weak informational flag; real PEAD requires confirmed post-beat data
    ticker_results = [r for r in results_calendar if r["ticker"] == ticker]
    if ticker_results:
        signals["results_due"] = True

    # Promoter / insider buying
    if promoter_data:
        signals["promoter_buying"] = promoter_data.get("promoter_bought", False)

    return signals


def _compute_score(signals: dict[str, bool], market_regime: str = "normal") -> tuple[int, str]:
    score = 0
    timeframe = "1-2d"

    for sig, weight in SHORT_TERM_WEIGHTS.items():
        if signals.get(sig):
            # RSI 50-70 produces false signals in high-volatility regimes; disable it
            if sig == "rsi_momentum" and market_regime == "high_vol":
                continue
            score += weight

    for sig, weight in SWING_WEIGHTS.items():
        if signals.get(sig):
            score += weight
            if sig in ("results_due", "promoter_buying"):
                timeframe = "5-7d"

    for sig, weight in DISQUALIFIER_WEIGHTS.items():
        if signals.get(sig):
            score += weight  # weights are already negative

    return score, timeframe


def _active_signals(signals: dict[str, bool]) -> list[str]:
    return [sig for sig, active in signals.items() if active and sig not in DISQUALIFIER_WEIGHTS]


def _disqualifiers(signals: dict[str, bool]) -> list[str]:
    return [sig for sig, active in signals.items() if active and sig in DISQUALIFIER_WEIGHTS]


def score_candidates(
    bulk_deals: list[dict],
    volume_gainers: list[dict],
    fo_ban_removed: list[str],
    results_calendar: list[dict],
    breakouts: list[dict],
    promoter_signals: dict | None = None,
    options_signals: dict | None = None,
    technical_signals: dict | None = None,
    delivery_signals: dict | None = None,
    market_regime: str = "normal",
) -> list[dict]:
    """
    Score every unique ticker across all data sources and return qualifying candidates.

    market_regime: "low_vol" | "normal" | "high_vol" — in high_vol, all scores drop by 1
    to account for increased false-signal rate.
    """
    # Collect all unique tickers across all data sources
    all_tickers: set[str] = set()
    all_tickers.update(d["ticker"] for d in bulk_deals)
    all_tickers.update(d["ticker"] for d in volume_gainers)
    all_tickers.update(r["ticker"] for r in results_calendar)
    all_tickers.update(b["ticker"] for b in breakouts)

    # Build lookup dicts for O(1) access
    volume_map: dict[str, dict] = {d["ticker"]: d for d in volume_gainers}
    breakout_map: dict[str, dict] = {b["ticker"]: b for b in breakouts}

    candidates = []
    for ticker in all_tickers:
        vol = volume_map.get(ticker)
        brk = breakout_map.get(ticker)
        prom = (promoter_signals or {}).get(ticker)
        opts = (options_signals or {}).get(ticker)
        tech  = (technical_signals or {}).get(ticker)
        deliv = (delivery_signals or {}).get(ticker)

        signals = _build_signal_map(
            ticker, bulk_deals, vol, fo_ban_removed, results_calendar, brk,
            promoter_data=prom, options_data=opts, technical_data=tech,
            delivery_data=deliv,
        )

        score, timeframe = _compute_score(signals, market_regime)

        # High-volatility regime: all signals are noisier — reduce score by 1
        if market_regime == "high_vol":
            score = max(0, score - 1)

        active = _active_signals(signals)
        disqs = _disqualifiers(signals)

        if score < MIN_SCORE or len(active) < MIN_SIGNALS:
            continue

        today_close = (vol or brk or {}).get("today_close") or (vol or {}).get("today_close")
        is_penny = today_close is not None and today_close < PENNY_THRESHOLD

        candidates.append({
            "ticker": f"[PENNY]{ticker}" if is_penny else ticker,
            "score": score,
            "active_signals": active,
            "disqualifiers": disqs,
            "timeframe": timeframe,
            "today_close": today_close,
            "volume_ratio": (vol or {}).get("volume_ratio"),
            "market_cap_cr": (vol or {}).get("market_cap_cr"),
            "52w_high": (brk or {}).get("52w_high"),
            "consolidation_breakout": signals.get("consolidation_breakout", False),
            "promoter_pct": (prom or {}).get("promoter_pct"),
            "promoter_bought": (prom or {}).get("promoter_bought", False),
            "options_pcr": (opts or {}).get("pcr"),
            # Pass through raw deal info for LLM context
            "bulk_deals": [d for d in bulk_deals if d["ticker"] == ticker],
            "upcoming_results": [r for r in results_calendar if r["ticker"] == ticker],
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    print(f"[scorer] {len(all_tickers)} tickers evaluated, {len(candidates)} qualified")
    return candidates
