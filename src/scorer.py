"""
Deterministic multi-signal scoring engine.

Takes outputs from all data fetchers and returns a scored candidate list.
Claude is NOT involved here — all scoring is rule-based and auditable.
"""

from __future__ import annotations

SHORT_TERM_WEIGHTS = {
    "bulk_deal_fii_dii": 3,
    "volume_5x": 3,
    "fo_ban_removed": 2,
    "near_52w_high": 2,
    "small_cap": 1,
}

SWING_WEIGHTS = {
    "eps_surprise_15pct": 4,   # sets timeframe → 5-7d
    "promoter_buying": 3,      # sets timeframe → 5-7d
    "sector_rotation": 2,
    "consolidation_breakout": 2,
}

DISQUALIFIER_WEIGHTS = {
    "sebi_investigation": -10,
    "f_group": -5,
    "distribution_signal": -5,  # volume spike + price DOWN
}

MIN_SCORE = 2
MIN_SIGNALS = 1
PENNY_THRESHOLD = 10.0  # ₹10


def _build_signal_map(
    ticker: str,
    bulk_deals: list[dict],
    volume_data: dict | None,
    fo_ban_removed: list[str],
    results_calendar: list[dict],
    breakout_data: dict | None,
) -> dict[str, bool]:
    """Build a boolean signal map for a single ticker."""
    signals: dict[str, bool] = {k: False for k in list(SHORT_TERM_WEIGHTS) + list(SWING_WEIGHTS) + list(DISQUALIFIER_WEIGHTS)}

    # --- SHORT-TERM signals ---

    # Bulk/block deal by known FII/DII
    ticker_deals = [d for d in bulk_deals if d["ticker"] == ticker and d["is_fii_dii"]]
    signals["bulk_deal_fii_dii"] = bool(ticker_deals)

    # Volume ≥ 5× 30-day average
    if volume_data:
        signals["volume_5x"] = volume_data.get("is_volume_surge", False)
        signals["small_cap"] = volume_data.get("is_small_cap", False)
        signals["distribution_signal"] = volume_data.get("is_distribution", False)

    # F&O ban removal
    signals["fo_ban_removed"] = ticker in fo_ban_removed

    # Near 52-week high (within 1%)
    if breakout_data:
        signals["near_52w_high"] = True
        signals["consolidation_breakout"] = breakout_data.get("consolidation_breakout", False)

    # --- SWING signals ---

    # Upcoming results (EPS surprise unknown without consensus data → [UNCONFIRMED])
    ticker_results = [r for r in results_calendar if r["ticker"] == ticker]
    if ticker_results:
        # We flag this as a potential signal; Claude will mark it [UNCONFIRMED]
        # because we don't have consensus EPS estimates from free APIs
        signals["eps_surprise_15pct"] = True  # tagged [UNCONFIRMED] by Claude agent

    # sector_rotation and promoter_buying: not computed from free APIs in this version
    # Leave as False; extend with screener.in export or BSE SHP data in a future iteration

    return signals


def _compute_score(signals: dict[str, bool]) -> tuple[int, str]:
    score = 0
    timeframe = "1-2d"

    for sig, weight in SHORT_TERM_WEIGHTS.items():
        if signals.get(sig):
            score += weight

    for sig, weight in SWING_WEIGHTS.items():
        if signals.get(sig):
            score += weight
            if sig in ("eps_surprise_15pct", "promoter_buying"):
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
) -> list[dict]:
    """
    Merge all data sources, score every unique ticker, and return qualifying candidates
    sorted by score descending.

    Qualification: score >= MIN_SCORE and active_signals >= MIN_SIGNALS.
    """
    # Collect all unique tickers across all data sources
    all_tickers: set[str] = set()
    all_tickers.update(d["ticker"] for d in bulk_deals)
    all_tickers.update(d["ticker"] for d in volume_gainers)
    all_tickers.update(fo_ban_removed)
    all_tickers.update(r["ticker"] for r in results_calendar)
    all_tickers.update(b["ticker"] for b in breakouts)

    # Build lookup dicts for O(1) access
    volume_map: dict[str, dict] = {d["ticker"]: d for d in volume_gainers}
    breakout_map: dict[str, dict] = {b["ticker"]: b for b in breakouts}

    candidates = []
    for ticker in all_tickers:
        vol = volume_map.get(ticker)
        brk = breakout_map.get(ticker)

        signals = _build_signal_map(
            ticker, bulk_deals, vol, fo_ban_removed, results_calendar, brk
        )

        score, timeframe = _compute_score(signals)
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
            # Pass through raw deal info for Claude context
            "bulk_deals": [d for d in bulk_deals if d["ticker"] == ticker],
            "upcoming_results": [r for r in results_calendar if r["ticker"] == ticker],
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    print(f"[scorer] {len(all_tickers)} tickers evaluated, {len(candidates)} qualified")
    return candidates
