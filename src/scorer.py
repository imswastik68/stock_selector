"""
Deterministic multi-signal scoring engine.

Takes outputs from all data fetchers and returns a scored candidate list.
Claude is NOT involved here — all scoring is rule-based and auditable.
"""

from __future__ import annotations

# PENDING_VALIDATION (2026-07, Phase 2): every signal below carries a hand weight
# with ZERO rows in any backtest (outputs/backtest_signal_stats.json's
# note_untested_signals) -- bulk deals/SAST/promoter/delivery/announcements/options
# have no free historical archive today, per scripts/backtest.py's own docstring.
# Weight forced to 0 until each signal is either (a) SHIP-verdicted by
# scripts/backtest_events.py (historical delivery/SAST/bulk-deal backtest, next
# phase) with weight = round(3 x ret_lift), or (b) accumulates enough of
# data/event_archive/ for the same analysis. Value = the pre-zeroing hand weight,
# kept for the record / to restore quickly if a signal ships.
PENDING_VALIDATION: dict[str, int] = {
    "bulk_deal_fii_dii": 3, "fundamental_strong": 2,
    "options_pcr_fear": 1, "options_long_buildup": 1, "options_short_covering": 1,
    "fo_ban_lifted": 1, "sector_in_momentum": 1,
    "promoter_buying": 3, "sast_insider_buying": 3, "consolidation_breakout": 3,
    "results_beat_announced": 3, "buyback_announced": 2, "contract_win": 2,
    "dividend_announced": 1, "results_due": 1,
    # delivery_surge PROMOTED 2026-07 (Phase 7) -- see SHORT_TERM_WEIGHTS entry below.
}

SHORT_TERM_WEIGHTS = {
    "fundamental_strong": 0,        # PENDING_VALIDATION (was 2) — ROE>15% & D/E<1 & EPS-growth>0, untested
    "bulk_deal_fii_dii": 0,         # PENDING_VALIDATION (was 3) — untested
    "actual_52w_breakout": 3,      # price broke through prior 52w high — validated +3.59pp/n=7,138
    "near_52w_high": 3,            # BEST validated signal: +5.09pp WR / +1.01 ret lift, n=15,005
                                    # (156w backtest). Stacks with actual_52w_breakout (strict
                                    # subset, proximity<=0) -- a real breakout day scores 6, faithful
                                    # to how each was independently measured. See _build_signal_map.
    "delivery_surge": 1,            # PROMOTED (2026-07, Phase 7): scripts/backtest_events.py SHIP
                                    # verdict against real NSE bhavcopy history, n=38,279, ret_lift
                                    # +0.316 (wr_lift +1.85pp), 70/30 holdout sign-consistent
                                    # (train +0.406 / holdout +0.104). weight = round(3 x 0.316) = 1.
                                    # See outputs/event_backtest.json for full stats.
    "rsi_bearish_div": 3,          # price HH + RSI LH: data shows +1.04 lift over 15k trades
    "rsi_momentum": 2,             # RSI 50-75: momentum building
    "rs_quality_strong": 0,        # FAILED backtest (not pending) -- -1.47pp WR lift, avg_return
                                    # -1.0%, n=16,907 (156w backtest). Do not restore without a NEW
                                    # positive result; also removed from all 3 REGIME_WEIGHTS buckets
                                    # below (the regime merge would otherwise resurrect it).
    "rs_vs_nifty": 1,              # stock outperforming Nifty by 2%+ over 20d
    "options_pcr_fear": 0,          # PENDING_VALIDATION (was 1) — untested
    "options_long_buildup": 0,      # PENDING_VALIDATION (was 1) — untested
    "options_short_covering": 0,    # PENDING_VALIDATION (was 1) — untested
    "fo_ban_lifted": 0,             # PENDING_VALIDATION (was 1) — untested
    "weekly_trend_aligned": 1,     # weekly EMA10 > EMA20: daily signal aligns with larger timeframe
    "sector_in_momentum": 0,       # PENDING_VALIDATION (was 1) — untested
}

SWING_WEIGHTS = {
    "results_due": 0,               # PENDING_VALIDATION (was 1) — informational only, untested
    "promoter_buying": 0,           # PENDING_VALIDATION (was 3) — sets timeframe -> 5-7d still (see _compute_score)
    "sast_insider_buying": 0,       # FAILED backtest (not pending) -- fires on ANY SAST filing (pledges,
                                    # creeping acquisitions, inter-se transfers), ret_lift=-0.407, n=2010.
                                    # The isolated open-market-buy version (promoter_open_mkt_buy,
                                    # scripts/backtest_events.py --source pit) has a promising point
                                    # estimate (wr_lift +5.38pp, ret_lift +0.833) but INSUFFICIENT_SAMPLE
                                    # (n=258, needs >=500) -- not live-wired, revisit once more PIT
                                    # history accumulates. See outputs/event_backtest.json.
    "consolidation_breakout": 0,    # PENDING_VALIDATION (was 3) — untested
    "results_beat_announced": 0,    # FAILED backtest (not pending) -- fires on ANY results filing
                                    # (beat or miss), ret_lift=-1.234, n=8824. See pead_positive_surprise
                                    # below for the surprise-conditioned version that SHIPPED.
    "buyback_announced": 0,         # PENDING_VALIDATION (was 2) — untested
    "contract_win": 0,              # PENDING_VALIDATION (was 2) — untested
    "dividend_announced": 0,        # PENDING_VALIDATION (was 1) — untested
    "pead_positive_surprise": 1,    # PROMOTED (Alpha Round Phase 2/5): scripts/backtest_events.py SHIP
                                    # verdict, n=1696, ret_lift +0.308 (wr_lift +2.79pp), 70/30 holdout
                                    # sign-consistent and STRENGTHENING (train +0.214 -> holdout +0.527).
                                    # Fires on a results filing whose reaction-day close move is >=+3%
                                    # (see src.data.bse_announcements.classify_pead_reaction). weight =
                                    # round(3 x 0.308) = 1. See outputs/event_backtest.json.
}

DISQUALIFIER_WEIGHTS = {
    "fundamental_weak": -2,          # net loss or D/E>3 (yfinance .info); missing data → neutral
    "f_group": -5,                 # currently on NSE F&O ban list: liquidity risk, forced-exit danger
    "rsi_bullish_div": -5,         # price LL + RSI HL: false bottoms — data shows −1.63 lift over 253 trades
    "thin_market_extreme": -4,     # avg daily turnover < ₹1cr: unreliable signals + exit risk
    "thin_market_light": -2,       # avg daily turnover ₹1cr–₹5cr: reduced liquidity, higher spread
    "volume_5x": -1,               # raw volume surge at entry: data shows −0.25 lift (noise, not edge)
    "bb_squeeze_breakout": -1,     # BB squeeze at entry: exhaustion signal, −0.18 lift
    "macd_bearish_cross": -1,      # histogram / zero-line crossed down in last 3-5 bars
    "bullish_candle": -1,          # bullish candle at entry predicts underperformance (−0.43 lift)
    "bearish_candle": -1,          # shooting_star / bearish_engulfing / bearish_marubozu on last bar
    "options_pcr_greed": -1,       # PCR < 0.5: extreme complacency (min OI required)
    "options_long_unwinding": -1,  # price down + OI down = longs exiting (min OI required)
    "pead_negative_surprise": -2,  # PROMOTED (Alpha Round Phase 2/5): a results filing whose reaction-day
                                    # close move is <=-3% (src.data.bse_announcements.classify_pead_reaction).
                                    # Backtested as a would-be-buy: n=1772, ret_lift=-0.55 -- correctly
                                    # NO-SHIPs as a buy (it's a bearish signal), but train (-0.45) and
                                    # holdout (-0.784) are BOTH negative -- a real, consistent disqualifier
                                    # candidate. (The harness's stored holdout_consistent field is False
                                    # here only because it checks positive-direction consistency; this is
                                    # a hand reconciliation, same convention as every other disqualifier's
                                    # sign-aware validation -- see scripts/validate_signals.py.) Weight
                                    # magnitude follows the same round(3x|ret_lift|) convention as buy-side
                                    # promotions: round(3 x 0.55) = 2. See outputs/event_backtest.json.
}

# Bearish event signals — feed the short pipeline (src/data/breakdowns.py, is_heavy_selling
# in volume.py, bulk_deal side tracking). Previously distribution_signal/options_short_buildup
# lived in DISQUALIFIER_WEIGHTS (penalizing buys only); moved here as their own positive-scoring
# table for sells, with buy-side protection now applied directly in agent.py (a bearish event
# firing on a BUY candidate should still hurt it, same net effect, explicit instead of implicit).
#
# actual_52w_breakdown/distribution_signal/heavy_selling zeroed 2026-07 (Phase 5): 156-week
# backtest (185,673 closed trades), SELL-direction-only, sign-stable NEGATIVE in both train
# and holdout 70/30 split -- actual_52w_breakdown -2.50%/-2.48%, heavy_selling -2.44%/-2.67%,
# distribution_signal -1.76%/-0.43% (return_pct lift vs baseline). Matches the independent
# downtrend_short_edge verdict in outputs/big_mover_analysis.json (short_edge_negative,
# n=892, net -5.41% vs downtrend longs +0.45%). The short thesis doesn't have a measured
# edge yet even in its textbook case (F&O downtrend breakdown) -- see agent.py
# SHORT_PIPELINE_LIVE. Re-run this analysis and reconsider once more data accumulates or
# the market regime shifts out of the current correction.
BEARISH_EVENT_WEIGHTS = {
    "actual_52w_breakdown": 0,     # was 3 — zeroed, see comment above
    "distribution_signal": 0,      # was 2 — zeroed, see comment above
    "heavy_selling": 0,            # was 2 — zeroed, see comment above
    "consolidation_breakdown": 2,  # broke down from a tight range (not OHLCV-tested this pass)
    "bulk_deal_fii_sell": 2,       # FII/DII bulk/block SELL deal (not OHLCV-derivable, untested)
    "options_short_buildup": 1,    # price down + OI up = fresh shorts entering
}

# Regime-dependent overrides — MERGED into (not replacing) SHORT_TERM_WEIGHTS /
# DISQUALIFIER_WEIGHTS / BEARISH_EVENT_WEIGHTS by key, see _compute_score. A key
# here only takes effect if it matches a live signal-map key exactly
# (src/scorer.py:_build_signal_map).
#
# Generated by: python scripts/calibrate_weights.py --by-regime (2026-07, Phase 5,
# 156w/185,673-closed-trade backtest). All 3 buckets shipped this time (n>=1000 and
# beats flat OOS in-bucket for all three — uptrend n=124995, ranging n=40415,
# downtrend n=20263). Two corrections applied to the script's raw paste-block before
# shipping, same class of bug Phase 2 fixed:
#   1. volume_surge -> volume_5x, distribution -> distribution_signal (backtest-only
#      column names that don't match live signal-map keys — would have been inert).
#   2. near_52w_high / near_52w_low dropped entirely — these are backtest-only
#      exploratory signals; scorer.py's live signal map only computes the discrete
#      actual_52w_breakout/breakdown (proximity alone was deliberately dropped as
#      "redundant + noisy" pre-Phase-5). A REGIME_WEIGHTS entry for either would be
#      permanently inert in production, same failure mode Phase 2 fixed.
#
# rs_quality_strong REMOVED from all 3 buckets (2026-07, Phase 2): the flat weight
# failed its own backtest (-1.47pp, n=16,907) and was zeroed above -- leaving these
# regime overrides (-2/-5/+5) would resurrect a proven-negative signal via the
# merge in _compute_score (short_term = {**SHORT_TERM_WEIGHTS, **override}).
# None = fall back to flat hand weights for that bucket.
REGIME_WEIGHTS: dict[str, dict[str, int] | None] = {
    "uptrend":   {"rsi_momentum": 2, "rs_vs_nifty": 1, "rsi_bearish_div": 2, "rsi_bullish_div": -1, "macd_bearish_cross": -2, "bb_squeeze_breakout": -2, "bullish_candle": -2, "bearish_candle": -2, "volume_5x": -2, "distribution_signal": -4, "heavy_selling": -5, "actual_52w_breakout": 1, "actual_52w_breakdown": -5},
    "ranging":   {"rsi_momentum": -2, "rs_vs_nifty": -2, "rsi_bearish_div": -3, "rsi_bullish_div": 1, "macd_bearish_cross": 1, "bb_squeeze_breakout": 1, "bullish_candle": -2, "bearish_candle": 3, "volume_5x": -2, "distribution_signal": -1, "heavy_selling": -3, "actual_52w_breakdown": -2},
    "downtrend": {"rsi_momentum": 5, "rs_vs_nifty": 5, "rsi_bearish_div": 5, "rsi_bullish_div": -5, "macd_bearish_cross": -5, "bullish_candle": 5, "bearish_candle": -5, "volume_5x": -3, "distribution_signal": -5, "heavy_selling": -5, "actual_52w_breakout": 5, "actual_52w_breakdown": -5},
}

# Lowered to 2 to prevent dropping single-signal stocks before Pass 2 options/SAST fetch
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
    promoter_data: dict | None = None,
    options_data: dict | None = None,
    technical_data: dict | None = None,
    delivery_data: dict | None = None,
    fo_ban_current: set[str] | None = None,
    announcements_data: dict | None = None,
    hot_sector_tickers: set[str] | None = None,
    sast_data: bool = False,
    fundamental_data: dict | None = None,
    breakdown_data: dict | None = None,
    pead_signal: str | None = None,
) -> dict[str, bool]:
    """Build a boolean signal map for a single ticker."""
    signals: dict[str, bool] = {
        k: False for k in list(SHORT_TERM_WEIGHTS) + list(SWING_WEIGHTS)
        + list(DISQUALIFIER_WEIGHTS) + list(BEARISH_EVENT_WEIGHTS)
    }

    # --- DISQUALIFIERS computed early (can short-circuit scoring logic) ---

    # F&O ban: ticker is currently on NSE's banned list — position-building is prohibited,
    # exits can be forced at unfavourable prices, and signals from this stock are unreliable.
    signals["f_group"] = ticker in (fo_ban_current or set())

    # F&O ban lifted: ticker was just removed from the ban — liquidity restored, fresh derivative
    # positions are now allowed. Pent-up institutional demand often causes a near-term price pop.
    signals["fo_ban_lifted"] = ticker in set(fo_ban_removed)

    # --- SHORT-TERM signals ---

    # Bulk/block deal by known FII/DII — side "" (column missing) counts as BUY,
    # preserving pre-side-tracking behavior.
    fii_dii_deals = [d for d in bulk_deals if d["ticker"] == ticker and d["is_fii_dii"]]
    signals["bulk_deal_fii_dii"]  = any(d.get("side", "") != "SELL" for d in fii_dii_deals)
    signals["bulk_deal_fii_sell"] = any(d.get("side", "") == "SELL" for d in fii_dii_deals)

    # Volume ≥ 2.5× 30-day average / heavy-selling down day
    if volume_data:
        signals["volume_5x"]          = volume_data.get("is_volume_surge", False)
        signals["distribution_signal"] = volume_data.get("is_distribution", False)
        signals["heavy_selling"]      = volume_data.get("is_heavy_selling", False)
        # Thin market: avg daily turnover < ₹5 crore — signal quality is unreliable
        avg_30d = volume_data.get("avg_30d_volume", 0) or 0
        price   = volume_data.get("today_close", 0) or 0
        daily_turnover = avg_30d * price if (avg_30d > 0 and price > 0) else 0
        signals["thin_market_extreme"] = bool(0 < daily_turnover < 1e7)    # < ₹1cr
        signals["thin_market_light"]   = bool(1e7 <= daily_turnover < 5e7) # ₹1cr–₹5cr

    # 52-week high. src/data/breakouts.py only emits a candidate when a ticker is
    # within ATH_PROXIMITY_PCT (5%) of its 52w high with a volume gate -- i.e. every
    # ticker with breakout_data IS a near_52w_high hit by construction (same
    # definition backtest.py measured: near_52w_high n=15,005 includes actual
    # breakouts n=7,138 as a strict subset). actual_52w_breakout separately scores
    # the subset that has actually broken through (proximity<=0).
    if breakout_data:
        signals["near_52w_high"]          = True
        signals["actual_52w_breakout"]    = breakout_data.get("actual_breakout", False)
        signals["consolidation_breakout"] = breakout_data.get("consolidation_breakout", False)

    # 52-week low breakdown — mirror of the breakout block above (src/data/breakdowns.py)
    if breakdown_data:
        signals["actual_52w_breakdown"]   = breakdown_data.get("actual_breakdown", False)
        signals["consolidation_breakdown"] = breakdown_data.get("consolidation_breakdown", False)

    # Options signals — PCR only at extremes; long_buildup gated by min OI in options.py
    if options_data:
        signals["options_pcr_fear"]        = options_data.get("pcr_fear", False)
        signals["options_pcr_greed"]       = options_data.get("pcr_greed", False)
        signals["options_long_buildup"]    = options_data.get("long_buildup", False)
        signals["options_short_buildup"]   = options_data.get("short_buildup", False)
        signals["options_short_covering"]  = options_data.get("short_covering", False)
        signals["options_long_unwinding"]  = options_data.get("long_unwinding", False)

    # Technical signals (pass 3 only — after enrich_candidate_context)
    if technical_data:
        signals["rsi_momentum"]        = technical_data.get("rsi_momentum", False)
        signals["rs_vs_nifty"]         = technical_data.get("rs_vs_nifty", False)
        signals["rsi_bearish_div"]     = technical_data.get("rsi_bearish_div", False)
        signals["rsi_bullish_div"]     = technical_data.get("rsi_bullish_div", False)
        signals["macd_bearish_cross"]  = technical_data.get("macd_bearish_cross", False)
        signals["bb_squeeze_breakout"] = technical_data.get("bb_squeeze_breakout", False)
        signals["bullish_candle"]       = technical_data.get("bullish_candle", False)
        signals["bearish_candle"]       = technical_data.get("bearish_candle", False)
        signals["weekly_trend_aligned"] = technical_data.get("weekly_trend_aligned", False)
        signals["rs_quality_strong"]    = technical_data.get("rs_quality_strong", False)

    # PEAD v2 (SOTA Round Phase 1): computed EARLY from src.data.bse_announcements.
    # fetch_pead_signals, passed as its own `pead_signal` param (not technical_data)
    # so it's available in EVERY scoring pass, not just pass 3 -- a pure-PEAD
    # ticker (pead_positive_surprise alone scores 1, below MIN_SCORE=2) would
    # never reach top-20 OHLCV enrichment otherwise, making the validated edge
    # structurally untradeable on its own. Single source of truth: this is the
    # ONLY place these two keys are set.
    signals["pead_positive_surprise"] = pead_signal == "positive"
    signals["pead_negative_surprise"] = pead_signal == "negative"

    # Delivery volume signal from NSE bhav copy
    if delivery_data:
        signals["delivery_surge"] = delivery_data.get("delivery_surge", False)

    # Corporate announcements filed post-3:30 PM (pre-market alpha)
    if announcements_data:
        ann = announcements_data.get(ticker, {})
        signals["results_beat_announced"] = ann.get("results_beat_announced", False)
        signals["buyback_announced"]      = ann.get("buyback_announced", False)
        signals["contract_win"]           = ann.get("contract_win", False)
        signals["dividend_announced"]     = ann.get("dividend_announced", False)

    # Sector rotation: ticker is in top-2 performing NSE sector index (5d return)
    signals["sector_in_momentum"] = ticker in (hot_sector_tickers or set())

    # --- SWING signals ---

    # Upcoming results — weak informational flag; real PEAD requires confirmed post-beat data
    ticker_results = [r for r in results_calendar if r["ticker"] == ticker]
    if ticker_results:
        signals["results_due"] = True

    # Promoter / insider buying
    if promoter_data:
        signals["promoter_buying"] = promoter_data.get("promoter_bought", False)

    # SAST filing: SEBI mandatory disclosure for creeping acquisitions ≥5% stake
    signals["sast_insider_buying"] = bool(sast_data)

    # Fundamental quality from yfinance .info (soft gate — missing data → neutral)
    if fundamental_data:
        signals["fundamental_strong"] = bool(fundamental_data.get("fundamental_strong", False))
        signals["fundamental_weak"]   = bool(fundamental_data.get("fundamental_weak",   False))

    return signals


def _compute_score(
    signals: dict[str, bool],
    market_regime: str = "normal",
    nifty_trend: str = "ranging",
) -> tuple[int, str]:
    score = 0
    timeframe = "1-2d"

    # Regime override merges into (not replaces) the hand weights, and only touches
    # keys belonging to the table they're meant for — see REGIME_WEIGHTS comment.
    override   = REGIME_WEIGHTS.get(nifty_trend) or {}
    short_term = {**SHORT_TERM_WEIGHTS, **{k: v for k, v in override.items() if k in SHORT_TERM_WEIGHTS}}
    disq       = {**DISQUALIFIER_WEIGHTS, **{k: v for k, v in override.items() if k in DISQUALIFIER_WEIGHTS}}
    bearish    = {**BEARISH_EVENT_WEIGHTS, **{k: v for k, v in override.items() if k in BEARISH_EVENT_WEIGHTS}}

    for sig, weight in short_term.items():
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

    for sig, weight in bearish.items():
        if signals.get(sig):
            score += weight

    for sig, weight in disq.items():
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
    fo_ban_current: set[str] | None = None,
    market_regime: str = "normal",
    announcements: list[dict] | None = None,
    hot_sector_tickers: set[str] | None = None,
    sast_signals: dict[str, bool] | None = None,
    breadth_label: str = "neutral",
    nifty_trend: str = "ranging",
    fundamental_signals: dict[str, dict] | None = None,
    breakdowns: list[dict] | None = None,
    pead_signals: dict[str, str] | None = None,
) -> list[dict]:
    """
    Score every unique ticker across all data sources and return qualifying candidates.

    market_regime:       "low_vol" | "normal" | "high_vol" — high_vol applies -1 to all scores.
    breadth_label:       "strong" | "neutral" | "weak" — weak applies -1 (stacks with high_vol).
    nifty_trend:         "uptrend" | "ranging" | "downtrend" — selects REGIME_WEIGHTS override.
    fundamental_signals: {ticker: {fundamental_strong, fundamental_weak, ...}} from fundamentals.py
    breakdowns:          52-week-low candidates from src/data/breakdowns.py — short-pipeline source.
    pead_signals:        {ticker: "positive"|"negative"} from src.data.bse_announcements.
                          fetch_pead_signals — computed EARLY (before pass 1) so a pure-PEAD
                          ticker enters the pool in every pass, not just post-enrichment.
    """
    breakdowns = breakdowns or []

    # Collect all unique tickers across all data sources
    all_tickers: set[str] = set()
    all_tickers.update(d["ticker"] for d in bulk_deals)
    all_tickers.update(d["ticker"] for d in volume_gainers)
    all_tickers.update(fo_ban_removed)  # ban-lifted tickers enter scoring universe
    all_tickers.update(r["ticker"] for r in results_calendar)
    all_tickers.update(b["ticker"] for b in breakouts)
    all_tickers.update(b["ticker"] for b in breakdowns)
    all_tickers.update((pead_signals or {}).keys())  # PEAD-only tickers enter the pool

    # Build lookup dicts for O(1) access
    volume_map: dict[str, dict] = {d["ticker"]: d for d in volume_gainers}
    breakout_map: dict[str, dict] = {b["ticker"]: b for b in breakouts}
    breakdown_map: dict[str, dict] = {b["ticker"]: b for b in breakdowns}

    # Build announcements lookup: {ticker: {signal_key: True, ...}}
    # A ticker can have multiple announcements; last one per signal_key wins (all are True anyway).
    ann_map: dict[str, dict] = {}
    for ann in (announcements or []):
        t = ann.get("ticker", "")
        if t:
            all_tickers.add(t)  # announced ticker enters universe even without price data
            ann_map.setdefault(t, {})[ann.get("signal_key", "")] = True

    candidates = []
    for ticker in all_tickers:
        vol  = volume_map.get(ticker)
        brk  = breakout_map.get(ticker)
        bkd  = breakdown_map.get(ticker)
        prom = (promoter_signals or {}).get(ticker)
        opts = (options_signals or {}).get(ticker)
        tech = (technical_signals or {}).get(ticker)
        deliv = (delivery_signals or {}).get(ticker)
        ann   = ann_map.get(ticker)
        sast  = bool((sast_signals or {}).get(ticker, False))
        fund  = (fundamental_signals or {}).get(ticker)
        pead  = (pead_signals or {}).get(ticker)

        signals = _build_signal_map(
            ticker, bulk_deals, vol, fo_ban_removed, results_calendar, brk,
            promoter_data=prom, options_data=opts, technical_data=tech,
            delivery_data=deliv, fo_ban_current=fo_ban_current,
            announcements_data={ticker: ann} if ann else None,
            hot_sector_tickers=hot_sector_tickers,
            sast_data=sast,
            fundamental_data=fund,
            breakdown_data=bkd,
            pead_signal=pead,
        )

        score, timeframe = _compute_score(signals, market_regime, nifty_trend)

        # High-volatility regime: all signals are noisier — reduce score by 1
        if market_regime == "high_vol":
            score = max(0, score - 1)
        # Weak breadth: market internals deteriorating — reduce score by 1 (stacks with high_vol)
        if breadth_label == "weak":
            score = max(0, score - 1)

        active = _active_signals(signals)
        disqs = _disqualifiers(signals)

        if score < MIN_SCORE or len(active) < MIN_SIGNALS:
            continue

        today_close = (vol or brk or bkd or {}).get("today_close")
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
            "52w_low": (bkd or {}).get("52w_low"),
            "consolidation_breakout": signals.get("consolidation_breakout", False),
            "consolidation_breakdown": signals.get("consolidation_breakdown", False),
            "promoter_pct": (prom or {}).get("promoter_pct"),
            "promoter_bought": (prom or {}).get("promoter_bought", False),
            "options_pcr": (opts or {}).get("pcr"),
            "delivery_pct": (deliv or {}).get("delivery_pct"),
            "delivery_spike_pp": (deliv or {}).get("delivery_spike_pp"),
            # Pass through raw deal info for LLM context
            "bulk_deals": [d for d in bulk_deals if d["ticker"] == ticker],
            "upcoming_results": [r for r in results_calendar if r["ticker"] == ticker],
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    print(f"[scorer] {len(all_tickers)} tickers evaluated, {len(candidates)} qualified")
    return candidates
