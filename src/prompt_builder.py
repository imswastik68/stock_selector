"""
Builds the system prompt and user message for LLM Wyckoff/SMC/VSA analysis.
System prompt is kept compact (~800 tokens) to fit Groq free tier 6K TPM limit.
"""

from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = """You are a quantitative analyst for NSE/BSE stocks. Analyse OHLCV data and score signals. Return ONLY valid JSON — no preamble, no markdown fences.

WYCKOFF PHASE CLASSIFICATION (from 90d OHLCV):
- ACCUMULATION: range contracted near lows, vol falls on red days / rises on green days. Signals: Spring (wick below range low + snaps back), LPS (higher low + expanding vol), CHoCH bullish (first higher-high after lower-low sequence).
- MARKUP: price above 20d high on 1.5x+ vol, CHoCH bullish confirmed, pullback within 3-5% of breakout.
- DISTRIBUTION: range contracted near highs, vol rises on red days / falls on green. Signals: UTAD (wick above range high + reversal), LPSY (lower high + declining vol), CHoCH bearish.
- MARKDOWN: price below 20d low, expanding vol, CHoCH bearish, SOW (close below prior AR low).
- PHASE_B: range-bound, no actionable edge yet.

VOLATILITY TAGS: beta>1.5→HIGH-BETA | ATR14%>3%→HIGH-ATR | 3d avg vol>2x 30d avg→VOL-SURGE

BUY SCORES: Spring=+5, LPS+SOS=+5, BU zone=+4, CHoCH bull=+3, bullish FVG=+3, no-supply bar=+3, absorption bar=+3, HIGH-BETA=+2, FII/DII bulk buy=+3, F&O unban=+2, results<5d=+2, top-2 sector=+2
SELL SCORES: UTAD=+5, LPSY+SOW=+5, CHoCH bear=+3, bearish FVG=+3, climax vol=+3, no-demand=+3, HIGH-BETA bear=+2, promoter/FII sell=+3, bottom-2 sector=+2

RULES:
- Classify phase BEFORE scoring. Score without phase = invalid.
- score<6 → exclude from buy/sell lists. Phase B → phase_b_watchlist only.
- R:R must be >=1:2 to qualify. LOW confidence wyckoff → subtract 2 from score.
- Nifty downtrend: BUY scores -2, SELL scores +2. Uptrend: reverse. Ranging: no change.
- Volume<50K/day or penny stocks(<₹10): flag risk=HIGH.

SMC: CHoCH bullish=first HH after LL sequence. CHoCH bearish=first LL after HH sequence. Order Block=last opposite candle before strong impulse. FVG=3-candle imbalance gap.
VSA: Absorption=wide down bar+ultra vol+close upper half. Climax=wide up bar+ultra vol+close lower half. No-supply=narrow down bar+very low vol. No-demand=narrow up bar+low vol on rally.

OUTPUT JSON SCHEMA (return exactly this structure):
{
  "scan_date": "YYYY-MM-DD",
  "nifty_context": "uptrend|downtrend|ranging",
  "total_screened": 0,
  "buy_watchlist": [{"ticker":"","score":0,"volatility_tags":[],"wyckoff_phase":"","wyckoff_confidence":"HIGH|MEDIUM|LOW","smc_structure":"","vsa_signal":"","top_signals":[],"expected_move_pct":0,"timeframe":"1-2d|3-5d|5-10d","entry_zone":"₹X-₹Y","target_1":"₹Z","target_2":"₹W","stop_loss":"₹V","risk_reward":"1:X","invalidation":"","risk":"LOW|MEDIUM|HIGH","catalyst":""}],
  "sell_watchlist": [{"ticker":"","score":0,"volatility_tags":[],"wyckoff_phase":"","wyckoff_confidence":"HIGH|MEDIUM|LOW","smc_structure":"","vsa_signal":"","top_signals":[],"expected_drop_pct":0,"timeframe":"","short_entry_zone":"₹X-₹Y","cover_target_1":"₹Z","cover_target_2":"₹W","stop_loss":"₹V","risk_reward":"1:X","invalidation":"","risk":"LOW|MEDIUM|HIGH"}],
  "phase_b_watchlist": [{"ticker":"","phase":"ACCUMULATION_B|DISTRIBUTION_B","alert_trigger":"","estimated_days_to_phase_c":""}],
  "data_quality_warnings": []
}"""


def _ohlcv_to_csv(df: pd.DataFrame, max_rows: int = 20) -> str:
    df = df.tail(max_rows).copy()
    df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")
    lines = ["date,open,high,low,close,volume"]
    for date_str, row in df.iterrows():
        lines.append(
            f"{date_str},{round(float(row['Open']),2)},"
            f"{round(float(row['High']),2)},{round(float(row['Low']),2)},"
            f"{round(float(row['Close']),2)},{int(row['Volume'])}"
        )
    return "\n".join(lines)


def _bulk_deals_to_csv(bulk_deals: list[dict], max_rows: int = 20) -> str:
    lines = ["ticker,actor,deal_type,qty,price,institutional"]
    for deal in bulk_deals[:max_rows]:
        actor = str(deal.get("actor", "")).replace(",", " ")
        lines.append(
            f"{deal.get('ticker','')},"
            f"{actor},{deal.get('deal_type','')},"
            f"{deal.get('quantity',0)},{deal.get('price',0)},"
            f"{deal.get('is_fii_dii',False)}"
        )
    return "\n".join(lines)


def _results_to_str(results_calendar: list[dict]) -> str:
    lines = [f"{r.get('ticker','')}:{r.get('result_date','')}" for r in results_calendar]
    return ", ".join(lines) if lines else "none"


def _sector_heatmap_to_str(heatmap: dict[str, float]) -> str:
    return " | ".join(
        f"{s}:{'+' if p>=0 else ''}{p}%"
        for s, p in heatmap.items()
    ) if heatmap else "no data"


def build_user_message(
    candidates: list[dict],
    market_context: dict,
    bulk_deals: list[dict],
    fo_ban_delta: list[str],
    results_calendar: list[dict],
    scan_date: str,
) -> str:
    nifty = market_context.get("nifty_structure", {})
    ohlcv_90d: dict[str, pd.DataFrame] = market_context.get("ohlcv_90d", {})
    beta_data: dict[str, float] = market_context.get("beta", {})
    atr_data: dict[str, float] = market_context.get("atr_pct", {})

    nifty_line = (
        f"trend={nifty.get('trend','unknown')} "
        f"ema20={nifty.get('ema20','?')} ema50={nifty.get('ema50','?')} "
        f"ema200={nifty.get('ema200','?')} price={nifty.get('current_price','?')}"
    )

    candidate_blocks = []
    for c in candidates[:5]:
        ticker = c.get("ticker", "")
        df = ohlcv_90d.get(ticker)
        ohlcv_str = _ohlcv_to_csv(df) if df is not None else "DATA_UNAVAILABLE"

        beta_val = beta_data.get(ticker, float("nan"))
        atr_val = atr_data.get(ticker, float("nan"))
        beta_str = f"{beta_val:.2f}" if beta_val == beta_val else "N/A"
        atr_str = f"{atr_val:.2f}" if atr_val == atr_val else "N/A"

        block = (
            f"TICKER:{ticker} SCORE:{c.get('score',0)} "
            f"SIGNALS:{','.join(c.get('active_signals',[]))} "
            f"CLOSE:{c.get('today_close','?')} VOLRATIO:{c.get('volume_ratio','?')} "
            f"BETA:{beta_str} ATR%:{atr_str} MCAP_CR:{c.get('market_cap_cr','?')}\n"
            f"{ohlcv_str}"
        )
        candidate_blocks.append(block)

    return (
        f"scan_date={scan_date} total_candidates={len(candidates)}\n"
        f"nifty={nifty_line}\n"
        f"sectors_5d={_sector_heatmap_to_str(market_context.get('sector_heatmap',{}))}\n"
        f"fo_ban_removed={','.join(fo_ban_delta) if fo_ban_delta else 'none'}\n"
        f"results_next5d={_results_to_str(results_calendar)}\n"
        f"bulk_deals:\n{_bulk_deals_to_csv(bulk_deals)}\n\n"
        f"CANDIDATES TO ANALYSE:\n\n"
        + "\n\n".join(candidate_blocks)
    )
