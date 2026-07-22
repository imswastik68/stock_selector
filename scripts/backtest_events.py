"""
Historical event-signal backtest (Phase 6) -- the genuinely unexplored alpha
hunt in this codebase. "Event signals have no free archive" (scripts/backtest.py's
own docstring) is FALSE for at least two of the three tested here:

  delivery: NSE publishes per-date bhavcopy with delivery % going back years
            (src/data/delivery.py already fetches single days from this URL).
  SAST:     per-symbol date-range queryable via pnsea (src/data/sast.py).
  bulk deals: NSE has a historical API endpoint -- flakier, attempted anyway.

Each source is independent and fail-soft: a blocked/failed source reports
UNTESTABLE and does not block the other two ("all three, fail-soft" -- user
decision). Point-in-time discipline throughout: a signal on date T only uses
data <= T; forward returns are evaluated from T+1 (mirrors simulate_raw's own
entry window, trade_sim.py:67).

Imports the LIVE thresholds/filters from src/data/* -- never copies them --
so this can't silently drift from what the scorer actually fires on:
  src.data.delivery._MIN_DELIVERY_PCT/_MIN_DELIVERY_SPIKE/_make_session
  src.data.bulk_deals._is_fii_dii
  src.data.sast._pnsea_available/_sast_one

Ship gate (pre-declared, not tuned after seeing results):
  SHIP     = n >= 500 AND wr_lift_pp > 0 AND ret_lift > 0 AND a 70/30
             chronological holdout split is sign-consistent (both halves positive).
  NO-SHIP  = n >= 500 but lifts/holdout fail.
  INSUFFICIENT_SAMPLE = n < 500 -- "weight stays 0", not a verdict on the signal.
  UNTESTABLE = the source itself couldn't produce data (NSE blocked, pnsea
               missing, ...).

suggested_weight = max(0, min(5, round(3 * ret_lift))) -- same convention as
scripts/calibrate_weights.py's _simple_weights (return-lift, not WR-lift).

Usage:
  python scripts/backtest_events.py                     # all three sources
  python scripts/backtest_events.py --source delivery   # one source
  python scripts/backtest_events.py --weeks 156 --skip-download  # re-analyze from cache only
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import time
import warnings
from datetime import date, timedelta, time as _dtime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import pandas as pd
    import numpy as np
    import requests
except ImportError:
    sys.exit("Run: pip install pandas numpy requests")

from mine_big_movers import _load_ticker_df, _turnover_ok  # noqa: E402
from backtest import _compute_atr  # noqa: E402
from factor_backtest import _load_universe  # noqa: E402
from src.technicals import compute_entry_levels, _rsi_series  # noqa: E402
from src.trade_sim import simulate_raw  # noqa: E402
from src.costs import round_trip_cost_pct  # noqa: E402
from src.data.delivery import _MIN_DELIVERY_PCT, _MIN_DELIVERY_SPIKE, _make_session as _delivery_session  # noqa: E402
from src.data.bulk_deals import _is_fii_dii  # noqa: E402
from src.data.sast import _pnsea_available, _sast_one  # noqa: E402
from src.data.options import _MIN_CALL_OI  # noqa: E402 (same liquidity gate the live scorer uses)
from src.data.bse_announcements import _match_signal as _ann_match_signal  # noqa: E402
from src.data.reversal import RET_3D_THRESHOLD as _REV_RET_3D, RSI2_THRESHOLD as _REV_RSI2  # noqa: E402
from src.data.insider import (  # noqa: E402
    PIT_URL as _PIT_URL, PIT_PRIME_URL as _PIT_PRIME_URL, _matches_filter as _pit_matches_filter,
)

OUTPUTS       = ROOT / "outputs"
NIFTY_CSV     = ROOT / "cache" / "backtest_nifty.csv"
BHAV_CACHE    = ROOT / "cache" / "bhavcopy"
FO_CACHE      = ROOT / "cache" / "fo_bhavcopy"
SAST_CACHE    = ROOT / "cache" / "sast_events"
BULK_CACHE    = ROOT / "cache" / "bulk_deals_hist"
ANN_CACHE     = ROOT / "cache" / "announcements_hist"
SHP_CACHE     = ROOT / "cache" / "shareholding_hist"
PIT_CACHE     = ROOT / "cache" / "pit_chunks"
OUT_FILE      = OUTPUTS / "event_backtest.json"

_BHAV_URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date}.csv"
# F&O UDiFF bhavcopy -- the free historical archive for options OI (see the
# nse-free-historical-data-map memory). Reconstructs PCR + OI-buildup signals.
_FO_BHAV_URL = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date}_F_0000.csv.zip"
_BULK_HIST_URL = "https://www.nseindia.com/api/historical/bulk-deals?from={frm}&to={to}"
_BULK_PRIME_URL = "https://www.nseindia.com/report-detail/display-bulk-and-block-deals"
_ANN_URL = "https://www.nseindia.com/api/corporate-announcements?index=equities&from_date={frm}&to_date={to}"
_ANN_PRIME_URL = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
_SHP_URL = "https://www.nseindia.com/api/corporate-share-holdings-master?index=equities&symbol={sym}"
_SHP_PRIME_URL = "https://www.nseindia.com/companies-listing/corporate-filings-shareholding-pattern"

MIN_N_SHIP = 500
MIN_BHAV_DAYS = 20  # below this, too few sessions for a meaningful surge/lift read
DIRECTION  = "buy"  # all sources here are buy-side signals


# ── shared: trading-day calendar ─────────────────────────────────────────────

def trading_days(weeks: int) -> list[date]:
    """Trading days from the cached Nifty index -- avoids hammering NSE to
    figure out which calendar days were holidays/weekends."""
    if not NIFTY_CSV.exists():
        sys.exit(f"Not found: {NIFTY_CSV}. Run scripts/backtest.py first to build the cache.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_csv(NIFTY_CSV, index_col=0)
        # older yfinance multi-index CSV dumps leave a 2-row "Ticker/Date" header
        # artifact as data rows (non-numeric Close, unparseable index) -- strip
        # them (mirrors scripts/mine_big_movers.py:_load_nifty exactly).
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df[df["Close"].notna()]
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()]
    cutoff = pd.Timestamp(date.today() - timedelta(weeks=weeks))
    idx = df.index[df.index >= cutoff]
    return sorted({ts.date() for ts in idx})


# ── shared: forward-return evaluation + baseline + verdict ──────────────────

def _evaluate_one(ticker: str, as_of: date, ohlcv: dict, cost_multiplier: float = 1.0) -> dict | None:
    """Point-in-time entry/exit simulation for one (ticker, as_of) event,
    mirroring scripts/backtest.py's own trade mechanics exactly (2xATR stop,
    3xATR target, simulate_raw's first-2-bar entry rule).

    cost_multiplier: stress-test knob (SOTA Round Phase 2) -- 1.0 (default,
    zero behavior change for every existing caller) applies the normal
    round-trip cost; 2.0 doubles it, for a "does the edge survive 2x
    transaction costs" robustness check."""
    df = ohlcv.get(ticker)
    if df is None:
        return None
    as_of_ts = pd.Timestamp(as_of)
    df_slice = df[df.index <= as_of_ts]
    if len(df_slice) < 60:
        return None
    if not _turnover_ok(df, as_of_ts):
        return None

    close = float(df_slice["Close"].iloc[-1])
    atr = _compute_atr(df_slice)
    if atr <= 0 or close <= 0:
        return None

    levels = compute_entry_levels(close, atr, DIRECTION)
    if not levels:
        return None
    try:
        ez = levels["entry_zone"].replace("₹", "")
        entry_lo, entry_hi = (float(x) for x in ez.split("-"))
        entry_mid = (entry_lo + entry_hi) / 2
        sl = float(levels["stop_loss"].replace("₹", ""))
        t1 = float(levels["target_1"].replace("₹", ""))
        t2 = float(levels["target_2"].replace("₹", ""))
    except Exception:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = simulate_raw(entry_lo, entry_hi, entry_mid, sl, t1, DIRECTION, as_of_ts, df,
                            t2=t2, cost_pct=round_trip_cost_pct(DIRECTION) * cost_multiplier)

    if not sim.get("triggered") or sim.get("outcome") not in ("t1_hit", "t2_hit", "sl_hit", "timeout"):
        return None

    return {
        "ticker": ticker, "as_of": as_of, "outcome": sim["outcome"],
        "return_pct": sim["return_pct"], "win": sim["outcome"] in ("t1_hit", "t2_hit"),
    }


def _evaluate_one_time_exit(ticker: str, as_of: date, ohlcv: dict, hold_days: int = 5,
                             cost_multiplier: float = 1.0) -> dict | None:
    """5-trading-day time-exit diagnostic (SOTA Round Phase 2, reversal_v2's
    second exit model) -- no ATR stop/target, pure close-to-close holding
    period. Entry = close of the first bar after as_of (matches simulate_raw's
    own entry-fill convention); exit = close `hold_days` trading bars later."""
    df = ohlcv.get(ticker)
    if df is None:
        return None
    as_of_ts = pd.Timestamp(as_of)
    future = df[df.index > as_of_ts]
    if len(future) < hold_days + 1:
        return None
    if not _turnover_ok(df, as_of_ts):
        return None

    entry_px = float(future["Close"].iloc[0])
    exit_px = float(future["Close"].iloc[hold_days])
    if entry_px <= 0:
        return None

    ret_pct = (exit_px / entry_px - 1) * 100 - round_trip_cost_pct(DIRECTION) * cost_multiplier
    return {"ticker": ticker, "as_of": as_of, "return_pct": round(ret_pct, 3), "win": ret_pct > 0}


def _sample_baseline(events: list[tuple[date, str]], ohlcv: dict, universe: list[str],
                      cap_per_date: int = 20, seed: int = 42) -> list[tuple[date, str]]:
    """Same-date, turnover-decile-matched non-signal names -- so the lift is
    measured against comparably liquid stocks on the SAME day, not the whole
    universe's average (which would conflate the signal with a liquidity bias)."""
    rng = np.random.default_rng(seed)
    signal_by_date: dict[date, set[str]] = {}
    for d, t in events:
        signal_by_date.setdefault(d, set()).add(t)

    baseline: list[tuple[date, str]] = []
    for d, signal_tickers in signal_by_date.items():
        as_of_ts = pd.Timestamp(d)
        candidates = [t for t in universe if t not in signal_tickers]
        rng.shuffle(candidates)
        picked = 0
        for t in candidates:
            if picked >= cap_per_date:
                break
            df = ohlcv.get(t)
            if df is None:
                continue
            df_slice = df[df.index <= as_of_ts]
            if len(df_slice) < 60 or not _turnover_ok(df, as_of_ts):
                continue
            baseline.append((d, t))
            picked += 1
    return baseline


def _lift_stats(signal_results: list[dict], baseline_results: list[dict]) -> dict:
    n = len(signal_results)
    n_baseline = len(baseline_results)
    wr = sum(r["win"] for r in signal_results) / n * 100 if n else None
    ret = float(np.mean([r["return_pct"] for r in signal_results])) if n else None
    wr_b = sum(r["win"] for r in baseline_results) / n_baseline * 100 if n_baseline else None
    ret_b = float(np.mean([r["return_pct"] for r in baseline_results])) if n_baseline else None
    return {
        "n": n, "wr_pct": round(wr, 2) if wr is not None else None,
        "ret_pct": round(ret, 3) if ret is not None else None,
        "n_baseline": n_baseline,
        "wr_baseline_pct": round(wr_b, 2) if wr_b is not None else None,
        "ret_baseline_pct": round(ret_b, 3) if ret_b is not None else None,
        "wr_lift_pp": round(wr - wr_b, 2) if (wr is not None and wr_b is not None) else None,
        "ret_lift": round(ret - ret_b, 3) if (ret is not None and ret_b is not None) else None,
    }


def suggested_weight(ret_lift: float | None) -> int:
    if ret_lift is None:
        return 0
    return max(0, min(5, round(3 * ret_lift)))


def verdict_for(signal_results: list[dict], baseline_results: list[dict]) -> dict:
    stats = _lift_stats(signal_results, baseline_results)
    n = stats["n"]
    if n < MIN_N_SHIP:
        stats["verdict"] = "INSUFFICIENT_SAMPLE"
        stats["suggested_weight"] = 0
        return stats

    # 70/30 chronological holdout -- both halves must independently show a
    # positive return lift, not just the pooled average (guards against one
    # lucky half dragging up an otherwise-flat signal).
    sr_sorted = sorted(signal_results, key=lambda r: r["as_of"])
    br_sorted = sorted(baseline_results, key=lambda r: r["as_of"])
    split_s = int(len(sr_sorted) * 0.7)
    split_b = int(len(br_sorted) * 0.7)
    train_stats = _lift_stats(sr_sorted[:split_s], br_sorted[:split_b])
    holdout_stats = _lift_stats(sr_sorted[split_s:], br_sorted[split_b:])
    holdout_consistent = (
        train_stats["ret_lift"] is not None and train_stats["ret_lift"] > 0
        and holdout_stats["ret_lift"] is not None and holdout_stats["ret_lift"] > 0
    )
    stats["holdout_consistent"] = holdout_consistent
    stats["train_ret_lift"] = train_stats["ret_lift"]
    stats["holdout_ret_lift"] = holdout_stats["ret_lift"]

    ships = (
        stats["wr_lift_pp"] is not None and stats["wr_lift_pp"] > 0
        and stats["ret_lift"] is not None and stats["ret_lift"] > 0
        and holdout_consistent
    )
    stats["verdict"] = "SHIP" if ships else "NO-SHIP"
    stats["suggested_weight"] = suggested_weight(stats["ret_lift"]) if ships else 0
    return stats


def evaluate_events(events: list[tuple[date, str]], ohlcv: dict, universe: list[str],
                     cost_multiplier: float = 1.0) -> dict:
    if not events:
        return {"n": 0, "verdict": "INSUFFICIENT_SAMPLE", "suggested_weight": 0}

    signal_results = []
    for d, t in events:
        r = _evaluate_one(t, d, ohlcv, cost_multiplier=cost_multiplier)
        if r is not None:
            signal_results.append(r)

    baseline_events = _sample_baseline(events, ohlcv, universe)
    baseline_results = []
    for d, t in baseline_events:
        r = _evaluate_one(t, d, ohlcv, cost_multiplier=cost_multiplier)
        if r is not None:
            baseline_results.append(r)

    return verdict_for(signal_results, baseline_results)


def _evaluate_events_time_exit(events: list[tuple[date, str]], ohlcv: dict, universe: list[str],
                                hold_days: int = 5, cost_multiplier: float = 1.0) -> dict:
    """Same event set, evaluated with the 5-day time-exit model instead of the
    ATR-target template (SOTA Round Phase 2 robustness check (d): both exit
    models must show a positive lift). Returns pooled _lift_stats only (no
    SHIP/NO-SHIP verdict of its own -- reversal_v2_verdict combines it)."""
    if not events:
        return {"n": 0, "ret_lift": None, "wr_lift_pp": None}

    signal_results = []
    for d, t in events:
        r = _evaluate_one_time_exit(t, d, ohlcv, hold_days=hold_days, cost_multiplier=cost_multiplier)
        if r is not None:
            signal_results.append(r)

    baseline_events = _sample_baseline(events, ohlcv, universe)
    baseline_results = []
    for d, t in baseline_events:
        r = _evaluate_one_time_exit(t, d, ohlcv, hold_days=hold_days, cost_multiplier=cost_multiplier)
        if r is not None:
            baseline_results.append(r)

    return _lift_stats(signal_results, baseline_results)


def _multi_split_stats(signal_results: list[dict], baseline_results: list[dict],
                        n_splits: int = 5) -> list[dict]:
    """Chronological date-range split into n_splits equal-width buckets
    (SOTA Round Phase 2 gate (a)) -- unlike the 70/30 holdout's index-based
    slicing, this splits by calendar time so each bucket represents a
    genuinely distinct period, not just "however many events happened to
    fall in the last 30% of the list"."""
    if not signal_results:
        return []
    dates = sorted(r["as_of"] for r in signal_results)
    start, end = dates[0], dates[-1]
    total_days = (end - start).days
    if total_days <= 0:
        return [_lift_stats(signal_results, baseline_results)]

    bucket_days = total_days / n_splits
    boundaries = [start + timedelta(days=round(bucket_days * i)) for i in range(n_splits + 1)]
    boundaries[-1] = end + timedelta(days=1)  # inclusive of the final date

    splits = []
    for i in range(n_splits):
        lo, hi = boundaries[i], boundaries[i + 1]
        sr = [r for r in signal_results if lo <= r["as_of"] < hi]
        br = [r for r in baseline_results if lo <= r["as_of"] < hi]
        stats = _lift_stats(sr, br)
        stats["split_start"] = lo.isoformat()
        stats["split_end"] = (hi - timedelta(days=1)).isoformat()
        splits.append(stats)
    return splits


def reversal_v2_verdict(events: list[tuple[date, str]], ohlcv: dict, universe: list[str]) -> dict:
    """
    SOTA Round Phase 2 -- the pre-declared, STRICTER re-test of the no-200DMA-
    filter reversal variant that shipped every Alpha-Round gate but was
    excluded from promotion as diagnostic-only (anti-gate-shopping). All four
    checks below were pre-declared before this function was run against real
    data:
      (a) 5 chronological splits: pooled-vs-baseline ret_lift > 0 in >= 4/5
          splits AND in the most recent split.
      (b) pooled wr_lift_pp > 0.
      (c) survives 2x transaction costs: pooled ret_lift still > 0.
      (d) both exit models positive: the standard ATR-template AND the
          5-day time exit.
    NO-SHIP if n < MIN_N_SHIP (same floor as every other signal here) or any
    check fails. This is the LAST pre-declared attempt for this signal family
    -- the Alpha-Round primary (with a 200DMA filter) already NO-SHIPped.
    """
    if not events:
        return {"n": 0, "verdict": "INSUFFICIENT_SAMPLE", "suggested_weight": 0}

    signal_results = []
    for d, t in events:
        r = _evaluate_one(t, d, ohlcv)
        if r is not None:
            signal_results.append(r)

    baseline_events = _sample_baseline(events, ohlcv, universe)
    baseline_results = []
    for d, t in baseline_events:
        r = _evaluate_one(t, d, ohlcv)
        if r is not None:
            baseline_results.append(r)

    pooled = _lift_stats(signal_results, baseline_results)
    n = pooled["n"]
    if n < MIN_N_SHIP:
        pooled["verdict"] = "INSUFFICIENT_SAMPLE"
        pooled["suggested_weight"] = 0
        return pooled

    # (a) 5 chronological splits
    splits = _multi_split_stats(signal_results, baseline_results, n_splits=5)
    n_positive_splits = sum(1 for s in splits if s["ret_lift"] is not None and s["ret_lift"] > 0)
    most_recent_positive = bool(splits and splits[-1]["ret_lift"] is not None and splits[-1]["ret_lift"] > 0)
    splits_pass = n_positive_splits >= 4 and most_recent_positive

    # (b) pooled win-rate lift
    wr_pass = pooled["wr_lift_pp"] is not None and pooled["wr_lift_pp"] > 0

    # (c) 2x transaction costs
    signal_2x = []
    for d, t in events:
        r = _evaluate_one(t, d, ohlcv, cost_multiplier=2.0)
        if r is not None:
            signal_2x.append(r)
    baseline_2x = []
    for d, t in baseline_events:
        r = _evaluate_one(t, d, ohlcv, cost_multiplier=2.0)
        if r is not None:
            baseline_2x.append(r)
    stats_2x = _lift_stats(signal_2x, baseline_2x)
    cost_2x_pass = stats_2x["ret_lift"] is not None and stats_2x["ret_lift"] > 0

    # (d) both exit models
    time_exit_stats = _evaluate_events_time_exit(events, ohlcv, universe, hold_days=5)
    atr_template_pass = pooled["ret_lift"] is not None and pooled["ret_lift"] > 0
    time_exit_pass = time_exit_stats["ret_lift"] is not None and time_exit_stats["ret_lift"] > 0
    both_exits_pass = atr_template_pass and time_exit_pass

    ships = splits_pass and wr_pass and cost_2x_pass and both_exits_pass

    pooled["verdict"] = "SHIP" if ships else "NO-SHIP"
    pooled["suggested_weight"] = min(3, suggested_weight(pooled["ret_lift"])) if ships else 0
    pooled["gate_detail"] = {
        "splits": splits,
        "n_positive_splits": n_positive_splits,
        "most_recent_split_positive": most_recent_positive,
        "splits_pass": splits_pass,
        "wr_pass": wr_pass,
        "cost_2x_ret_lift": stats_2x["ret_lift"],
        "cost_2x_pass": cost_2x_pass,
        "time_exit_ret_lift": time_exit_stats["ret_lift"],
        "time_exit_pass": time_exit_pass,
        "atr_template_pass": atr_template_pass,
        "both_exits_pass": both_exits_pass,
    }
    return pooled


def _load_ohlcv_cache(universe: list[str]) -> dict:
    ohlcv = {}
    for t in universe:
        df = _load_ticker_df(t)
        if df is not None:
            ohlcv[t] = df
    return ohlcv


# ── source (a): delivery surge ───────────────────────────────────────────────

def _bhav_path(d: date) -> Path:
    return BHAV_CACHE / f"{d.strftime('%Y%m%d')}.csv.gz"


def _bhav_miss_path(d: date) -> Path:
    return BHAV_CACHE / f"{d.strftime('%Y%m%d')}.miss"


def _download_bhavcopy(days: list[date]) -> None:
    BHAV_CACHE.mkdir(parents=True, exist_ok=True)
    session = _delivery_session()
    consecutive_failures = 0
    n_downloaded = 0
    for i, d in enumerate(days):
        if _bhav_path(d).exists() or _bhav_miss_path(d).exists():
            continue
        if i > 0 and i % 50 == 0:
            session = _delivery_session()
        url = _BHAV_URL.format(date=d.strftime("%d%m%Y"))
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code != 200 or len(resp.content) < 1000:
                _bhav_miss_path(d).touch()
                consecutive_failures = 0  # a 404 (holiday) is expected, not a failure
                continue
            df = pd.read_csv(io.BytesIO(resp.content))
            df.columns = df.columns.str.strip().str.upper()
            if not {"SYMBOL", "SERIES", "DELIV_PER"}.issubset(df.columns):
                _bhav_miss_path(d).touch()
                continue
            df = df[df["SERIES"].str.strip() == "EQ"][["SYMBOL", "DELIV_PER"]]
            with gzip.open(_bhav_path(d), "wt") as f:
                df.to_csv(f, index=False)
            n_downloaded += 1
            consecutive_failures = 0
        except Exception as exc:
            consecutive_failures += 1
            print(f"[backtest_events] delivery {d} fetch error: {exc}")
            if consecutive_failures >= 25:
                print("[backtest_events] delivery: 25 consecutive failures, aborting download "
                      "(partial cache kept)")
                return
        time.sleep(0.7)
    print(f"[backtest_events] delivery: downloaded {n_downloaded} new bhavcopy files")


def _load_bhavcopy(d: date) -> dict[str, float] | None:
    path = _bhav_path(d)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt") as f:
            df = pd.read_csv(f)
        out = {}
        for _, row in df.iterrows():
            try:
                out[f"{str(row['SYMBOL']).strip()}.NS"] = float(row["DELIV_PER"])
            except (ValueError, TypeError):
                continue
        return out
    except Exception:
        return None


def collect_delivery_events(weeks: int, skip_download: bool) -> tuple[list[tuple[date, str]], str | None]:
    days = trading_days(weeks)
    if not skip_download:
        _download_bhavcopy(days)

    day_maps: dict[date, dict[str, float]] = {}
    for d in days:
        m = _load_bhavcopy(d)
        if m is not None:
            day_maps[d] = m

    if len(day_maps) < MIN_BHAV_DAYS:
        return [], f"only {len(day_maps)} bhavcopy days cached — UNTESTABLE this pass"

    sorted_days = sorted(day_maps.keys())
    events: list[tuple[date, str]] = []
    for i, d in enumerate(sorted_days):
        if i < 4:
            continue  # need 4 prior sessions for the baseline average
        today_map = day_maps[d]
        prior_maps = [day_maps[sorted_days[j]] for j in range(i - 4, i)]
        for ticker, today_pct in today_map.items():
            prior_pcts = [pm.get(ticker, 0.0) for pm in prior_maps]
            avg_prior = sum(prior_pcts) / len(prior_pcts)
            spike_pp = today_pct - avg_prior
            if today_pct >= _MIN_DELIVERY_PCT and spike_pp >= _MIN_DELIVERY_SPIKE:
                events.append((d, ticker))
    return events, None


# ── source (options): PCR + OI buildup from F&O bhavcopy ─────────────────────
# The live scorer's option signals (src/data/options.py) read a live option
# chain -- no history. But the F&O UDiFF bhavcopy is a per-day archive with the
# same per-strike OI, so PCR/buildup can be reconstructed exactly (same
# _MIN_CALL_OI liquidity gate, same PCR>1.5 / <0.5 and price-vs-OI buildup
# logic as _parse_option_chain). Buy-side signals only: options_pcr_fear,
# options_long_buildup, options_short_covering.

def _fo_path(d: date) -> Path:
    return FO_CACHE / f"{d.strftime('%Y%m%d')}.csv.gz"


def _fo_miss_path(d: date) -> Path:
    return FO_CACHE / f"{d.strftime('%Y%m%d')}.miss"


def _download_fo_bhavcopy(days: list[date]) -> None:
    """Fetch each day's F&O UDiFF bhavcopy, aggregate STOCK-option rows to one
    slim row per underlying (put_oi, call_oi, net_oi_chg, ul_price), gzip-cache.
    Aggregating at download time keeps the cache tiny (~1 row/symbol/day vs the
    ~42k raw rows). Resumable + .miss sentinels + fail-soft, mirroring
    _download_bhavcopy."""
    FO_CACHE.mkdir(parents=True, exist_ok=True)
    session = _delivery_session()
    consecutive_failures = 0
    n_downloaded = 0
    for i, d in enumerate(days):
        if _fo_path(d).exists() or _fo_miss_path(d).exists():
            continue
        if i > 0 and i % 50 == 0:
            session = _delivery_session()
        url = _FO_BHAV_URL.format(date=d.strftime("%Y%m%d"))
        try:
            resp = session.get(url, timeout=20)
            if resp.status_code != 200 or len(resp.content) < 1000:
                _fo_miss_path(d).touch()
                consecutive_failures = 0  # holiday/no-file is expected, not a failure
                continue
            import zipfile
            z = zipfile.ZipFile(io.BytesIO(resp.content))
            df = pd.read_csv(z.open(z.namelist()[0]))
            # STO = stock options; STF = stock futures (used for underlying price)
            opts = df[df["FinInstrmTp"] == "STO"].copy()
            if opts.empty:
                _fo_miss_path(d).touch()
                continue
            opts["OpnIntrst"] = pd.to_numeric(opts["OpnIntrst"], errors="coerce").fillna(0)
            opts["ChngInOpnIntrst"] = pd.to_numeric(opts["ChngInOpnIntrst"], errors="coerce").fillna(0)
            opts["UndrlygPric"] = pd.to_numeric(opts["UndrlygPric"], errors="coerce")
            rows = []
            for sym, g in opts.groupby("TckrSymb"):
                ce, pe = g[g["OptnTp"] == "CE"], g[g["OptnTp"] == "PE"]
                rows.append({
                    "symbol": sym,
                    "call_oi": float(ce["OpnIntrst"].sum()),
                    "put_oi": float(pe["OpnIntrst"].sum()),
                    "net_oi_chg": float(g["ChngInOpnIntrst"].sum()),
                    "ul_price": float(g["UndrlygPric"].dropna().iloc[0]) if g["UndrlygPric"].notna().any() else 0.0,
                })
            slim = pd.DataFrame(rows)
            with gzip.open(_fo_path(d), "wt") as f:
                slim.to_csv(f, index=False)
            n_downloaded += 1
            consecutive_failures = 0
        except Exception as exc:
            consecutive_failures += 1
            print(f"[backtest_events] fo {d} fetch error: {exc}")
            if consecutive_failures >= 25:
                print("[backtest_events] options: 25 consecutive failures, aborting download "
                      "(partial cache kept)")
                return
        time.sleep(0.7)
    print(f"[backtest_events] options: downloaded {n_downloaded} new F&O bhavcopy files")


def _load_fo(d: date) -> dict[str, dict] | None:
    path = _fo_path(d)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt") as f:
            df = pd.read_csv(f)
        return {f"{str(r['symbol']).strip()}.NS": {
            "call_oi": r["call_oi"], "put_oi": r["put_oi"],
            "net_oi_chg": r["net_oi_chg"], "ul_price": r["ul_price"],
        } for _, r in df.iterrows()}
    except Exception:
        return None


def collect_options_events(weeks: int, skip_download: bool) -> tuple[dict[str, list[tuple[date, str]]], str | None]:
    """Returns ({signal_name: [(date, ticker), ...]}, reason). Reconstructs the
    three BUY-side option signals per (date, ticker) exactly as the live scorer
    does, using the prior session's underlying price for buildup direction."""
    days = trading_days(weeks)
    if not skip_download:
        _download_fo_bhavcopy(days)

    day_maps: dict[date, dict[str, dict]] = {}
    for d in days:
        m = _load_fo(d)
        if m is not None:
            day_maps[d] = m

    if len(day_maps) < MIN_BHAV_DAYS:
        return {}, f"only {len(day_maps)} F&O bhavcopy days cached — UNTESTABLE this pass"

    sorted_days = sorted(day_maps.keys())
    out: dict[str, list[tuple[date, str]]] = {
        # Buy-side signals (weight 0, tested 2026-07: both NO-SHIP).
        "options_pcr_fear": [], "options_long_buildup": [], "options_short_covering": [],
        # Added 2026-07-22. These three carry LIVE NON-ZERO weights
        # (options_pcr_greed -1, options_long_unwinding -1 in DISQUALIFIER_WEIGHTS;
        # options_short_buildup +1 in BEARISH_EVENT_WEIGHTS) yet had never been
        # tested -- they were unreachable while the option-chain fetch was broken,
        # so nobody noticed. Rebuilding the fetch on the bhavcopy re-activates
        # them (17% of the 210-name liquid F&O universe would fire pcr_greed on
        # 2026-07-22 alone), which makes validating them a prerequisite, not a
        # nice-to-have.
        #
        # All three are evaluated BUY-side, like the harness's other
        # disqualifiers (see pead_negative_surprise): a penalty is justified by a
        # NEGATIVE ret_lift -- the signal marking names that go on to
        # underperform. A positive lift would mean the live penalty is backwards.
        "options_pcr_greed": [], "options_long_unwinding": [], "options_short_buildup": [],
    }
    for i, d in enumerate(sorted_days):
        if i == 0:
            continue  # need a prior session for the price-direction of buildup
        today_map = day_maps[d]
        prev_map = day_maps[sorted_days[i - 1]]
        for ticker, cur in today_map.items():
            call_oi, put_oi = cur["call_oi"], cur["put_oi"]
            if call_oi < _MIN_CALL_OI:   # same liquidity gate as the live fetcher
                continue
            pcr = put_oi / call_oi if call_oi > 0 else None
            prev = prev_map.get(ticker)
            price_up = prev is not None and prev["ul_price"] > 0 and cur["ul_price"] > prev["ul_price"]
            price_dn = prev is not None and prev["ul_price"] > 0 and cur["ul_price"] < prev["ul_price"]
            oi_up = cur["net_oi_chg"] > 0
            if pcr is not None and pcr > 1.5:
                out["options_pcr_fear"].append((d, ticker))
            if pcr is not None and pcr < 0.5:
                out["options_pcr_greed"].append((d, ticker))
            if price_up and oi_up:
                out["options_long_buildup"].append((d, ticker))
            if price_up and not oi_up:
                out["options_short_covering"].append((d, ticker))
            if price_dn and oi_up:
                out["options_short_buildup"].append((d, ticker))
            if price_dn and not oi_up:
                out["options_long_unwinding"].append((d, ticker))
    return out, None


# ── source (reversal): short-horizon mean reversion ──────────────────────────
# Deliberately the OPPOSITE bet to the trend/momentum cluster (near_52w_high,
# rsi_momentum, ...) that dominates the shipped signals so far -- buys sharp
# oversold dips instead of strength, so a real edge here would be genuinely
# orthogonal, not just another way of saying "price is going up." Pre-declared
# primary definition (Alpha Round Phase 1) -- do not loosen after seeing
# results: 3-day return <= -7%, close > 200DMA (dip inside an uptrend, not a
# falling knife), RSI(2) < 10. Needs no network: derived from the same OHLCV
# cache scripts/backtest.py already maintains.

def collect_reversal_events(weeks: int, ohlcv: dict) -> tuple[list[tuple[date, str]], str | None]:
    cutoff = pd.Timestamp(date.today() - timedelta(weeks=weeks))
    events: list[tuple[date, str]] = []
    for ticker, df in ohlcv.items():
        if len(df) < 200:
            continue
        close = df["Close"]
        sma200 = close.rolling(200).mean()
        ret3 = close / close.shift(3) - 1
        rsi2 = _rsi_series(close, period=2)
        mask = (df.index >= cutoff) & (ret3 <= _REV_RET_3D) & (close > sma200) & (rsi2 < _REV_RSI2)
        for ts in df.index[mask]:
            if not _turnover_ok(df, ts):
                continue
            events.append((ts.date(), ticker))
    return events, None


def collect_reversal_diag_events(weeks: int, ohlcv: dict) -> tuple[list[tuple[date, str]], str | None]:
    """The reversal_oversold_v2 condition (no 200DMA uptrend filter) -- module
    constants (_REV_RET_3D/_REV_RSI2) are imported from src.data.reversal,
    the SAME module the live scanner uses, so live and backtest can't
    silently drift. Still also used, unchanged, as the diagnostic-only event
    set under --source reversal (see the '_diag' suffix skip in main()'s
    promotion-candidates loop) -- reversal_v2's own stricter gate
    (reversal_v2_verdict) is what makes THIS event set promotable under
    --source reversal2."""
    cutoff = pd.Timestamp(date.today() - timedelta(weeks=weeks))
    events: list[tuple[date, str]] = []
    for ticker, df in ohlcv.items():
        if len(df) < 10:
            continue
        close = df["Close"]
        ret3 = close / close.shift(3) - 1
        rsi2 = _rsi_series(close, period=2)
        mask = (df.index >= cutoff) & (ret3 <= _REV_RET_3D) & (rsi2 < _REV_RSI2)
        for ts in df.index[mask]:
            if not _turnover_ok(df, ts):
                continue
            events.append((ts.date(), ticker))
    return events, None


# ── source (b): SAST filings ─────────────────────────────────────────────────

def collect_sast_events(weeks: int, universe: list[str]) -> tuple[list[tuple[date, str]], str | None]:
    if not _pnsea_available():
        return [], "pnsea not installed — UNTESTABLE (run: pip install pnsea)"

    SAST_CACHE.mkdir(parents=True, exist_ok=True)
    end = date.today()
    start = end - timedelta(weeks=weeks)

    events: list[tuple[date, str]] = []
    n_symbols = 0
    for ticker in universe:
        symbol = ticker.replace(".NS", "")
        cache_file = SAST_CACHE / f"{symbol}.json"
        cached = {"ranges_done": [], "records": []}
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text())
            except Exception:
                pass

        chunk_start = start
        new_records = []
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=90), end)
            range_key = f"{chunk_start.isoformat()}:{chunk_end.isoformat()}"
            if range_key not in cached["ranges_done"]:
                from_dt = chunk_start.strftime("%d-%m-%Y")
                to_dt = chunk_end.strftime("%d-%m-%Y")
                _, hit = _sast_one(symbol, from_dt, to_dt)
                if hit:
                    # _sast_one only returns a bool, not filing dates -- record
                    # the chunk midpoint as an approximate event date (the
                    # backtest's forward-return window is measured in weeks,
                    # so a few days of date imprecision within a 90-day chunk
                    # is immaterial to the lift measurement).
                    mid = chunk_start + (chunk_end - chunk_start) / 2
                    new_records.append(mid.isoformat())
                cached["ranges_done"].append(range_key)
                time.sleep(0.5)
            chunk_start = chunk_end

        cached["records"] = list(set(cached.get("records", []) + new_records))
        cache_file.write_text(json.dumps(cached))
        n_symbols += 1
        for rec in cached["records"]:
            events.append((date.fromisoformat(rec), ticker))

    if n_symbols == 0:
        return [], "no NIFTY500 symbols processed — UNTESTABLE"
    return events, None


# ── source (announcements): historical corporate filings ─────────────────────
# The live scorer reads a recent-only feed; the same endpoint accepts
# from_date/to_date, so the full PEAD-style history is fetchable. Classified
# with the LIVE keyword map (_ann_match_signal, imported) so it can't drift.

def _make_www_session(prime_url: str) -> requests.Session:
    """NSE www-API session, warmed by hitting a listing page first (the API host
    503s a cold session -- see nse-free-historical-data-map memory)."""
    session = _delivery_session()
    try:
        session.get(prime_url, timeout=15)
        time.sleep(1.0)
    except Exception:
        pass
    return session


def _www_get_json(session, url: str, retries: int = 4):
    """GET a www-API JSON with retries for the intermittent 503. Returns parsed
    JSON or None."""
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=25)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(2.5)
    return None


def _fetch_announcement_items(weeks: int, universe: list[str]) -> tuple[list[dict], str | None]:
    """Chunked, cache-first fetch of RAW NSE corporate-announcement rows for
    the window, restricted to `universe` symbols. Shared by
    collect_announcement_events (per-signal classification) and
    collect_pead_events (results-only reaction-day analysis) so the network/
    cache logic lives in exactly one place."""
    ANN_CACHE.mkdir(parents=True, exist_ok=True)
    end = date.today()
    start = end - timedelta(weeks=weeks)
    uni = {t.replace(".NS", "") for t in universe}
    session = _make_www_session(_ANN_PRIME_URL)

    items: list[dict] = []
    any_success = False
    total_failures = 0
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=45), end)
        cache_file = ANN_CACHE / f"ann_{chunk_start.isoformat()}_{chunk_end.isoformat()}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text()); any_success = True
            except Exception:
                data = None
        else:
            url = _ANN_URL.format(frm=chunk_start.strftime("%d-%m-%Y"), to=chunk_end.strftime("%d-%m-%Y"))
            data = _www_get_json(session, url)
            if data is None:
                total_failures += 1
            else:
                cache_file.write_text(json.dumps(data)); any_success = True
                session = _make_www_session(_ANN_PRIME_URL)  # re-warm between chunks
            time.sleep(1.0)

        if total_failures >= 4 and not any_success:
            return [], "corporate-announcements API unreachable (503) — UNTESTABLE this pass"

        rows = data if isinstance(data, list) else (data or {}).get("data", data)
        if isinstance(rows, list):
            for item in rows:
                if not isinstance(item, dict):
                    continue
                sym = str(item.get("symbol") or "").strip().upper()
                if sym not in uni:
                    continue
                items.append(item)
        chunk_start = chunk_end

    if not any_success:
        return [], "corporate-announcements API unreachable (503) — UNTESTABLE this pass"
    return items, None


def collect_announcement_events(weeks: int, universe: list[str]) -> tuple[dict[str, list[tuple[date, str]]], str | None]:
    """Returns ({signal_key: [(date, ticker), ...]}, reason). Classifies each
    raw filing with the live _ann_match_signal."""
    items, reason = _fetch_announcement_items(weeks, universe)
    if reason:
        return {}, reason

    out: dict[str, list[tuple[date, str]]] = {
        "results_beat_announced": [], "buyback_announced": [],
        "contract_win": [], "dividend_announced": [],
    }
    for item in items:
        sym = str(item.get("symbol") or "").strip().upper()
        desc = str(item.get("desc") or "")
        headline = str(item.get("attchmntText") or desc)
        sig = _ann_match_signal(desc, headline)
        if sig is None or sig not in out:
            continue
        dt_str = str(item.get("an_dt") or item.get("dt") or "").strip()
        try:
            d = pd.to_datetime(dt_str[:11], format="%d-%b-%Y").date()
        except Exception:
            continue
        out[sig].append((d, f"{sym}.NS"))
    return out, None


# ── source (PEAD v2): surprise-proxied post-earnings drift ───────────────────
# results_beat_announced (fires on ANY results filing, unconditioned) backtested
# at ret_lift=-1.234, n=8824 -- proven net-harmful (see nse-free-historical-
# data-map / event_backtest.json). The literature-supported version conditions
# on SURPRISE. Without a paid consensus-estimate feed, the standard retail
# proxy is the announcement-day price reaction itself: drift continues in the
# direction of the initial reaction. Pre-declared (Alpha Round Phase 2, do not
# tune after seeing results):
#   reaction day R = announcement day if filed by 15:30 IST, else next trading day
#   r_R = close_R / close_{R-1} - 1
#   r_R >= +3%  -> pead_positive_surprise   (candidate BUY signal)
#   r_R <= -3%  -> pead_negative_surprise   (candidate DISQUALIFIER -- validated
#                  by NEGATIVE lift, mirroring the sign-aware disqualifier logic
#                  in validate_signals.py; NOT expected to "ship" as a buy)
# Entry is R+1 (evaluate_events' as_of=R, whose entry window is as_of+1) --
# never R itself, since the reaction has already happened by R's close.

_PEAD_REACTION_CUTOFF = _dtime(15, 30)


def collect_pead_events(weeks: int, universe: list[str], ohlcv: dict) -> tuple[dict[str, list[tuple[date, str]]], str | None]:
    items, reason = _fetch_announcement_items(weeks, universe)
    if reason:
        return {}, reason

    day_set = trading_days(weeks + 4)  # pad so a late R can still resolve to R+1
    day_index = {d: i for i, d in enumerate(day_set)}

    def _next_trading_day(d: date) -> date | None:
        for td in day_set:
            if td > d:
                return td
        return None

    out: dict[str, list[tuple[date, str]]] = {
        "pead_positive_surprise": [], "pead_negative_surprise": [],
    }
    for item in items:
        desc = str(item.get("desc") or "")
        headline = str(item.get("attchmntText") or desc)
        if _ann_match_signal(desc, headline) != "results_beat_announced":
            continue
        sym = str(item.get("symbol") or "").strip().upper()
        ticker = f"{sym}.NS"
        df = ohlcv.get(ticker)
        if df is None:
            continue
        dt_str = str(item.get("an_dt") or "").strip()
        try:
            filed = pd.to_datetime(dt_str, format="%d-%b-%Y %H:%M:%S")
        except Exception:
            continue
        filed_date = filed.date()
        if filed.time() <= _PEAD_REACTION_CUTOFF and filed_date in day_index:
            r_day = filed_date
        else:
            r_day = _next_trading_day(filed_date)
        if r_day is None or r_day not in day_index:
            continue
        r_idx = day_index[r_day]
        if r_idx == 0:
            continue
        r_prev = day_set[r_idx - 1]
        df_r = df[df.index == pd.Timestamp(r_day)]
        df_prev = df[df.index == pd.Timestamp(r_prev)]
        if df_r.empty or df_prev.empty:
            continue
        close_r, close_prev = float(df_r["Close"].iloc[0]), float(df_prev["Close"].iloc[0])
        if close_prev <= 0:
            continue
        r_react = close_r / close_prev - 1
        if r_react >= 0.03:
            out["pead_positive_surprise"].append((r_day, ticker))
        elif r_react <= -0.03:
            out["pead_negative_surprise"].append((r_day, ticker))
    return out, None


# ── source (promoter): quarterly shareholding pattern ────────────────────────
# promoter_buying live = promoter % increased quarter-over-quarter. The
# shareholding-master API returns ~90 historical quarters per symbol with
# pr_and_prgrp (promoter+group %) and the quarter-end date -- reconstructs the
# same "promoter accumulating" event historically.

def collect_promoter_events(weeks: int, universe: list[str]) -> tuple[list[tuple[date, str]], str | None]:
    SHP_CACHE.mkdir(parents=True, exist_ok=True)
    cutoff = date.today() - timedelta(weeks=weeks)
    session = _make_www_session(_SHP_PRIME_URL)

    events: list[tuple[date, str]] = []
    any_success = False
    total_failures = 0
    for idx, ticker in enumerate(universe):
        sym = ticker.replace(".NS", "")
        cache_file = SHP_CACHE / f"{sym}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text()); any_success = True
            except Exception:
                data = None
        else:
            data = _www_get_json(session, _SHP_URL.format(sym=sym))
            if data is None:
                total_failures += 1
            else:
                cache_file.write_text(json.dumps(data)); any_success = True
            if idx % 50 == 49:
                session = _make_www_session(_SHP_PRIME_URL)
            time.sleep(0.6)

        if total_failures >= 5 and not any_success:
            return [], "shareholding-master API unreachable (503) — UNTESTABLE this pass"

        rows = data if isinstance(data, list) else (data or {}).get("data", data)
        if not isinstance(rows, list):
            continue
        # parse (quarter_end_date, promoter_pct), sort chronologically
        quarters = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                qd = pd.to_datetime(str(row.get("date")).strip(), format="%d-%b-%Y").date()
                pr = float(str(row.get("pr_and_prgrp")).strip())
            except (ValueError, TypeError):
                continue
            quarters.append((qd, pr))
        quarters.sort()
        # event = quarter where promoter % rose vs the prior quarter, dated at the
        # broadcast (filing) date so forward returns start after the market knew.
        for i in range(1, len(quarters)):
            qd, pr = quarters[i]
            if qd < cutoff:
                continue
            if pr > quarters[i - 1][1] + 0.01:  # a real increase, not rounding noise
                events.append((qd, ticker))

    if not any_success:
        return [], "shareholding-master API unreachable (503) — UNTESTABLE this pass"
    return events, None


# ── source (PIT): promoter open-market purchases ─────────────────────────────
# sast_insider_buying (fires on ANY SAST filing -- pledges, creeping
# acquisitions, inter-se transfers) backtested net-harmful (ret_lift=-0.407,
# n=2010). The research-supported version is promoters buying their OWN stock
# in the OPEN MARKET -- a genuine "insiders think it's cheap" signal, only
# isolatable via the PIT (Prohibition of Insider Trading) disclosure feed.
# The filter logic (_pit_matches_filter, imported above) is the SINGLE SOURCE
# OF TRUTH shared with src.data.insider's live fetcher (SOTA Round Phase 3) --
# vocabulary confirmed live against the corporates-pit API before writing it
# (2026-07-05, 1192-row sample): acqMode has exactly one value meaning a
# genuine open-market trade -- "Market Purchase" -- distinct from "Off
# Market", "Preferential Offer", "Gift", "ESOP", "Pledge Creation", "Scheme of
# Amalgamation/...". personCategory is "Promoters" or "Promoter Group".
# tdpTransactionType is "Buy" or "Sell".


def _fetch_pit_items(weeks: int) -> tuple[list[dict], str | None]:
    """Chunked, cache-first fetch of RAW NSE insider-trading (PIT) rows for the
    window. Global feed (one call covers every listed company, unlike
    shareholding-master's per-symbol calls). A 0-row response is treated as
    suspect (observed live: one short window returned 200 with zero rows while
    a longer window had data) -- retried once before being trusted/cached."""
    PIT_CACHE.mkdir(parents=True, exist_ok=True)
    end = date.today()
    start = end - timedelta(weeks=weeks)
    session = _make_www_session(_PIT_PRIME_URL)

    items: list[dict] = []
    any_success = False
    total_failures = 0
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=45), end)
        cache_file = PIT_CACHE / f"pit_{chunk_start.isoformat()}_{chunk_end.isoformat()}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text()); any_success = True
            except Exception:
                data = None
        else:
            url = _PIT_URL.format(frm=chunk_start.strftime("%d-%m-%Y"), to=chunk_end.strftime("%d-%m-%Y"))
            data = _www_get_json(session, url)
            rows_now = data if isinstance(data, list) else (data or {}).get("data", data) if data else None
            if data is not None and isinstance(rows_now, list) and len(rows_now) == 0:
                time.sleep(1.5)
                data_retry = _www_get_json(session, url, retries=2)
                rows_retry = (data_retry if isinstance(data_retry, list)
                              else (data_retry or {}).get("data", data_retry) if data_retry else None)
                if isinstance(rows_retry, list) and len(rows_retry) > 0:
                    data = data_retry
            if data is None:
                total_failures += 1
            else:
                cache_file.write_text(json.dumps(data)); any_success = True
                session = _make_www_session(_PIT_PRIME_URL)
            time.sleep(1.0)

        if total_failures >= 4 and not any_success:
            return [], "corporates-pit API unreachable (503) — UNTESTABLE this pass"

        rows = data if isinstance(data, list) else (data or {}).get("data", data) if data else None
        if isinstance(rows, list):
            items.extend(r for r in rows if isinstance(r, dict))
        chunk_start = chunk_end

    if not any_success:
        return [], "corporates-pit API unreachable (503) — UNTESTABLE this pass"
    return items, None


def collect_pit_events(weeks: int, universe: list[str]) -> tuple[list[tuple[date, str]], str | None]:
    """promoter_open_mkt_buy. Event date = intimation date (`date` field), NOT
    the acquisition date (`acqfromDt`/`acqtoDt`) -- the trade is public
    knowledge only once intimated; using the acquisition date would be
    look-ahead."""
    items, reason = _fetch_pit_items(weeks)
    if reason:
        return [], reason

    uni = {t.replace(".NS", "") for t in universe}
    seen: set[tuple] = set()
    events: list[tuple[date, str]] = []
    for row in items:
        sym = str(row.get("symbol") or "").strip().upper()
        if sym not in uni:
            continue
        if not _pit_matches_filter(row):
            continue
        dt_str = str(row.get("date") or "").strip()
        try:
            d = pd.to_datetime(dt_str[:11], format="%d-%b-%Y").date()
        except Exception:
            continue
        key = (d, sym)
        if key in seen:
            continue
        seen.add(key)
        events.append((d, f"{sym}.NS"))
    return events, None


# ── source (c): historical bulk deals ────────────────────────────────────────

def _make_bulk_session() -> requests.Session:
    session = _delivery_session()
    try:
        session.get(_BULK_PRIME_URL, timeout=10)
        time.sleep(0.3)
    except Exception:
        pass
    return session


def collect_bulk_events(weeks: int) -> tuple[list[tuple[date, str]], str | None]:
    BULK_CACHE.mkdir(parents=True, exist_ok=True)
    end = date.today()
    start = end - timedelta(weeks=weeks)
    session = _make_bulk_session()

    events: list[tuple[date, str]] = []
    total_failures = 0
    chunk_start = start
    any_success = False
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=180), end)
        cache_file = BULK_CACHE / f"bulk_{chunk_start.isoformat()}_{chunk_end.isoformat()}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                any_success = True
            except Exception:
                data = None
        else:
            url = _BULK_HIST_URL.format(frm=chunk_start.strftime("%d-%m-%Y"), to=chunk_end.strftime("%d-%m-%Y"))
            try:
                resp = session.get(url, timeout=15)
                if resp.status_code != 200:
                    total_failures += 1
                    data = None
                else:
                    data = resp.json()
                    cache_file.write_text(json.dumps(data))
                    any_success = True
            except Exception as exc:
                print(f"[backtest_events] bulk deals fetch error: {exc}")
                total_failures += 1
                data = None
            time.sleep(1.0)

        if total_failures >= 3 and not any_success:
            return [], "bulk_deal_fii_dii UNTESTABLE from free archives — weight stays 0 until live archive matures"

        if data:
            rows = data.get("data", data) if isinstance(data, dict) else data
            if isinstance(rows, list):
                for row in rows:
                    symbol = str(row.get("symbol") or row.get("SYMBOL") or "").strip().upper()
                    actor = str(row.get("clientName") or row.get("CLIENT_NAME") or "").strip()
                    side = str(row.get("buySell") or row.get("BD_BUY_SELL") or "").strip().upper()
                    date_str = str(row.get("date") or row.get("DATE1") or "").strip()
                    if not symbol or not actor or not date_str:
                        continue
                    if not _is_fii_dii(actor) or side == "SELL":
                        continue
                    try:
                        d = pd.to_datetime(date_str, dayfirst=True).date()
                    except Exception:
                        continue
                    events.append((d, f"{symbol}.NS"))
        chunk_start = chunk_end

    if not any_success:
        return [], "bulk_deal_fii_dii UNTESTABLE from free archives — weight stays 0 until live archive matures"
    return events, None


# ── source (rebalance): index-reconstitution drift ───────────────────────────
# Forced passive buying between announcement and effective date -- the Nifty
# family reconstitutes semi-annually (effective the day after the last
# trading day of March/September), announced ~4 weeks ahead. This is an
# EVENT STUDY, not the ATR-target trade template used everywhere else: the
# drift window spans WEEKS (announce -> effective), so entry/exit are fixed
# to that window, not a stop/target. Data lives in data/index_changes.csv
# (hand-compiled from public press coverage, checked into git for
# auditability -- see its header for sourcing).
#
# Pre-declared low-frequency gate (event-study; the standard n>=500 bar is
# impossible for a twice-a-year event and wasn't designed for it -- this
# replacement is stricter PER-EVENT via a t-stat to compensate):
#   SHIP     = n>=100 AND mean_abnormal_pct>0 AND t_stat>=2 AND both
#              chronological halves (70/30) have positive mean abnormal return.
#   INSUFFICIENT_SAMPLE = n<100.
# DELETE events are measured and reported as a diagnostic (candidate future
# avoid/short signal) -- not gate-eligible this round.

INDEX_CHANGES_CSV = ROOT / "data" / "index_changes.csv"


def _load_index_changes() -> list[dict]:
    if not INDEX_CHANGES_CSV.exists():
        return []
    try:
        df = pd.read_csv(INDEX_CHANGES_CSV, comment="#")
    except Exception:
        return []
    rows: list[dict] = []
    for _, r in df.iterrows():
        try:
            a = pd.to_datetime(str(r["announce_date"])).date()
            e = pd.to_datetime(str(r["effective_date"])).date()
        except Exception:
            continue
        action = str(r.get("action", "")).strip().upper()
        sym = str(r.get("symbol", "")).strip().upper()
        if action not in ("ADD", "DELETE") or not sym or e <= a:
            continue
        rows.append({"announce_date": a, "effective_date": e,
                     "index": str(r.get("index", "")).strip(), "symbol": sym, "action": action})
    return rows


def collect_rebalance_events() -> tuple[list[dict], list[dict], str | None]:
    """Returns (add_events, delete_events, reason). Deduped by
    (announce_date, symbol) -- a stock entering multiple indices in the same
    cycle (e.g. NIFTY 500 + NIFTY Next 50 together) counts once."""
    rows = _load_index_changes()
    if not rows:
        return [], [], f"{INDEX_CHANGES_CSV.name} not found or empty — UNTESTABLE this pass"
    seen_add, seen_del = set(), set()
    adds, dels = [], []
    for r in rows:
        key = (r["announce_date"], r["symbol"])
        if r["action"] == "ADD":
            if key in seen_add:
                continue
            seen_add.add(key); adds.append(r)
        else:
            if key in seen_del:
                continue
            seen_del.add(key); dels.append(r)
    return adds, dels, None


def _trading_day_at_or_after(day_set: list[date], d: date) -> date | None:
    for td in day_set:
        if td >= d:
            return td
    return None


def _trading_day_at_or_before(day_set: list[date], d: date) -> date | None:
    result = None
    for td in day_set:
        if td > d:
            break
        result = td
    return result


def _close_on(df, d: date) -> float | None:
    row = df[df.index == pd.Timestamp(d)]
    return float(row["Close"].iloc[0]) if not row.empty else None


def _rebalance_window(day_set: list[date], a: date, e: date) -> tuple[date, date] | None:
    """Buy at A+1's close, sell at E-1's close (the drift window: the day
    after announcement through the day before the forced passive-fund trade)."""
    idx_after_a = None
    for i, td in enumerate(day_set):
        if td > a:
            idx_after_a = i
            break
    if idx_after_a is None:
        return None
    buy_day = day_set[idx_after_a]
    sell_day = _trading_day_at_or_before(day_set, e - timedelta(days=1))
    if sell_day is None or sell_day <= buy_day:
        return None
    return buy_day, sell_day


def _rebalance_abnormal_returns(events: list[dict], ohlcv: dict, universe: list[str],
                                 day_set: list[date], seed: int = 42,
                                 baseline_cap: int = 10) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for r in events:
        ticker = f"{r['symbol']}.NS"
        df = ohlcv.get(ticker)
        if df is None:
            continue
        window = _rebalance_window(day_set, r["announce_date"], r["effective_date"])
        if window is None:
            continue
        buy_day, sell_day = window
        buy_px, sell_px = _close_on(df, buy_day), _close_on(df, sell_day)
        if buy_px is None or sell_px is None or buy_px <= 0:
            continue
        sig_ret = (sell_px / buy_px - 1) * 100

        candidates = [t for t in universe if t != ticker]
        rng.shuffle(candidates)
        baseline_rets = []
        for t in candidates:
            if len(baseline_rets) >= baseline_cap:
                break
            bdf = ohlcv.get(t)
            if bdf is None or not _turnover_ok(bdf, pd.Timestamp(buy_day)):
                continue
            bp, sp = _close_on(bdf, buy_day), _close_on(bdf, sell_day)
            if bp is None or sp is None or bp <= 0:
                continue
            baseline_rets.append((sp / bp - 1) * 100)
        if not baseline_rets:
            continue
        abnormal = sig_ret - float(np.mean(baseline_rets))
        out.append({"as_of": r["announce_date"], "abnormal_pct": abnormal})
    return out


def _rebalance_verdict(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0, "verdict": "INSUFFICIENT_SAMPLE", "suggested_weight": 0}

    vals = np.array([r["abnormal_pct"] for r in results])
    mean_abn = float(np.mean(vals))
    std_abn = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    t_stat = mean_abn / (std_abn / np.sqrt(n)) if std_abn > 0 else 0.0

    sr_sorted = sorted(results, key=lambda r: r["as_of"])
    split = int(n * 0.7)
    train_vals = np.array([r["abnormal_pct"] for r in sr_sorted[:split]])
    holdout_vals = np.array([r["abnormal_pct"] for r in sr_sorted[split:]])
    train_mean = float(np.mean(train_vals)) if len(train_vals) else None
    holdout_mean = float(np.mean(holdout_vals)) if len(holdout_vals) else None
    holdout_consistent = (train_mean is not None and train_mean > 0
                          and holdout_mean is not None and holdout_mean > 0)

    if n < 100:
        verdict = "INSUFFICIENT_SAMPLE"
    else:
        ships = mean_abn > 0 and t_stat >= 2.0 and holdout_consistent
        verdict = "SHIP" if ships else "NO-SHIP"

    return {
        "n": n, "mean_abnormal_pct": round(mean_abn, 3), "t_stat": round(t_stat, 2),
        "train_abnormal_pct": round(train_mean, 3) if train_mean is not None else None,
        "holdout_abnormal_pct": round(holdout_mean, 3) if holdout_mean is not None else None,
        "holdout_consistent": holdout_consistent,
        "verdict": verdict,
        # the standard suggested_weight() formula (max(0,min(5,round(3*ret_lift))))
        # is calibrated for a single ATR-target trade's return, not a multi-week
        # event-study abnormal return -- deliberately NOT auto-converted here.
        # A SHIP verdict needs a human weight decision in Phase 5, same as any
        # first-of-its-kind measurement.
        "suggested_weight": None if verdict == "SHIP" else 0,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def _eval_single(signal: str, events, reason, ohlcv, universe, results: dict) -> None:
    if reason:
        print(f"  {signal}: UNTESTABLE: {reason}")
        results[signal] = {"verdict": "UNTESTABLE", "reason": reason, "suggested_weight": 0}
        return
    print(f"  {signal}: {len(events)} raw events, evaluating...")
    stats = evaluate_events(events, ohlcv, universe)
    results[signal] = stats
    print(f"  {signal}: n={stats['n']} wr_lift={stats.get('wr_lift_pp')}pp "
          f"ret_lift={stats.get('ret_lift')} -> {stats['verdict']}")


def _eval_multi(events_by_signal, reason, ohlcv, universe, results: dict) -> None:
    if reason:
        print(f"  UNTESTABLE: {reason}")
        for sig in (events_by_signal or {}):
            results[sig] = {"verdict": "UNTESTABLE", "reason": reason, "suggested_weight": 0}
        return
    for sig, events in events_by_signal.items():
        _eval_single(sig, events, None, ohlcv, universe, results)


def _load_existing_results() -> dict:
    """Merge into the prior run's results instead of overwriting, so running one
    --source doesn't wipe another source's verdict from event_backtest.json."""
    if OUT_FILE.exists():
        try:
            return json.loads(OUT_FILE.read_text()).get("signals", {})
        except Exception:
            pass
    return {}


def main(source: str, weeks: int, skip_download: bool) -> None:
    universe = _load_universe(None)
    print(f"[backtest_events] universe: {len(universe)} tickers")
    ohlcv = _load_ohlcv_cache(universe)
    print(f"[backtest_events] loaded {len(ohlcv)}/{len(universe)} tickers from OHLCV cache")

    results: dict[str, dict] = _load_existing_results()

    if source in ("delivery", "all"):
        print(f"\n[backtest_events] === delivery ===")
        events, reason = collect_delivery_events(weeks, skip_download)
        _eval_single("delivery_surge", events, reason, ohlcv, universe, results)

    if source in ("reversal", "all"):
        print(f"\n[backtest_events] === reversal (mean reversion) ===")
        events, reason = collect_reversal_events(weeks, ohlcv)
        _eval_single("reversal_oversold", events, reason, ohlcv, universe, results)
        diag_events, diag_reason = collect_reversal_diag_events(weeks, ohlcv)
        _eval_single("reversal_oversold_diag_no_trend", diag_events, diag_reason, ohlcv, universe, results)

    if source in ("reversal2", "all"):
        print(f"\n[backtest_events] === reversal v2 (pre-declared re-test, stricter gate) ===")
        events, reason = collect_reversal_diag_events(weeks, ohlcv)
        if reason:
            print(f"  reversal_oversold_v2: UNTESTABLE: {reason}")
            results["reversal_oversold_v2"] = {"verdict": "UNTESTABLE", "reason": reason, "suggested_weight": 0}
        else:
            print(f"  reversal_oversold_v2: {len(events)} raw events, evaluating (4 pre-declared checks)...")
            stats = reversal_v2_verdict(events, ohlcv, universe)
            results["reversal_oversold_v2"] = stats
            gd = stats.get("gate_detail", {})
            print(f"  reversal_oversold_v2: n={stats['n']} wr_lift={stats.get('wr_lift_pp')}pp "
                  f"ret_lift={stats.get('ret_lift')} -> {stats['verdict']}")
            if gd:
                print(f"    splits: {gd['n_positive_splits']}/5 positive, most_recent_positive="
                      f"{gd['most_recent_split_positive']} -> {gd['splits_pass']}")
                print(f"    wr_pass={gd['wr_pass']}  cost_2x_ret_lift={gd['cost_2x_ret_lift']} "
                      f"({gd['cost_2x_pass']})  time_exit_ret_lift={gd['time_exit_ret_lift']} "
                      f"({gd['time_exit_pass']})")

    if source in ("options", "all"):
        print(f"\n[backtest_events] === options (F&O bhavcopy) ===")
        events_by_signal, reason = collect_options_events(weeks, skip_download)
        _eval_multi(events_by_signal, reason, ohlcv, universe, results)

    if source in ("announcements", "all"):
        print(f"\n[backtest_events] === announcements ===")
        events_by_signal, reason = collect_announcement_events(weeks, universe)
        _eval_multi(events_by_signal, reason, ohlcv, universe, results)

    if source in ("pead", "all"):
        print(f"\n[backtest_events] === PEAD v2 (surprise-proxied) ===")
        events_by_signal, reason = collect_pead_events(weeks, universe, ohlcv)
        _eval_multi(events_by_signal, reason, ohlcv, universe, results)

    if source in ("promoter", "all"):
        print(f"\n[backtest_events] === promoter (shareholding) ===")
        events, reason = collect_promoter_events(weeks, universe)
        _eval_single("promoter_buying", events, reason, ohlcv, universe, results)

    if source in ("pit", "all"):
        print(f"\n[backtest_events] === PIT (promoter open-market buys) ===")
        events, reason = collect_pit_events(weeks, universe)
        _eval_single("promoter_open_mkt_buy", events, reason, ohlcv, universe, results)

    if source in ("sast", "all"):
        print(f"\n[backtest_events] === sast ===")
        events, reason = collect_sast_events(weeks, universe)
        _eval_single("sast_insider_buying", events, reason, ohlcv, universe, results)

    if source in ("rebalance", "all"):
        print(f"\n[backtest_events] === index rebalance (event-study) ===")
        adds, dels, reason = collect_rebalance_events()
        if reason:
            print(f"  UNTESTABLE: {reason}")
            results["index_rebalance_add"] = {"verdict": "UNTESTABLE", "reason": reason, "suggested_weight": 0}
        else:
            day_set = trading_days(9999)
            add_results = _rebalance_abnormal_returns(adds, ohlcv, universe, day_set)
            stats = _rebalance_verdict(add_results)
            results["index_rebalance_add"] = stats
            print(f"  index_rebalance_add: n={stats['n']} mean_abnormal={stats.get('mean_abnormal_pct')}pp "
                  f"t={stats.get('t_stat')} -> {stats['verdict']}")

            del_results = _rebalance_abnormal_returns(dels, ohlcv, universe, day_set)
            del_stats = _rebalance_verdict(del_results)
            results["index_rebalance_delete_diag"] = del_stats
            print(f"  index_rebalance_delete_diag (diagnostic only, not gate-eligible): "
                  f"n={del_stats['n']} mean_abnormal={del_stats.get('mean_abnormal_pct')}pp "
                  f"t={del_stats.get('t_stat')} -> {del_stats['verdict']}")

            # secondary diagnostic: standard ATR-target template, entry A+1 --
            # not gate-eligible (the plan's primary measurement is the A+1->E-1
            # event-study above; this is reported only for context).
            atr_events = [(r["announce_date"], f"{r['symbol']}.NS") for r in adds]
            atr_stats = evaluate_events(atr_events, ohlcv, universe)
            results["index_rebalance_add_atr_diag"] = atr_stats
            print(f"  index_rebalance_add_atr_diag: n={atr_stats['n']} "
                  f"ret_lift={atr_stats.get('ret_lift')} -> {atr_stats['verdict']} "
                  f"(diagnostic only, not gate-eligible)")

    if source in ("bulk", "all"):
        print(f"\n[backtest_events] === bulk deals ===")
        events, reason = collect_bulk_events(weeks)
        _eval_single("bulk_deal_fii_dii", events, reason, ohlcv, universe, results)

    out = {
        "meta": {
            "weeks": weeks, "generated": date.today().isoformat(), "min_n_ship": MIN_N_SHIP,
            "conventions": "wr/ret from simulate_raw closed trades, cost-netted; "
                           "baseline turnover-decile-matched same-date non-signal names",
        },
        "signals": results,
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[backtest_events] saved -> {OUT_FILE.name}")

    print(f"\n{'='*70}\n  PROMOTION CANDIDATES (paste into src/scorer.py if SHIP)\n{'='*70}")
    any_ship = False
    for sig, stats in results.items():
        if "_diag" in sig:
            continue  # diagnostic-only variant, never a promotion candidate
        if stats.get("verdict") == "SHIP":
            any_ship = True
            print(f'  "{sig}": {stats["suggested_weight"]},  # SHIP: ret_lift={stats["ret_lift"]} n={stats["n"]}')
    if not any_ship:
        print("  (none -- no signal cleared the ship gate this pass)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",
                     choices=["delivery", "options", "announcements", "promoter", "sast", "bulk",
                              "reversal", "reversal2", "pead", "pit", "rebalance", "all"],
                     default="all")
    ap.add_argument("--weeks", type=int, default=156)
    ap.add_argument("--skip-download", action="store_true",
                     help="Re-analyze from cache only, no new network fetches")
    args = ap.parse_args()
    main(args.source, args.weeks, args.skip_download)
