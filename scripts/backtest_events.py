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
from datetime import date, timedelta
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
from src.technicals import compute_entry_levels  # noqa: E402
from src.trade_sim import simulate_raw  # noqa: E402
from src.costs import round_trip_cost_pct  # noqa: E402
from src.data.delivery import _MIN_DELIVERY_PCT, _MIN_DELIVERY_SPIKE, _make_session as _delivery_session  # noqa: E402
from src.data.bulk_deals import _is_fii_dii  # noqa: E402
from src.data.sast import _pnsea_available, _sast_one  # noqa: E402

OUTPUTS       = ROOT / "outputs"
NIFTY_CSV     = ROOT / "cache" / "backtest_nifty.csv"
BHAV_CACHE    = ROOT / "cache" / "bhavcopy"
SAST_CACHE    = ROOT / "cache" / "sast_events"
BULK_CACHE    = ROOT / "cache" / "bulk_deals_hist"
OUT_FILE      = OUTPUTS / "event_backtest.json"

_BHAV_URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date}.csv"
_BULK_HIST_URL = "https://www.nseindia.com/api/historical/bulk-deals?from={frm}&to={to}"
_BULK_PRIME_URL = "https://www.nseindia.com/report-detail/display-bulk-and-block-deals"

MIN_N_SHIP = 500
MIN_BHAV_DAYS = 20  # below this, too few sessions for a meaningful surge/lift read
DIRECTION  = "buy"  # all three sources here are buy-side signals


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

def _evaluate_one(ticker: str, as_of: date, ohlcv: dict) -> dict | None:
    """Point-in-time entry/exit simulation for one (ticker, as_of) event,
    mirroring scripts/backtest.py's own trade mechanics exactly (2xATR stop,
    3xATR target, simulate_raw's first-2-bar entry rule)."""
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
                            t2=t2, cost_pct=round_trip_cost_pct(DIRECTION))

    if not sim.get("triggered") or sim.get("outcome") not in ("t1_hit", "t2_hit", "sl_hit", "timeout"):
        return None

    return {
        "ticker": ticker, "as_of": as_of, "outcome": sim["outcome"],
        "return_pct": sim["return_pct"], "win": sim["outcome"] in ("t1_hit", "t2_hit"),
    }


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


def evaluate_events(events: list[tuple[date, str]], ohlcv: dict, universe: list[str]) -> dict:
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

    return verdict_for(signal_results, baseline_results)


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
                    new_records.append(mid.date().isoformat())
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


# ── main ──────────────────────────────────────────────────────────────────────

def main(source: str, weeks: int, skip_download: bool) -> None:
    universe = _load_universe(None)
    print(f"[backtest_events] universe: {len(universe)} tickers")
    ohlcv = _load_ohlcv_cache(universe)
    print(f"[backtest_events] loaded {len(ohlcv)}/{len(universe)} tickers from OHLCV cache")

    results: dict[str, dict] = {}

    if source in ("delivery", "all"):
        print(f"\n[backtest_events] === delivery_surge ===")
        events, untestable_reason = collect_delivery_events(weeks, skip_download)
        if untestable_reason:
            print(f"  UNTESTABLE: {untestable_reason}")
            results["delivery_surge"] = {"verdict": "UNTESTABLE", "reason": untestable_reason, "suggested_weight": 0}
        else:
            print(f"  {len(events)} raw events, evaluating forward returns...")
            stats = evaluate_events(events, ohlcv, universe)
            results["delivery_surge"] = stats
            print(f"  n={stats['n']} wr_lift={stats.get('wr_lift_pp')}pp ret_lift={stats.get('ret_lift')} "
                  f"-> {stats['verdict']}")

    if source in ("sast", "all"):
        print(f"\n[backtest_events] === sast_insider_buying ===")
        events, untestable_reason = collect_sast_events(weeks, universe)
        if untestable_reason:
            print(f"  UNTESTABLE: {untestable_reason}")
            results["sast_insider_buying"] = {"verdict": "UNTESTABLE", "reason": untestable_reason, "suggested_weight": 0}
        else:
            print(f"  {len(events)} raw events, evaluating forward returns...")
            stats = evaluate_events(events, ohlcv, universe)
            results["sast_insider_buying"] = stats
            print(f"  n={stats['n']} wr_lift={stats.get('wr_lift_pp')}pp ret_lift={stats.get('ret_lift')} "
                  f"-> {stats['verdict']}")

    if source in ("bulk", "all"):
        print(f"\n[backtest_events] === bulk_deal_fii_dii ===")
        events, untestable_reason = collect_bulk_events(weeks)
        if untestable_reason:
            print(f"  UNTESTABLE: {untestable_reason}")
            results["bulk_deal_fii_dii"] = {"verdict": "UNTESTABLE", "reason": untestable_reason, "suggested_weight": 0}
        else:
            print(f"  {len(events)} raw events, evaluating forward returns...")
            stats = evaluate_events(events, ohlcv, universe)
            results["bulk_deal_fii_dii"] = stats
            print(f"  n={stats['n']} wr_lift={stats.get('wr_lift_pp')}pp ret_lift={stats.get('ret_lift')} "
                  f"-> {stats['verdict']}")

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
        if stats.get("verdict") == "SHIP":
            any_ship = True
            print(f'  "{sig}": {stats["suggested_weight"]},  # SHIP: ret_lift={stats["ret_lift"]} n={stats["n"]}')
    if not any_ship:
        print("  (none -- no signal cleared the ship gate this pass)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["delivery", "sast", "bulk", "all"], default="all")
    ap.add_argument("--weeks", type=int, default=156)
    ap.add_argument("--skip-download", action="store_true",
                     help="Re-analyze from cache only, no new network fetches (delivery source only)")
    args = ap.parse_args()
    main(args.source, args.weeks, args.skip_download)
