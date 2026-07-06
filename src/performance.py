"""
Signal performance tracker.

Records daily pick outcomes (T1/T2 hit / SL hit / still open) using the same
exit engine as live trading (src.trade_sim.simulate_raw + WINNER_POLICY), so
tracked win-rate always reflects the chosen exit policy.

Data stored in outputs/performance.json:
  {
    "YYYY-MM-DD": {
      "TICKER.NS": {
        "direction": "buy",
        "entry": 245.0,
        "stop_loss": 238.0,
        "target_1": 262.0,
        "target_2": 280.0,          # stored so policies using T2 work correctly
        "outcome": "t1_hit" | "t2_hit" | "sl_hit" | "timeout" | "open",
        "outcome_date": "YYYY-MM-DD"
      }, ...
    }, ...
  }

NOTE: This tracks signal hit-rate over *all* generated picks (advisory).
      src/portfolio.py tracks the capital-constrained live book (actual P&L).

Running hit-rate stats computed over last 30 days.
"""

from __future__ import annotations

import json
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.trade_sim import WINNER_POLICY, simulate_raw
from src.costs import round_trip_cost_pct

_PERF_FILE = Path(__file__).parent.parent / "outputs" / "performance.json"
_NIFTY_TICKER = "^NSEI"
_ALPHA_LOOKBACK_DAYS = 45  # entry + up to 20 trading bars forward + weekend/holiday buffer


def _parse_price(s) -> float | None:
    try:
        return float(str(s).replace("₹", "").replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _parse_zone(s) -> tuple[float, float] | None:
    """Parse compute_entry_levels' '₹95.5-₹104.5' format -> (95.5, 104.5).
    Returns None for missing/"N/A" zones (legacy picks, LLM fallback path)."""
    try:
        lo, hi = (float(x) for x in str(s).replace("₹", "").replace(",", "").split("-"))
        return (lo, hi) if 0 < lo <= hi else None
    except (ValueError, TypeError):
        return None


def _load_perf() -> dict:
    if _PERF_FILE.exists():
        try:
            return json.loads(_PERF_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_perf(data: dict) -> None:
    _PERF_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PERF_FILE.write_text(json.dumps(data, indent=2, default=str))


def record_picks(watchlist_data: dict, nifty_at_emission: float | None = None) -> None:
    """
    Record today's buy/sell picks into performance.json.
    Call this after save_output() each EOD run.

    nifty_at_emission: NIFTY close on the scan day (main.py threads
    market_wide_ctx["nifty_structure"]["current_price"]). Stored per-pick so
    evaluate_prior_picks can later compute abnormal return vs NIFTY over the
    same window (Live-Proof Round Phase 3) without re-fetching history that
    predates when tracking started.
    """
    scan_date = watchlist_data.get("scan_date", date.today().isoformat())
    perf = _load_perf()

    if scan_date in perf:
        return  # already recorded today

    picks = {}
    for direction, key in [("buy", "buy_watchlist"), ("sell", "sell_watchlist")]:
        for entry in watchlist_data.get(key, []):
            t  = entry.get("ticker", "")
            sl = _parse_price(entry.get("stop_loss"))
            t1 = _parse_price(entry.get("target_1"))
            t2 = _parse_price(entry.get("target_2"))
            price = entry.get("today_close")
            zone  = _parse_zone(entry.get("entry_zone") or entry.get("short_entry_zone"))
            if t and sl and t1:
                picks[t] = {
                    "direction":  direction,
                    "entry":      price,
                    "entry_lo":   zone[0] if zone else price,
                    "entry_hi":   zone[1] if zone else price,
                    "stop_loss":  sl,
                    "target_1":   t1,
                    "target_2":   t2,
                    "outcome":    "open",
                    "outcome_date": None,
                    # Stamped so evaluate_prior_picks re-simulates each pick under the
                    # policy that was live when it was recorded, not whatever WINNER_POLICY
                    # is today — otherwise a policy change silently rewrites history.
                    "exit_policy": WINNER_POLICY,
                    # Methodology stamp: picks recorded from this point on carry a real
                    # entry_lo/entry_hi zone and are evaluated starting the NEXT trading
                    # day (as_of = scan_date). Picks recorded before this existed have no
                    # zone (entry_lo==entry_hi==close) and were evaluated starting on the
                    # scan-day bar itself -- see evaluate_prior_picks' as_of comment for
                    # why that was a bug, not a design choice.
                    "eval_method": "next_day_zone_v2",
                    "big_mover":  entry.get("big_mover", False),
                    "instrument": entry.get("instrument"),
                    # Live-Proof Round (Phase 2) -- write-once proof inputs, never
                    # mutated after recording (only "outcome"/"outcome_date" and the
                    # Phase-3 alpha fields below are updated post-emission):
                    "active_signals":     entry.get("active_signals", []),
                    "regime":             watchlist_data.get("nifty_context"),
                    "nifty_at_emission":  nifty_at_emission,
                }

    if picks:
        perf[scan_date] = picks
        _save_perf(perf)
        print(f"[performance] recorded {len(picks)} picks for {scan_date}")


def evaluate_prior_picks(lookback_days: int = 7) -> dict:
    """
    For each open pick from the last `lookback_days` days, fetch OHLCV and
    update outcome via simulate_raw(exit_policy=WINNER_POLICY).
    Returns updated performance dict.
    """
    perf = _load_perf()
    today = date.today()
    cutoff = (today - timedelta(days=lookback_days)).isoformat()

    open_by_ticker: dict[str, list[tuple[str, str, dict]]] = {}
    for scan_date, picks in perf.items():
        if scan_date < cutoff:
            continue
        for ticker, pick in picks.items():
            if pick.get("outcome") == "open":
                open_by_ticker.setdefault(ticker, []).append((scan_date, ticker, pick))

    if not open_by_ticker:
        return perf

    tickers = list(open_by_ticker.keys())
    print(f"[performance] evaluating {len(tickers)} open picks (per-pick exit_policy)...")

    # 30d window: long enough for time_10d policy + buffer
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                tickers, period="30d", interval="1d",
                auto_adjust=True, progress=False, group_by="ticker",
            )
    except Exception as exc:
        print(f"[performance] OHLC fetch error: {exc}")
        return perf

    updated = 0
    for ticker, entries in open_by_ticker.items():
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker not in raw.columns.get_level_values(0):
                    continue
                df = raw[ticker].dropna(how="all")
            elif len(tickers) == 1:
                df = raw.dropna(how="all")
            else:
                continue

            if df.empty or len(df) < 2:
                continue

            for scan_date, t, pick in entries:
                # as_of = scan_date itself, NOT scan_date - 1 business day. simulate_raw
                # checks entry in the first 2 bars strictly AFTER as_of (future = df[df.index
                # > as_of_date], trade_sim.py:67) -- so as_of=scan_date means the first bar
                # checked is the NEXT trading day, matching reality: the EOD scan runs after
                # close, so the earliest a real trader can fill is the next session. The old
                # "as_of = scan_date - 1 business day" checked the scan-day bar itself, and
                # combined with entry_lo==entry_hi==close (no zone) below, that collapsed the
                # entry to a single point on a bar that had ALREADY happened by scan time --
                # its own intraday low could trip the 2xATR stop same-day before any real
                # fill was possible. Verified false positives: JAGRAN 2026-06-04, MBAPL
                # 2026-07-02 and 07-03, all sl_hit on their own scan date.
                as_of = pd.Timestamp(scan_date)

                entry_price = float(pick["entry"] or 0)
                # entry_lo/entry_hi: real entry zone (compute_entry_levels, close +/- 0.5*ATR)
                # for picks recorded after the eval_method stamp; point-at-close for legacy
                # picks (the as_of fix alone already removes their same-day self-trip).
                entry_lo  = float(pick.get("entry_lo") or entry_price)
                entry_hi  = float(pick.get("entry_hi") or entry_price)
                entry_mid = (entry_lo + entry_hi) / 2 if pick.get("entry_lo") else entry_price
                sl  = float(pick["stop_loss"])
                t1  = float(pick["target_1"])
                t2  = float(pick["target_2"]) if pick.get("target_2") else None
                direction = pick["direction"]

                if entry_price <= 0:
                    continue

                # Picks recorded before this stamp existed were all recorded under
                # "static" (the only policy that ever ran before Phase 5) — fall back
                # to that, not the current global WINNER_POLICY.
                pick_policy = pick.get("exit_policy", "static")

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sim = simulate_raw(
                        entry_lo, entry_hi, entry_mid,
                        sl, t1, direction, as_of, df,
                        t2=t2, exit_policy=pick_policy,
                    )

                if not sim.get("triggered"):
                    continue  # keep "open" — entry didn't fill

                pick["outcome"]      = sim["outcome"]
                pick["outcome_date"] = sim.get("exit_date")
                if sim["outcome"] != "open":
                    updated += 1
        except Exception:
            continue

    if updated:
        _save_perf(perf)
        print(f"[performance] updated {updated} pick outcomes")

    return perf


def _index_fwd_return(index_df: pd.DataFrame, entry_date: str | None, n: int) -> float | None:
    """Benchmark's raw price return from entry_date's close to the close n-1
    bars later -- the SAME 'n bars, counting entry day as bar 1' window
    convention src.trade_sim.simulate_raw uses for its own fwd_Nd fields, so
    the stock's forward window and the benchmark's forward window line up
    exactly. (One asymmetry, unavoidable and documented here: the stock side
    measures from its actual FILL PRICE -- possibly an intraday gap-open --
    while the benchmark has no equivalent "fill," so it measures from its
    closing price on the entry day. This is the standard way to construct a
    passive-benchmark comparison window; it is not a bug.)"""
    if not entry_date:
        return None
    future = index_df[index_df.index >= pd.Timestamp(entry_date)]
    if len(future) < n:
        return None
    entry_close = float(future["Close"].iloc[0])
    fwd_close = float(future["Close"].iloc[n - 1])
    if entry_close <= 0:
        return None
    return round((fwd_close - entry_close) / entry_close * 100, 2)


def evaluate_live_alpha(lookback_days: int = _ALPHA_LOOKBACK_DAYS) -> dict:
    """
    Compute fixed-horizon forward returns + NIFTY-relative alpha for every
    pick whose entry has filled, once enough forward data exists (Live-Proof
    Round Phase 3) -- the inputs src.gates.live_alpha_gate reconciles into a
    proven/insufficient/no-edge verdict per signal.

    Deliberately SEPARATE from evaluate_prior_picks (which resolves SL/T1/
    timeout outcomes over a 7-day lookback): alpha needs a much LONGER
    lookback (an entry needs up to 20 TRADING days of forward data to exist
    at all) and must also revisit picks whose outcome is already DECIDED --
    fwd_Nd measures the raw signal over a fixed horizon, deliberately
    independent of whether/when the SL or T1 exit rule fired first.

    Write-once per pick (same discipline as record_picks): a pick that
    already has fwd_10d set is never recomputed, even if this runs again.

    Adds, once entry fills and >=10 forward trading bars exist:
      realized_return_pct  -- cost-netted P&L, but ONLY once the trade is
                              DECIDED (None while still open -- "realized"
                              means closed, not a mark-to-market snapshot)
      fwd_5d / fwd_10d / fwd_20d  -- fixed-horizon forward return from the
                              actual entry fill, cost-netted, ignoring the
                              stop entirely (isolates the raw signal from the
                              exit rule -- a pick that got stopped out at -2%
                              might still have fwd_10d > 0)
      nifty_fwd_10d         -- NIFTY's return over the IDENTICAL window
      abnormal_10d          -- fwd_10d - nifty_fwd_10d: the live alpha, net
                              of cost. THE primary proof metric.
      abnormal_10d_stress   -- reversal_oversold_v2 picks ONLY: fwd_10d
                              recomputed with a 2x cost haircut instead of
                              1x, minus nifty_fwd_10d. reversal's falling-
                              knife entries have the worst real slippage of
                              any shipped signal (see src/scorer.py's
                              reversal_oversold_v2 comment) -- this is an
                              explicit, honest stress test on that one
                              signal, reported alongside rather than hidden
                              behind the same cost model as everything else.
    """
    perf = _load_perf()
    today = date.today()
    cutoff = (today - timedelta(days=lookback_days)).isoformat()

    pending: list[tuple[str, str, dict]] = []  # (scan_date, ticker, pick)
    for scan_date, picks in perf.items():
        if scan_date < cutoff:
            continue
        for ticker, pick in picks.items():
            if pick.get("fwd_10d") is not None:
                continue  # already computed -- write-once
            if not pick.get("entry_lo") and not pick.get("entry"):
                continue  # malformed/legacy pick with no price reference
            pending.append((scan_date, ticker, pick))

    if not pending:
        return perf

    tickers = sorted({t for _, t, _ in pending})
    print(f"[performance] computing live alpha for {len(tickers)} ticker(s)...")

    period_str = f"{lookback_days + 30}d"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                tickers, period=period_str, interval="1d",
                auto_adjust=True, progress=False, group_by="ticker",
            )
            nifty_df = yf.download(
                _NIFTY_TICKER, period=period_str, interval="1d",
                auto_adjust=True, progress=False,
            )
    except Exception as exc:
        print(f"[performance] live-alpha OHLC fetch error: {exc}")
        return perf

    if nifty_df is None or nifty_df.empty:
        print("[performance] no NIFTY data -- skipping live-alpha this run")
        return perf
    if isinstance(nifty_df.columns, pd.MultiIndex):
        nifty_df.columns = nifty_df.columns.get_level_values(0)

    updated = 0
    for scan_date, ticker, pick in pending:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker not in raw.columns.get_level_values(0):
                    continue
                df = raw[ticker].dropna(how="all")
            elif len(tickers) == 1:
                df = raw.dropna(how="all")
            else:
                continue
            if df.empty:
                continue

            as_of = pd.Timestamp(scan_date)
            entry_price = float(pick["entry"] or 0)
            entry_lo  = float(pick.get("entry_lo") or entry_price)
            entry_hi  = float(pick.get("entry_hi") or entry_price)
            entry_mid = (entry_lo + entry_hi) / 2 if pick.get("entry_lo") else entry_price
            sl  = float(pick["stop_loss"])
            t1  = float(pick["target_1"])
            t2  = float(pick["target_2"]) if pick.get("target_2") else None
            direction = pick["direction"]
            if entry_price <= 0:
                continue

            pick_policy = pick.get("exit_policy", "static")
            cost_pct = round_trip_cost_pct(direction)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sim = simulate_raw(
                    entry_lo, entry_hi, entry_mid, sl, t1, direction, as_of, df,
                    t2=t2, exit_policy=pick_policy, cost_pct=cost_pct,
                )

            if not sim.get("triggered"):
                continue  # entry never filled -- nothing to measure yet

            fwd10_gross = sim.get("fwd_10d_pct")
            if fwd10_gross is None:
                continue  # not enough forward bars yet -- retry on a future run

            nifty_fwd_10d = _index_fwd_return(nifty_df, sim.get("entry_date"), 10)

            fwd5_gross  = sim.get("fwd_5d_pct")
            fwd20_gross = sim.get("fwd_20d_pct")

            pick["fwd_5d"]  = round(fwd5_gross - cost_pct, 2) if fwd5_gross is not None else None
            pick["fwd_10d"] = round(fwd10_gross - cost_pct, 2)
            pick["fwd_20d"] = round(fwd20_gross - cost_pct, 2) if fwd20_gross is not None else None
            pick["nifty_fwd_10d"] = nifty_fwd_10d
            pick["abnormal_10d"] = (
                round(pick["fwd_10d"] - nifty_fwd_10d, 2) if nifty_fwd_10d is not None else None
            )

            if "reversal_oversold_v2" in (pick.get("active_signals") or []):
                stress_fwd_10d = round(fwd10_gross - 2 * cost_pct, 2)
                pick["abnormal_10d_stress"] = (
                    round(stress_fwd_10d - nifty_fwd_10d, 2) if nifty_fwd_10d is not None else None
                )
            else:
                pick["abnormal_10d_stress"] = None

            pick["realized_return_pct"] = sim["return_pct"] if sim["outcome"] != "open" else None

            updated += 1
        except Exception:
            continue

    if updated:
        _save_perf(perf)
        print(f"[performance] computed live alpha for {updated} pick(s)")

    return perf


def performance_summary(lookback_days: int = 30) -> dict:
    """
    Compute hit-rate stats over last `lookback_days` days.
    t2_hit and t1_hit both count as wins; timeout is neutral.
    Returns {total, t1_hit, sl_hit, open, win_rate_pct}.
    """
    perf = evaluate_prior_picks(lookback_days)
    today = date.today()
    cutoff = (today - timedelta(days=lookback_days)).isoformat()

    total = t1 = sl = open_count = 0
    # Evidence stream for the two pending near-miss decisions (BIG_MOVER gate n=199/200;
    # short pipeline re-enable) — split win rate by big_mover flag, report-only.
    bm_t1 = bm_sl = other_t1 = other_sl = 0
    # v2-only tally: picks recorded with the fixed next-day-zone entry methodology
    # (eval_method == "next_day_zone_v2"). Mixed-methodology windows understate the
    # true win rate with legacy scan-day-point-entry picks in them -- this lets a
    # caller (e.g. src/gates.py) restrict to clean-method data once enough accumulates.
    v2_t1 = v2_sl = 0
    for scan_date, picks in perf.items():
        if scan_date < cutoff:
            continue
        for pick in picks.values():
            total += 1
            outcome = pick.get("outcome", "open")
            is_win  = outcome in ("t1_hit", "t2_hit")
            is_loss = outcome == "sl_hit"
            if is_win:
                t1 += 1
            elif is_loss:
                sl += 1
            else:
                open_count += 1

            if pick.get("big_mover"):
                bm_t1 += is_win
                bm_sl += is_loss
            else:
                other_t1 += is_win
                other_sl += is_loss

            if pick.get("eval_method") == "next_day_zone_v2":
                v2_t1 += is_win
                v2_sl += is_loss

    decided = t1 + sl
    win_rate = round(t1 / decided * 100, 1) if decided > 0 else None

    bm_decided = bm_t1 + bm_sl
    other_decided = other_t1 + other_sl
    v2_decided = v2_t1 + v2_sl

    return {
        "total_picks":    total,
        "t1_hit":         t1,
        "sl_hit":         sl,
        "open":           open_count,
        "win_rate_pct":   win_rate,
        "lookback_days":  lookback_days,
        "exit_policy":    WINNER_POLICY,
        "big_mover_win_rate_pct":   round(bm_t1 / bm_decided * 100, 1) if bm_decided > 0 else None,
        "big_mover_n_decided":      bm_decided,
        "other_win_rate_pct":       round(other_t1 / other_decided * 100, 1) if other_decided > 0 else None,
        "other_n_decided":          other_decided,
        "n_decided_v2":             v2_decided,
        "win_rate_v2_pct":          round(v2_t1 / v2_decided * 100, 1) if v2_decided > 0 else None,
        "methodology_note": (
            "picks recorded before the next_day_zone_v2 stamp were evaluated with a "
            "scan-day point-entry (bug: same-day SL trips inflated losses); this window "
            "may mix both methods -- see n_decided_v2/win_rate_v2_pct for clean-method-only figures"
        ),
    }


LOSS_STREAK_THRESHOLD = 5
LOSS_STREAK_COOLDOWN_DAYS = 7


def loss_streak_state(lookback_days: int = 60) -> dict:
    """
    5 consecutive sl_hit outcomes (across all tickers, most recent decided
    picks first) -> 7-day watch-only cooldown on new daily-track entries.
    Momentum book (scripts/factor_scan.py) is exempt -- its own discipline is
    the monthly rebalance + 200DMA gate, not this daily-pick loss streak.

    Returns {"in_cooldown": bool, "streak": int, "cooldown_until": "YYYY-MM-DD" | None}.
    """
    perf = _load_perf()
    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

    decided: list[tuple[str, str]] = []  # (outcome_date, outcome)
    for scan_date, picks in perf.items():
        if scan_date < cutoff:
            continue
        for pick in picks.values():
            outcome = pick.get("outcome", "open")
            outcome_date = pick.get("outcome_date")
            if outcome != "open" and outcome_date:
                decided.append((outcome_date, outcome))

    if not decided:
        return {"in_cooldown": False, "streak": 0, "cooldown_until": None}

    decided.sort(key=lambda x: x[0])  # chronological

    streak = 0
    for _, outcome in reversed(decided):
        if outcome == "sl_hit":
            streak += 1
        else:
            break

    if streak < LOSS_STREAK_THRESHOLD:
        return {"in_cooldown": False, "streak": streak, "cooldown_until": None}

    last_loss_date = decided[-1][0]
    cooldown_until = (date.fromisoformat(last_loss_date) + timedelta(days=LOSS_STREAK_COOLDOWN_DAYS)).isoformat()
    in_cooldown = date.today().isoformat() <= cooldown_until
    return {"in_cooldown": in_cooldown, "streak": streak, "cooldown_until": cooldown_until}
