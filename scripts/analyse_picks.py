"""
Analyse pick performance from outputs/telegram_picks.json.

For each pick (must have entry_mid + SL + T1):
  1. Fetch actual OHLCV from recommendation date → today
  2. Check if entry zone was touched within 2 days (entry triggered)
  3. Scan forward day-by-day: T1 hit = WIN, SL hit = LOSS, else OPEN
  4. Compute actual return, 1-week return, max adverse/favorable excursion

Outputs:
  - Console summary (win rate, expectancy, signal accuracy)
  - outputs/pick_performance.json (per-pick detail)

Run: python scripts/analyse_picks.py
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

try:
    import pandas as pd
    import yfinance as yf
except ImportError:
    sys.exit("Run: pip install yfinance pandas")

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent))
from src.trade_sim import simulate_trade as _simulate_trade

_HERE     = Path(__file__).parent
_ROOT     = _HERE.parent
_PICKS_IN = _ROOT / "outputs" / "telegram_picks.json"
_OUT      = _ROOT / "outputs" / "pick_performance.json"


# ── load ─────────────────────────────────────────────────────────────────────

def load_picks() -> list[dict]:
    if not _PICKS_IN.exists():
        sys.exit(f"Not found: {_PICKS_IN}\nRun fetch_telegram_history.py first.")
    picks = json.loads(_PICKS_IN.read_text())
    # Need entry zone + SL + T1 to simulate anything
    return [p for p in picks if p.get("entry_mid") and p.get("sl") and p.get("t1")]


# ── yfinance fetch ────────────────────────────────────────────────────────────

def fetch_ohlcv(tickers: list[str], start: str) -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            tickers, start=start, auto_adjust=True, progress=False, group_by="ticker"
        )
    if raw.empty:
        return {}

    result: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0).unique().tolist()
        for t in tickers:
            if t in level0:
                df = raw[t].dropna(how="all")
                df = df[df["Close"].notna()]
                if not df.empty:
                    result[t] = df
    else:
        # Single-ticker fallback
        if len(tickers) == 1:
            df = raw.dropna(how="all")
            df = df[df["Close"].notna()]
            if not df.empty:
                result[tickers[0]] = df
    return result


# ── trade simulator (delegated to src/trade_sim.py) ──────────────────────────

def simulate(pick: dict, df: pd.DataFrame) -> dict:
    return _simulate_trade(pick, df)


# ── reporting ─────────────────────────────────────────────────────────────────

_W = 58

def _bar(label: str, val: str):
    print(f"  {label:<28} {val}")


def print_summary(results: list[dict]):
    total       = len(results)
    triggered   = [r for r in results if r["sim"].get("triggered")]
    not_trig    = [r for r in results if not r["sim"].get("triggered")]
    wins        = [r for r in triggered if r["sim"]["outcome"] == "t1_hit"]
    losses      = [r for r in triggered if r["sim"]["outcome"] == "sl_hit"]
    opens       = [r for r in triggered if r["sim"]["outcome"] == "open"]

    print("\n" + "═"*_W)
    print(f"  PICK PERFORMANCE  ({total} picks with full data)")
    print("═"*_W)
    _bar("Total picks",          f"{total}")
    _bar("Entry triggered",      f"{len(triggered)}  ({len(triggered)/total*100:.0f}%)")
    _bar("Not triggered",        f"{len(not_trig)}")
    print()

    if triggered:
        closed   = wins + losses
        win_rate = len(wins) / len(closed) * 100 if closed else 0
        _bar("Closed trades",        f"{len(closed)}  →  Win rate: {win_rate:.0f}%")
        _bar("  T1 hits  (WIN)",     str(len(wins)))
        _bar("  SL hits  (LOSS)",    str(len(losses)))
        _bar("  Still open",         str(len(opens)))
        print()

        if wins:
            avg_w = sum(r["sim"]["return_pct"] for r in wins) / len(wins)
            avg_d = sum(r["sim"]["days_held"]  for r in wins) / len(wins)
            _bar("Avg WIN return",    f"+{avg_w:.2f}%  in {avg_d:.1f} days")
        if losses:
            avg_l = sum(r["sim"]["return_pct"]  for r in losses) / len(losses)
            avg_d = sum(r["sim"]["days_held"]   for r in losses) / len(losses)
            _bar("Avg LOSS return",   f"{avg_l:.2f}%  in {avg_d:.1f} days")
        if opens:
            avg_o = sum(r["sim"]["return_pct"] for r in opens) / len(opens)
            _bar("Avg OPEN return",   f"{avg_o:+.2f}%  (vs today's price)")
        print()

        if closed:
            expectancy = sum(r["sim"]["return_pct"] for r in closed) / len(closed)
            _bar("Expectancy (closed)",   f"{expectancy:+.2f}% per trade")

        week_vals = [r["sim"]["week_return"] for r in triggered if r["sim"].get("week_return") is not None]
        if week_vals:
            _bar("Avg 1-week return",  f"{sum(week_vals)/len(week_vals):+.2f}%")

    print()
    print("  TOP PICKS (by return):")
    for r in sorted(triggered, key=lambda x: x["sim"]["return_pct"], reverse=True)[:6]:
        s = r["sim"]
        flag = "✓" if s["outcome"] == "t1_hit" else ("✗" if s["outcome"] == "sl_hit" else "~")
        print(f"    {flag} {r['pick']['ticker']:<16} {r['pick']['date']}  "
              f"{s['return_pct']:>+6.1f}%  [{s['outcome']}]  {s['days_held']}d")

    print()
    print("  WORST PICKS:")
    for r in sorted(triggered, key=lambda x: x["sim"]["return_pct"])[:4]:
        s = r["sim"]
        flag = "✓" if s["outcome"] == "t1_hit" else ("✗" if s["outcome"] == "sl_hit" else "~")
        print(f"    {flag} {r['pick']['ticker']:<16} {r['pick']['date']}  "
              f"{s['return_pct']:>+6.1f}%  [{s['outcome']}]  {s['days_held']}d")

    # Per-week accuracy
    print()
    print("  ACCURACY BY WEEK:")
    week_buckets: dict[str, dict] = defaultdict(lambda: {"wins": 0, "losses": 0, "open": 0})
    for r in triggered:
        wk = pd.Timestamp(r["pick"]["date"]).strftime("W%W (%b %d)")
        if r["sim"]["outcome"] == "t1_hit":
            week_buckets[wk]["wins"] += 1
        elif r["sim"]["outcome"] == "sl_hit":
            week_buckets[wk]["losses"] += 1
        else:
            week_buckets[wk]["open"] += 1
    for wk in sorted(week_buckets):
        b   = week_buckets[wk]
        tot = b["wins"] + b["losses"] + b["open"]
        cl  = b["wins"] + b["losses"]
        wr  = f"{b['wins']}/{cl} ({b['wins']/cl*100:.0f}%)" if cl else "all open"
        print(f"    {wk:<14}  {tot} picks  {wr}")

    # Signal accuracy
    print()
    print("  SIGNAL WIN RATE (triggered, ≥2 occurrences):")
    sig_stats: dict[str, dict] = defaultdict(lambda: {"wins": 0, "total": 0})
    for r in triggered:
        for sig in r["pick"].get("signals", []):
            sig_stats[sig]["total"] += 1
            if r["sim"]["outcome"] == "t1_hit":
                sig_stats[sig]["wins"] += 1
    for sig, st in sorted(sig_stats.items(), key=lambda x: -x[1]["wins"] / max(x[1]["total"], 1)):
        if st["total"] >= 2:
            rate = st["wins"] / st["total"] * 100
            print(f"    {sig:<34}  {st['wins']}/{st['total']}  ({rate:.0f}%)")

    print("═"*_W)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    picks = load_picks()
    print(f"Loaded {len(picks)} picks with entry+SL+T1 data.")

    start  = min(p["date"] for p in picks)
    tickers = list({p["ticker"] for p in picks})
    print(f"Fetching OHLCV for {len(tickers)} tickers from {start} ...")

    ohlcv = fetch_ohlcv(tickers, start)
    print(f"Got OHLCV for {len(ohlcv)} / {len(tickers)} tickers.")

    results = []
    no_data = []
    for pick in picks:
        df = ohlcv.get(pick["ticker"])
        if df is None or df.empty:
            no_data.append(pick["ticker"])
            continue
        sim = simulate(pick, df)
        results.append({"pick": pick, "sim": sim})

    if no_data:
        print(f"No OHLCV data for: {', '.join(set(no_data))}")

    print_summary(results)

    # Save detailed per-pick results
    _OUT.write_text(json.dumps([
        {
            "ticker":      r["pick"]["ticker"],
            "date":        r["pick"]["date"],
            "direction":   r["pick"]["direction"],
            "score":       r["pick"].get("score"),
            "confidence":  r["pick"].get("confidence"),
            "risk":        r["pick"].get("risk"),
            "signals":     r["pick"].get("signals", []),
            "entry_lo":    r["pick"].get("entry_lo"),
            "entry_hi":    r["pick"].get("entry_hi"),
            "entry_mid":   r["pick"].get("entry_mid"),
            "sl":          r["pick"].get("sl"),
            "t1":          r["pick"].get("t1"),
            "t2":          r["pick"].get("t2"),
            **r["sim"],
        }
        for r in results
    ], indent=2, ensure_ascii=False))
    print(f"\nDetailed results → outputs/pick_performance.json")


if __name__ == "__main__":
    main()
