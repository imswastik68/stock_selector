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


# ── trade simulator ───────────────────────────────────────────────────────────

def simulate(pick: dict, df: pd.DataFrame) -> dict:
    """
    Simulate a single pick against actual OHLCV.

    Entry rule:
      Entry triggers if within the first 2 bars after rec_date,
      the bar's low <= entry_hi (price dipped into or below the zone).
      Entry price = entry_mid if within range, else entry_hi (bought at top of zone).

    Exit rule (scanned bar by bar after entry):
      BUY:  low <= sl → SL hit;  high >= t1 → T1 hit
      SELL: high >= sl → SL hit; low  <= t1 → T1 hit
      If both hit same bar → SL (conservative).
    """
    rec_date   = pd.Timestamp(pick["date"])
    entry_lo   = pick.get("entry_lo") or pick["entry_mid"]
    entry_hi   = pick.get("entry_hi") or pick["entry_mid"]
    entry_mid  = pick["entry_mid"]
    sl         = pick["sl"]
    t1         = pick["t1"]
    direction  = pick["direction"]  # "buy" or "sell"

    df = df[df.index >= rec_date].copy()
    if df.empty:
        return {"triggered": False, "outcome": "no_data"}

    today_close = float(df["Close"].iloc[-1])

    # ── 1. Entry check (first 2 bars) ────────────────────────────────────────
    entry_window = df.iloc[:2]
    triggered    = False
    entry_price  = None
    entry_idx    = None

    for idx, row in entry_window.iterrows():
        bar_low  = float(row["Low"])
        bar_high = float(row["High"])
        bar_open = float(row["Open"])

        if direction == "buy":
            # Entry if price ever dipped into or through the entry zone
            if bar_low <= entry_hi:
                triggered   = True
                entry_price = min(entry_mid, entry_hi) if bar_open <= entry_hi else entry_hi
                entry_idx   = idx
                break
        else:
            # Sell/short: entry if price rose into entry zone
            if bar_high >= entry_lo:
                triggered   = True
                entry_price = max(entry_mid, entry_lo) if bar_open >= entry_lo else entry_lo
                entry_idx   = idx
                break

    if not triggered:
        gap = (today_close - entry_mid) / entry_mid * 100
        return {
            "triggered":    False,
            "outcome":      "not_triggered",
            "entry_mid":    entry_mid,
            "current_price": today_close,
            "gap_pct":      round(gap, 2),
        }

    # ── 2. Scan forward for T1 / SL ──────────────────────────────────────────
    post_entry = df[df.index >= entry_idx]
    outcome    = "open"
    exit_price = today_close
    exit_idx   = df.index[-1]

    for idx, row in post_entry.iterrows():
        bar_low  = float(row["Low"])
        bar_high = float(row["High"])
        bar_open = float(row["Open"])

        if direction == "buy":
            sl_hit = bar_low  <= sl
            t1_hit = bar_high >= t1
            # Gap open past SL
            if bar_open <= sl:
                outcome    = "sl_hit"
                exit_price = bar_open   # exit at open, not SL (gap-down)
                exit_idx   = idx
                break
            if sl_hit and t1_hit:        # both on same bar → SL (conservative)
                outcome    = "sl_hit"
                exit_price = sl
                exit_idx   = idx
                break
            if sl_hit:
                outcome    = "sl_hit"
                exit_price = sl
                exit_idx   = idx
                break
            if t1_hit:
                outcome    = "t1_hit"
                exit_price = t1
                exit_idx   = idx
                break
        else:
            sl_hit = bar_high >= sl
            t1_hit = bar_low  <= t1
            if bar_open >= sl:
                outcome    = "sl_hit"
                exit_price = bar_open
                exit_idx   = idx
                break
            if sl_hit and t1_hit:
                outcome    = "sl_hit"
                exit_price = sl
                exit_idx   = idx
                break
            if sl_hit:
                outcome    = "sl_hit"
                exit_price = sl
                exit_idx   = idx
                break
            if t1_hit:
                outcome    = "t1_hit"
                exit_price = t1
                exit_idx   = idx
                break

    days_held = (exit_idx - entry_idx).days

    # ── 3. Return calculations ────────────────────────────────────────────────
    if direction == "buy":
        return_pct    = (exit_price - entry_price) / entry_price * 100
        t1_return_pct = (t1         - entry_price) / entry_price * 100
        sl_risk_pct   = (entry_price - sl)         / entry_price * 100  # positive = risk
    else:
        return_pct    = (entry_price - exit_price) / entry_price * 100
        t1_return_pct = (entry_price - t1)         / entry_price * 100
        sl_risk_pct   = (sl - entry_price)         / entry_price * 100

    # Max adverse / favorable excursion from entry bar onwards
    lows  = post_entry["Low"].astype(float).values
    highs = post_entry["High"].astype(float).values
    if direction == "buy":
        mae = (min(lows)  - entry_price) / entry_price * 100
        mfe = (max(highs) - entry_price) / entry_price * 100
    else:
        mae = (entry_price - max(highs)) / entry_price * 100
        mfe = (entry_price - min(lows))  / entry_price * 100

    # 1-week return (5 trading bars from entry)
    week_return = None
    if len(post_entry) >= 5:
        week_close = float(post_entry.iloc[4]["Close"])
        if direction == "buy":
            week_return = round((week_close - entry_price) / entry_price * 100, 2)
        else:
            week_return = round((entry_price - week_close) / entry_price * 100, 2)

    return {
        "triggered":     True,
        "outcome":       outcome,
        "entry_price":   round(entry_price, 2),
        "exit_price":    round(exit_price, 2),
        "current_price": round(today_close, 2),
        "return_pct":    round(return_pct, 2),
        "t1_return_pct": round(t1_return_pct, 2),
        "sl_risk_pct":   round(sl_risk_pct, 2),
        "week_return":   week_return,
        "days_held":     days_held,
        "mae_pct":       round(mae, 2),
        "mfe_pct":       round(mfe, 2),
    }


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
