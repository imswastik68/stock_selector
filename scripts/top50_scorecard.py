"""
Score every Telegram alert against the top-50-mover success bar (adopted 2026-11-11).

An alert succeeds if, from a price the user could actually buy at, the stock
reaches the AVERAGE move of the top-50 tradable gainers/losers over the same
window (BSE bhavcopy, Nov-2025 -> Nov-2026):

    bullish  +5.71% in 1 day   +12.63% in 1 week (5 bars)   +27.54% in 1 month (21 bars)
    bearish  -4.44%            -9.65%                        -19.71%   (F&O names only)

"Touched" = the bar was reached intraday inside the window, i.e. a limit order
at the bar would have filled. Entry depends on when the alert was SENT:
    before 09:15 IST -> that day's open
    09:15-15:30      -> that day's close (daily data cannot see the alert-time price)
    after 15:30 / non-trading day -> next trading day's open

Tradable = BSE group A/B/X (excludes SME, T2T, Z), prior close >= Rs 20,
prior-20-day average turnover >= Rs 1 cr. Prices are corporate-action
adjusted by chaining ClsPric/PrvsClsgPric, so splits and bonuses don't
show up as fake moves.

The system PASSES a horizon only if, over >= 100 tradable alerts on >= 40
alert dates, all three hold:
    1. touched rate >= 30%
    2. touched rate >= 2x a random tradable stock entered on the SAME days
       (the base rate swings with the market; one hot month can fake 30%)
    3. average P&L per alert > 0 after 0.30% cost, exiting with a limit at
       the bar, else at the window's last close (hit rate alone ignores losers)

Same method as stock-news-alerts' scorecard, so the two bots are comparable.

Usage:
    python scripts/fetch_telegram_history.py   # refresh outputs/telegram_picks.json
    python scripts/top50_scorecard.py
Output: console table + outputs/top50_scorecard.json
"""

from __future__ import annotations

import json
import re
import sys
import time as _time
import warnings
from datetime import date, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
BHAV_DIR = ROOT / "cache" / "bse_bhav"
EQUITY_L = ROOT / "cache" / "nse_equity_l.csv"
OUTPUTS = ROOT / "outputs"

BAR = {"bullish": {1: 5.71, 5: 12.63, 21: 27.54},
       "bearish": {1: -4.44, 5: -9.65, 21: -19.71}}
HORIZON = {1: "1 day", 5: "1 week", 21: "1 month"}
COST_PCT = 0.30
PASS_HIT_RATE = 0.30
PASS_VS_RANDOM = 2.0
MIN_ALERTS, MIN_DATES = 100, 40

_BSE_URL = "https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_{d}_F_0000.CSV"
_NSE_EQUITY_L = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}
_COLS = ["TradDt", "ISIN", "TckrSymb", "SctySrs", "OpnPric", "HghPric", "LwPric",
         "ClsPric", "PrvsClsgPric", "TtlTrfVal"]


# ── data ─────────────────────────────────────────────────────────────────────

def ensure_bhav(start: date, end: date) -> None:
    """Download any missing BSE equity bhavcopy in [start, end] into cache/bse_bhav/.

    Non-trading days return an HTML page, not a CSV; those are remembered in
    a .holidays file so they aren't re-requested on every run.
    """
    BHAV_DIR.mkdir(parents=True, exist_ok=True)
    hol_file = BHAV_DIR / ".holidays"
    holidays = set(hol_file.read_text().split()) if hol_file.exists() else set()
    d, fetched = start, 0
    while d <= min(end, date.today()):
        key = d.strftime("%Y%m%d")
        if d.weekday() < 5 and key not in holidays and not (BHAV_DIR / f"{key}.csv").exists():
            try:
                r = requests.get(_BSE_URL.format(d=key), headers=_UA, timeout=30)
                if r.ok and r.text.startswith("TradDt"):
                    (BHAV_DIR / f"{key}.csv").write_text(r.text)
                    fetched += 1
                elif d < date.today():  # today's file may simply not be published yet
                    holidays.add(key)
            except requests.RequestException as exc:
                print(f"[top50] bhavcopy {key} failed: {exc}")
            _time.sleep(0.3)
        d += timedelta(days=1)
    hol_file.write_text("\n".join(sorted(holidays)))
    if fetched:
        print(f"[top50] downloaded {fetched} bhavcopy files")


def nse_symbol_to_isin() -> dict[str, str]:
    if not EQUITY_L.exists():
        r = requests.get(_NSE_EQUITY_L, headers=_UA, timeout=30)
        r.raise_for_status()
        EQUITY_L.write_text(r.text)
    df = pd.read_csv(EQUITY_L)
    df.columns = [c.strip() for c in df.columns]
    return dict(zip(df["SYMBOL"].str.strip(), df["ISIN NUMBER"].str.strip()))


def load_panel(start: date, end: date) -> dict:
    files = [f for f in sorted(BHAV_DIR.glob("*.csv"))
             if start.strftime("%Y%m%d") <= f.stem <= end.strftime("%Y%m%d")]
    b = pd.concat(pd.read_csv(f, usecols=_COLS) for f in files)
    b["TradDt"] = pd.to_datetime(b["TradDt"])
    b = b[(b.PrvsClsgPric > 0) & (b.ClsPric > 0)]
    # a handful of ISINs appear twice in one file; keep the traded line
    b = b.sort_values("TtlTrfVal").drop_duplicates(["ISIN", "TradDt"], keep="last")
    b = b.sort_values(["ISIN", "TradDt"]).reset_index(drop=True)
    b["r"] = b.ClsPric / b.PrvsClsgPric
    b["L"] = b.groupby("ISIN").r.cumprod()          # adjusted close level
    b["Lp"] = b.L / b.r                               # adjusted level at prior close
    for out, col in (("aO", "OpnPric"), ("aH", "HghPric"), ("aL", "LwPric")):
        b[out] = b.Lp * b[col] / b.PrvsClsgPric
    b["liq"] = b.groupby("ISIN").TtlTrfVal.transform(
        lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    b["ok"] = b.SctySrs.isin({"A", "B", "X"}) & (b.PrvsClsgPric >= 20) & (b.liq >= 1e7)

    cal = pd.DatetimeIndex(sorted(b.TradDt.unique()))
    piv = {c: b.pivot(index="TradDt", columns="ISIN", values=c).reindex(cal)
           for c in ("L", "aO", "aH", "aL", "ok")}
    last = b.drop_duplicates("ISIN", keep="last")
    return {
        "cal": cal,
        "L": piv["L"].ffill().values,
        "O": piv["aO"].values, "H": piv["aH"].values, "Lo": piv["aL"].values,
        "ok": piv["ok"].fillna(False).values.astype(bool),
        "isin_col": {k: j for j, k in enumerate(piv["L"].columns)},
        "bse_symbol_to_isin": dict(zip(last.TckrSymb, last.ISIN)),
    }


# ── scoring ──────────────────────────────────────────────────────────────────

def _window(p: dict, e: int, n: int, basis: str):
    """Entry row-vector and the n-bar holding window for entry day index e."""
    if basis == "open":
        return p["O"][e], slice(e, e + n)
    return p["L"][e], slice(e + 1, e + 1 + n)


def _score(entry, end_level, extreme, direction: str, n: int) -> dict:
    """Vectorised outcome for one or many stocks. pnl: limit exit at the bar, else window close."""
    bar = BAR[direction][n]
    end = (end_level / entry - 1) * 100
    best = (extreme / entry - 1) * 100
    bull = direction == "bullish"
    touch = best >= bar if bull else best <= bar
    at_end = end >= bar if bull else end <= bar
    pnl = np.where(touch, abs(bar), end if bull else -end) - COST_PCT
    return {"end": end, "touch": touch, "at_end": at_end, "pnl": pnl}


def entry_index(p: dict, sent: pd.Timestamp) -> tuple[int, str] | None:
    cal = p["cal"]
    day = sent.tz_localize(None).normalize()
    t = sent.time()
    trading = day in cal
    if trading and time(9, 15) <= t < time(15, 30):
        return cal.get_loc(day), "close"
    k = cal.searchsorted(day, side="left" if (trading and t < time(9, 15)) else "right")
    return (k, "open") if k < len(cal) else None


def random_baseline(p: dict, e: int, n: int, basis: str, direction: str, mask_cols) -> dict | None:
    entry, w = _window(p, e, n, basis)
    if w.stop > len(p["cal"]):
        return None
    m = p["ok"][e] & np.isfinite(entry) & np.isfinite(p["L"][w.stop - 1]) & mask_cols
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN window = no trade -> no touch
        ext = (np.nanmax(p["H"][w][:, m], axis=0) if direction == "bullish"
               else np.nanmin(p["Lo"][w][:, m], axis=0))
    s = _score(entry[m], p["L"][w.stop - 1, m], ext, direction, n)
    return {"touch": float(s["touch"].mean()), "at_end": float(s["at_end"].mean()),
            "pnl": float(np.nanmean(s["pnl"]))}


def score_alerts(picks: list[dict], sent_at: dict[int, pd.Timestamp], p: dict,
                 nse_isin: dict[str, str], fo_symbols: set[str]) -> tuple[pd.DataFrame, list[str]]:
    cols = p["isin_col"]
    fo_mask = np.zeros(len(cols), dtype=bool)
    for s in fo_symbols:
        j = cols.get(nse_isin.get(s) or p["bse_symbol_to_isin"].get(s))
        if j is not None:
            fo_mask[j] = True
    all_mask = np.ones(len(cols), dtype=bool)

    rows, unmapped, cache = [], [], {}
    for pk in picks:
        sym = re.sub(r"\.(NS|BO)$", "", pk["ticker"])
        j = cols.get(nse_isin.get(sym) or p["bse_symbol_to_isin"].get(sym))
        sent = sent_at.get(pk.get("msg_id"))
        if j is None or sent is None:
            unmapped.append(pk["ticker"])
            continue
        ei = entry_index(p, sent)
        if ei is None:
            continue
        e, basis = ei
        direction = "bullish" if pk["direction"] == "buy" else "bearish"
        tradable = bool(p["ok"][e, j]) and (direction == "bullish" or sym in fo_symbols)
        for n in (1, 5, 21):
            entry, w = _window(p, e, n, basis)
            if w.stop > len(p["cal"]) or not np.isfinite(entry[j]) or not np.isfinite(p["L"][w.stop - 1, j]):
                continue
            ext = np.nanmax(p["H"][w, j]) if direction == "bullish" else np.nanmin(p["Lo"][w, j])
            s = _score(entry[j], p["L"][w.stop - 1, j], ext, direction, n)
            key = (e, n, basis, direction)
            if key not in cache:
                cache[key] = random_baseline(p, e, n, basis, direction,
                                             fo_mask if direction == "bearish" else all_mask)
            b = cache[key]
            rows.append({
                "date": str(sent.date()), "ticker": pk["ticker"], "dir": direction, "n": n,
                "basis": basis, "tradable": tradable, "score": pk.get("score"),
                "end": float(s["end"]), "touch": bool(s["touch"]), "at_end": bool(s["at_end"]),
                "pnl": float(s["pnl"]),
                "rand_touch": b["touch"], "rand_at_end": b["at_end"], "rand_pnl": b["pnl"],
            })
    return pd.DataFrame(rows), unmapped


def verdict(x: pd.DataFrame) -> dict:
    n, dates = len(x), int(x.date.nunique())
    hit, rand = float(x.touch.mean()), float(x.rand_touch.mean())
    pnl = float(x.pnl.mean())
    fails = []
    if n < MIN_ALERTS or dates < MIN_DATES:
        status = "INSUFFICIENT"
        fails.append(f"need >= {MIN_ALERTS} alerts on >= {MIN_DATES} dates")
    else:
        if hit < PASS_HIT_RATE:
            fails.append(f"hit {hit:.1%} < {PASS_HIT_RATE:.0%}")
        if rand > 0 and hit / rand < PASS_VS_RANDOM:
            fails.append(f"{hit / rand:.1f}x random < {PASS_VS_RANDOM:.0f}x")
        if pnl <= 0:
            fails.append(f"P&L {pnl:+.2f}%/alert <= 0")
        status = "FAIL" if fails else "PASS"
    return {"n": n, "dates": dates, "touched": round(hit, 4), "random_touched": round(rand, 4),
            "at_end": round(float(x.at_end.mean()), 4), "random_at_end": round(float(x.rand_at_end.mean()), 4),
            "avg_change": round(float(x.end.mean()), 2), "pnl_per_alert": round(pnl, 2),
            "random_pnl": round(float(x.rand_pnl.mean()), 2), "status": status, "why": fails}


def main() -> None:
    picks = json.loads((OUTPUTS / "telegram_picks.json").read_text())
    history = json.loads((OUTPUTS / "telegram_history.json").read_text())
    sent_at = {m["id"]: pd.Timestamp(m["date"]).tz_convert("Asia/Kolkata") for m in history}

    first = min(pd.Timestamp(p["date"]) for p in picks).date()
    start = first - timedelta(days=45)                 # 20-day turnover look-back
    ensure_bhav(start, date.today())
    panel = load_panel(start, date.today())
    nse_isin = nse_symbol_to_isin()
    fo = set(pd.read_csv(ROOT / "data" / "fo_eligible.csv")["symbol"].astype(str))

    R, unmapped = score_alerts(picks, sent_at, panel, nse_isin, fo)
    one = R[R.n == 1]
    print(f"{len(picks)} alerts | scored {one.shape[0]} | unmapped {len(unmapped)} "
          f"| tradable {one.tradable.mean():.0%} | entry basis {one.basis.value_counts().to_dict()}")

    report = {"as_of": date.today().isoformat(), "bar": BAR, "horizons": {}}
    T = R[R.tradable]
    print(f"\nTRADABLE alerts vs the top-50 bar (touched = limit at the bar fills)")
    print(f"{'':8} {'dir':8} {'n':>4} {'dates':>5} {'bar':>8} | {'touched':>7} {'random':>6} | "
          f"{'at end':>6} {'random':>6} | {'P&L/alert':>9} {'random':>7} | verdict")
    for d in ("bullish", "bearish"):
        for n in (1, 5, 21):
            x = T[(T.dir == d) & (T.n == n)]
            if x.empty:
                continue
            v = verdict(x)
            report["horizons"][f"{d}_{n}d"] = v
            print(f"{HORIZON[n]:8} {d:8} {v['n']:4} {v['dates']:5} {BAR[d][n]:+7.2f}% | "
                  f"{v['touched']:7.1%} {v['random_touched']:6.1%} | {v['at_end']:6.1%} {v['random_at_end']:6.1%} | "
                  f"{v['pnl_per_alert']:+8.2f}% {v['random_pnl']:+6.2f}% | {v['status']}"
                  + (f" ({'; '.join(v['why'])})" if v["why"] else ""))
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "top50_scorecard.json").write_text(json.dumps(report, indent=2))
    R.to_csv(OUTPUTS / "top50_scorecard_rows.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
