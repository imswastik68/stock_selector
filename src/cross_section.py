"""
Cross-sectional rank IC — does `score` actually rank forward returns?

WHY THIS EXISTS
---------------
Until now the only stored record of a scan was the handful of names it emitted
(outputs/YYYY-MM-DD.json + performance.json). That makes the system's central
question unanswerable: a top-5 list tells you how those 5 did, but not whether
score ORDERED the field. Correlating score against forward return on the
emitted names alone is range-restricted -- they're all high scores by
construction -- which attenuates the correlation toward zero regardless of
whether real skill exists.

Measured on the emitted picks only, score vs abnormal_10d came out at
r = -0.009 (n=19, 2026-07). That number is suggestive but NOT conclusive,
precisely because of the truncation. This module removes the excuse: it stores
the entire scored cross-section each scan (~145 tickers, not ~8) and computes
Spearman rank IC once forward returns mature.

WHY RANK IC RATHER THAN WIN RATE
--------------------------------
Win rate cannot separate "picked well" from "market went up" -- in a rising
tape a coin flip looks skilled. Rank IC is the cross-sectional correlation
between predicted ordering and realized ordering, so a market-wide move
affects every name and cancels out. It is the standard measure for exactly
this question, and it is what src/gates.py's momentum ship gate already uses
for the factor book (see outputs/factor_backtest.json ic_holdout) -- this
brings the daily scanner under the same yardstick.

Reference points for interpreting the output: IC ~0.02-0.07 sustained is a
genuinely useful signal in a liquid cross-section; |IC| < 0.01 is noise. The
t-statistic matters as much as the level -- a high IC over 5 days is nothing.

STATUS: this is instrumentation, not a strategy change. It reports; it does not
gate trading. Nothing here alters which stocks are picked.
"""

from __future__ import annotations

import json
import math
import statistics
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

_CS_DIR = Path(__file__).parent.parent / "outputs" / "cross_section"
_IC_FILE = Path(__file__).parent.parent / "outputs" / "score_ic.json"

# Forward horizon for the return leg. 10 trading days matches the horizon
# performance.py already uses for abnormal_10d, so the two measures describe
# the same window and can be read side by side.
_HORIZON_DAYS = 10

# Below this many names a single day's cross-section is too thin for its rank
# correlation to mean anything; the day is skipped rather than averaged in.
_MIN_NAMES_PER_DAY = 20

# Aggregate reporting thresholds. n here counts DAYS (independent
# cross-sections), not names -- 30 daily ICs is the usual minimum before the
# mean is worth reading, and it is the same bar live_alpha_gate applies.
_MIN_DAYS = 30


def record_cross_section(scan_date: str, scored: list[dict]) -> None:
    """Persist one scan's full scored cross-section. Write-once per date."""
    if not scored:
        return
    _CS_DIR.mkdir(parents=True, exist_ok=True)
    path = _CS_DIR / f"{scan_date}.json"
    if path.exists():
        return  # never rewrite history

    rows = [
        {"ticker": r["ticker"], "score": r["score"], "close": r["close"],
         "qualified": r.get("qualified", False)}
        for r in scored
        if r.get("close") and r.get("ticker")
    ]
    if not rows:
        return
    path.write_text(json.dumps({"scan_date": scan_date, "n": len(rows), "rows": rows}, indent=2))
    print(f"[cross_section] recorded {len(rows)} scored names for {scan_date}")


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation, average-ranking ties."""
    n = len(xs)
    if n < 3:
        return None

    def _ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    if den == 0:
        return None  # no variance in one leg (e.g. every score identical)
    return num / den


def _matured_dates(today: date | None = None) -> list[Path]:
    """Cross-section files old enough for a _HORIZON_DAYS forward return to exist.
    Calendar-day padding (1.5x + 5) approximates trading days conservatively."""
    today = today or date.today()
    cutoff = today - timedelta(days=int(_HORIZON_DAYS * 1.5) + 5)
    out = []
    for p in sorted(_CS_DIR.glob("*.json")) if _CS_DIR.exists() else []:
        try:
            if date.fromisoformat(p.stem) <= cutoff:
                out.append(p)
        except ValueError:
            continue
    return out


def compute_score_ic(today: date | None = None) -> dict:
    """
    Daily rank IC of score vs realized _HORIZON_DAYS forward return, averaged
    across matured cross-sections. Writes outputs/score_ic.json and returns it.
    """
    import pandas as pd
    import yfinance as yf

    files = _matured_dates(today)
    if not files:
        result = {"per_day": [], "n_days": 0, "mean_ic": None, "t_stat": None,
                  "verdict": "INSUFFICIENT", "horizon_days": _HORIZON_DAYS,
                  "note": "no cross-section is old enough to have a forward return yet"}
        _IC_FILE.write_text(json.dumps(result, indent=2))
        return result

    # One batched download covering every ticker ever scored in these files.
    tickers: set[str] = set()
    payloads = []
    for p in files:
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        payloads.append(d)
        tickers.update(r["ticker"] for r in d.get("rows", []))

    if not tickers:
        result = {"per_day": [], "n_days": 0, "mean_ic": None, "t_stat": None,
                  "verdict": "INSUFFICIENT", "horizon_days": _HORIZON_DAYS}
        _IC_FILE.write_text(json.dumps(result, indent=2))
        return result

    oldest = min(d["scan_date"] for d in payloads)
    span_days = (date.today() - date.fromisoformat(oldest)).days + _HORIZON_DAYS * 2 + 10
    print(f"[cross_section] computing rank IC over {len(payloads)} day(s), "
          f"{len(tickers)} ticker(s)...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(sorted(tickers), period=f"{span_days}d", interval="1d",
                              auto_adjust=True, progress=False, group_by="ticker")
    except Exception as exc:
        print(f"[cross_section] OHLC fetch failed: {exc}")
        return {"per_day": [], "n_days": 0, "mean_ic": None, "t_stat": None,
                "verdict": "INSUFFICIENT", "horizon_days": _HORIZON_DAYS,
                "note": f"fetch error: {exc}"}

    per_day = []
    for d in payloads:
        scan_date = d["scan_date"]
        as_of = pd.Timestamp(scan_date)
        scores, fwds = [], []
        for r in d.get("rows", []):
            t = r["ticker"]
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if t not in raw.columns.get_level_values(0):
                        continue
                    df = raw[t].dropna(how="all")
                else:
                    df = raw.dropna(how="all")
                fut = df[df.index > as_of]
                if len(fut) < _HORIZON_DAYS:
                    continue
                entry = float(r["close"])
                exit_px = float(fut["Close"].iloc[_HORIZON_DAYS - 1])
                if entry <= 0:
                    continue
                scores.append(float(r["score"]))
                fwds.append((exit_px - entry) / entry * 100.0)
            except Exception:
                continue

        if len(scores) < _MIN_NAMES_PER_DAY:
            continue
        ic = _spearman(scores, fwds)
        if ic is None:
            continue
        per_day.append({"date": scan_date, "n_names": len(scores), "ic": round(ic, 4)})

    ics = [p["ic"] for p in per_day]
    mean_ic = statistics.mean(ics) if ics else None
    t_stat = None
    if len(ics) > 1:
        sd = statistics.stdev(ics)
        if sd > 0:
            t_stat = mean_ic / (sd / math.sqrt(len(ics)))

    if len(ics) < _MIN_DAYS:
        verdict = "INSUFFICIENT"
    elif mean_ic is not None and mean_ic >= 0.03 and t_stat is not None and t_stat >= 2.0:
        verdict = "RANKS"
    elif mean_ic is not None and mean_ic > 0 and t_stat is not None and t_stat >= 1.0:
        verdict = "WEAK"
    else:
        verdict = "NO_RANKING_ABILITY"

    result = {
        "per_day": per_day,
        "n_days": len(ics),
        "min_days": _MIN_DAYS,
        "horizon_days": _HORIZON_DAYS,
        "mean_ic": round(mean_ic, 4) if mean_ic is not None else None,
        "t_stat": round(t_stat, 2) if t_stat is not None else None,
        "verdict": verdict,
    }
    _IC_FILE.parent.mkdir(parents=True, exist_ok=True)
    _IC_FILE.write_text(json.dumps(result, indent=2))
    return result


def ic_report_line(result: dict | None = None) -> str:
    """One-line summary for the gates report / Telegram footer."""
    if result is None:
        try:
            result = json.loads(_IC_FILE.read_text())
        except Exception:
            return "SCORE IC: INSUFFICIENT (not yet measured)"

    n, need = result.get("n_days", 0), result.get("min_days", _MIN_DAYS)
    verdict = result.get("verdict", "INSUFFICIENT")
    if verdict == "INSUFFICIENT":
        return f"SCORE IC: INSUFFICIENT ({n}/{need} days)"
    return (f"SCORE IC: {verdict} (IC {result.get('mean_ic')}, "
            f"t={result.get('t_stat')}, {n} days)")
