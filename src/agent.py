"""Pure-Python synthesis layer — no API needed, zero cost."""

from __future__ import annotations

from datetime import date

MAX_WATCHLIST = 10

# target move % by timeframe
TARGET_MOVE = {"1-2d": 7, "5-7d": 28}

# stop-loss buffer by timeframe
STOP_PCT = {"1-2d": 0.04, "5-7d": 0.06}

# entry band half-width
ENTRY_BAND = 0.015  # ±1.5% around today_close

# signals that need [UNCONFIRMED] because we have no consensus data
UNCONFIRMED_SIGNALS = {"eps_surprise_15pct"}


def _risk(candidate: dict) -> str:
    score = candidate["score"]
    is_penny = candidate["ticker"].startswith("[PENNY]")
    has_disqualifiers = bool(candidate.get("disqualifiers"))
    is_small_cap = candidate.get("market_cap_cr") is not None and candidate["market_cap_cr"] < 500

    if has_disqualifiers or is_penny:
        return "HIGH"
    if score >= 7 and not is_small_cap:
        return "LOW"
    if is_small_cap:
        return "HIGH"
    return "MEDIUM"


def _fmt_price(p: float) -> str:
    if p >= 100:
        return f"₹{p:.0f}"
    return f"₹{p:.2f}"


def _entry_and_stop(candidate: dict) -> tuple[str, str]:
    close = candidate.get("today_close")
    if close is None or close <= 0:
        return "data unavailable", "data unavailable"

    tf = candidate.get("timeframe", "1-2d")
    low = close * (1 - ENTRY_BAND)
    high = close * (1 + ENTRY_BAND)
    stop = close * (1 - STOP_PCT.get(tf, 0.04))

    return f"{_fmt_price(low)}–{_fmt_price(high)}", f"below {_fmt_price(stop)}"


def _label_signals(signals: list[str]) -> list[str]:
    return [
        f"{s} [UNCONFIRMED]" if s in UNCONFIRMED_SIGNALS else s
        for s in signals
    ]


def synthesize_watchlist(
    candidates: list[dict],
    total_scanned: int,
    scan_date: date | None = None,
) -> dict:
    """
    Convert pre-scored candidates into the final watchlist dict.
    Deterministic — no external API calls.
    """
    if scan_date is None:
        scan_date = date.today()

    top = sorted(candidates, key=lambda c: c["score"], reverse=True)[:MAX_WATCHLIST]

    watchlist = []
    for c in top:
        tf = c.get("timeframe", "1-2d")
        entry_zone, invalidation = _entry_and_stop(c)

        watchlist.append({
            "ticker": c["ticker"],
            "score": c["score"],
            "target_move_pct": TARGET_MOVE.get(tf, 7),
            "timeframe": tf,
            "top_signals": _label_signals(c.get("active_signals", [])),
            "risk": _risk(c),
            "entry_zone": entry_zone,
            "invalidation": invalidation,
        })

    print(f"[agent] watchlist built: {len(watchlist)} entries (no API)")
    return {
        "watchlist": watchlist,
        "scan_date": scan_date.isoformat(),
        "total_candidates_scanned": total_scanned,
    }
