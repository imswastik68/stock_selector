"""
Append-only proprietary event archive.

Every day's SAST/promoter/bulk-deal/delivery/announcement/breakout/breakdown
events are fetched live by the data layer, scored once, and then thrown away
-- src/cache.py's save_today() deletes yesterday's pickle for the same key by
design (it's a same-day reuse cache, not a history). No historical archive of
these event types exists anywhere in this codebase, which is exactly why
they were never backtestable (scripts/backtest.py's own docstring: "Event
signals... have no free archive and keep hand weights in the live scorer").
This module starts that archive. Its value is small on day 1 and compounds
with every scan -- a dataset nobody else has, built from data this system
was already fetching and discarding.

Storage: data/event_archive/YYYY-MM-DD.jsonl, one JSON object per line, one
line per (date, source, ticker) event. Append-only, deduped by a content
hash so re-running the same day's scan (retry, manual re-run) never creates
duplicate lines.

Usage (main.py, EOD mode only, fail-soft -- must never block the scan it
piggybacks on):
    from src.archive import archive_events
    archive_events(
        scan_date, bulk_deals=bulk_deals, breakouts=breakouts,
        breakdowns=breakdowns, volume_gainers=volume_gainers,
        announcements=announcements, results_calendar=results_calendar,
        delivery_signals=delivery_signals,
    )
    # ... later, once promoter_signals/sast_signals are computed for the top-20:
    archive_events(scan_date, promoter_signals=promoter_signals, sast_signals=sast_signals)

Later (not this pass, noted in data/event_archive/README.md): once >=3 months
accumulate, scripts/analyse_events.py computes forward 5/20/60d returns per
event type, reusing scripts/mine_big_movers.py's excursion patterns.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ARCHIVE_DIR = Path(__file__).parent.parent / "data" / "event_archive"


def _event_hash(scan_date: str, source: str, ticker: str, payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(f"{scan_date}|{source}|{ticker}|{blob}".encode()).hexdigest()[:16]


def _existing_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    hashes: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            hashes.add(json.loads(line)["hash"])
        except Exception:
            continue
    return hashes


def archive_events(scan_date: str, **sources) -> int:
    """
    Append today's records to data/event_archive/<scan_date>.jsonl.

    Each kwarg is either:
      - a list of dicts with a "ticker" key (bulk_deals, breakouts, breakdowns,
        volume_gainers, announcements, results_calendar), or
      - a dict keyed by ticker -> dict or bool (delivery_signals,
        promoter_signals, sast_signals).

    Every record is archived, including "nothing happened" entries (e.g.
    delivery_surge=False) -- later forward-return analysis needs the full
    population, not just positive hits, to compute lift vs baseline (same
    reasoning as scripts/backtest.py and scripts/mine_big_movers.py, which
    both score every candidate, not just the ones that fired).

    Fail-soft: any exception is caught and logged, never raised. Returns the
    number of NEW lines appended (0 on cache-hit-all-duplicate or on failure).
    """
    try:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        path = ARCHIVE_DIR / f"{scan_date}.jsonl"
        seen = _existing_hashes(path)
        new_lines = []

        for source, records in sources.items():
            if not records:
                continue
            if isinstance(records, dict):
                pairs = list(records.items())
            elif isinstance(records, list):
                pairs = [(r.get("ticker"), r) for r in records if isinstance(r, dict) and r.get("ticker")]
            else:
                continue

            for ticker, payload in pairs:
                if not ticker:
                    continue
                payload_dict = payload if isinstance(payload, dict) else {"value": payload}
                h = _event_hash(scan_date, source, ticker, payload_dict)
                if h in seen:
                    continue
                seen.add(h)
                new_lines.append(json.dumps({
                    "date": scan_date, "source": source, "ticker": ticker,
                    "payload": payload_dict, "hash": h,
                }, default=str))

        if new_lines:
            with path.open("a") as f:
                f.write("\n".join(new_lines) + "\n")
        print(f"[archive] {len(new_lines)} new events -> {path.name} ({len(seen)} total today)")
        return len(new_lines)
    except Exception as exc:
        print(f"[archive] failed (non-fatal): {exc}")
        return 0
