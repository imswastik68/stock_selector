# Event archive

Append-only daily archive of event-driven signals that were previously
fetched live every day and discarded (`src/cache.py`'s same-day cache
deletes yesterday's file for the same key). Written by `src/archive.py`,
called from `main.py`'s EOD scan. Nothing else in this repo consumed this
data before — it's what makes `SHORT_TERM_WEIGHTS`/`SWING_WEIGHTS`'s event
signals (`bulk_deal_fii_dii`, `promoter_buying`, `sast_insider_buying`,
`delivery_surge`, announcements, etc.) hand-weighted with no OOS validation
(see `scripts/backtest.py`'s own docstring: "Event signals... have no free
archive and keep hand weights").

## Format

One file per calendar day: `YYYY-MM-DD.jsonl`. One JSON object per line:

```json
{"date": "2026-07-05", "source": "bulk_deals", "ticker": "RELIANCE.NS", "payload": {...}, "hash": "..."}
```

`source` is one of: `bulk_deals`, `breakouts`, `breakdowns`, `volume_gainers`,
`announcements`, `results_calendar`, `delivery_signals`, `promoter_signals`,
`sast_signals`. `payload` is the raw record for that ticker from that day's
scan (unfiltered — includes "nothing happened" entries like
`delivery_surge: false`, not just positive hits; forward-return analysis
needs the full population to compute lift vs baseline, same reasoning as
`scripts/backtest.py`/`scripts/mine_big_movers.py`). `hash` is a content
hash used for append-only dedup — safe to re-run a scan (e.g. a CI retry)
without creating duplicate lines.

## Durability

This data is committed to the repo (not gitignored) and pushed back by CI
(`.github/workflows/daily_scan.yml`) after each EOD run — without that
commit step, the archive would evaporate with each GitHub Actions runner.

## Planned analysis (not yet built)

Once >=3 months accumulate (enough for a first look at 20/60-day forward
returns), `scripts/analyse_events.py` will compute forward-return lift per
event type and event combination, reusing the excursion patterns in
`scripts/mine_big_movers.py` (point-in-time price lookups, turnover
filtering) against the OHLCV already cached in `cache/backtest_ohlcv/`.
Until then this directory is purely accumulating — do not expect it to be
analysis-ready before that data volume exists.
