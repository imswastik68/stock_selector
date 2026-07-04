"""
Regression tests for src/archive.py: the proprietary event archive must
dedup re-runs (same day's scan retried in CI must not duplicate lines),
must archive "nothing happened" entries (not just positive hits -- forward
lift analysis needs the full population), and must never raise (fail-soft,
so it can never block the scan it piggybacks on).
"""

from __future__ import annotations

import json
from unittest import mock

import src.archive as a


def test_archive_writes_list_and_dict_sources(tmp_path):
    archive_dir = tmp_path / "event_archive"
    with mock.patch.object(a, "ARCHIVE_DIR", archive_dir):
        n = a.archive_events(
            "2026-07-04",
            bulk_deals=[{"ticker": "RELIANCE.NS", "qty": 1000}],
            promoter_signals={"TCS.NS": {"buying": True}, "INFY.NS": False},
        )
    assert n == 3
    lines = [json.loads(l) for l in (archive_dir / "2026-07-04.jsonl").read_text().splitlines()]
    sources = {(l["source"], l["ticker"]) for l in lines}
    assert sources == {
        ("bulk_deals", "RELIANCE.NS"),
        ("promoter_signals", "TCS.NS"),
        ("promoter_signals", "INFY.NS"),
    }
    # "nothing happened" entry (INFY.NS: False) is archived, not filtered out
    infy = next(l for l in lines if l["ticker"] == "INFY.NS")
    assert infy["payload"] == {"value": False}


def test_archive_dedups_rerun_same_day(tmp_path):
    archive_dir = tmp_path / "event_archive"
    with mock.patch.object(a, "ARCHIVE_DIR", archive_dir):
        first = a.archive_events("2026-07-04", bulk_deals=[{"ticker": "RELIANCE.NS", "qty": 1000}])
        second = a.archive_events("2026-07-04", bulk_deals=[{"ticker": "RELIANCE.NS", "qty": 1000}])
    assert first == 1
    assert second == 0
    lines = (archive_dir / "2026-07-04.jsonl").read_text().splitlines()
    assert len(lines) == 1


def test_archive_new_event_same_day_appends_not_overwrites(tmp_path):
    archive_dir = tmp_path / "event_archive"
    with mock.patch.object(a, "ARCHIVE_DIR", archive_dir):
        a.archive_events("2026-07-04", bulk_deals=[{"ticker": "RELIANCE.NS", "qty": 1000}])
        a.archive_events("2026-07-04", bulk_deals=[{"ticker": "TCS.NS", "qty": 500}])
    lines = (archive_dir / "2026-07-04.jsonl").read_text().splitlines()
    assert len(lines) == 2


def test_archive_never_raises_on_bad_input(tmp_path):
    archive_dir = tmp_path / "event_archive"
    with mock.patch.object(a, "ARCHIVE_DIR", archive_dir):
        n = a.archive_events("2026-07-04", weird_source="not a list or dict")
    assert n == 0


def test_archive_skips_empty_sources(tmp_path):
    archive_dir = tmp_path / "event_archive"
    with mock.patch.object(a, "ARCHIVE_DIR", archive_dir):
        n = a.archive_events("2026-07-04", bulk_deals=[], promoter_signals={})
    assert n == 0
    assert not (archive_dir / "2026-07-04.jsonl").exists()
