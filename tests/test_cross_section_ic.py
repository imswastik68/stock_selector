"""
Tests for src/cross_section.py -- the rank-IC instrumentation.

The point of this module is to answer "does `score` order forward returns?", so
the tests that matter are the ones that plant a KNOWN answer and check the
measurement recovers it:

  - a cross-section where high score really does lead to high forward return
    must come back with strongly positive IC,
  - the reverse must come back strongly negative,
  - noise must come back near zero,

plus the persistence/verdict plumbing around it. A measurement instrument that
can't be shown to read correctly on a known input is worth nothing.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

pd = pytest.importorskip("pandas")

from src import cross_section as cs


# ── rank correlation ──────────────────────────────────────────────────────────

def test_spearman_recovers_known_relationships():
    assert cs._spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)
    assert cs._spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(-1.0)
    # Monotone but non-linear: rank correlation should still be perfect where
    # Pearson would not be.
    assert cs._spearman([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)


def test_spearman_handles_degenerate_input():
    assert cs._spearman([1, 1, 1], [5, 5, 5]) is None   # no variance
    assert cs._spearman([1, 2], [1, 2]) is None          # n < 3


# ── recording ─────────────────────────────────────────────────────────────────

@pytest.fixture
def _tmp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "_CS_DIR", tmp_path / "cross_section")
    monkeypatch.setattr(cs, "_IC_FILE", tmp_path / "score_ic.json")
    return tmp_path


def test_record_cross_section_persists_all_rows(_tmp_paths):
    rows = [{"ticker": "A.NS", "score": 5, "close": 100.0, "qualified": True},
            {"ticker": "B.NS", "score": 1, "close": 50.0, "qualified": False}]
    cs.record_cross_section("2026-07-22", rows)

    written = json.loads((cs._CS_DIR / "2026-07-22.json").read_text())
    assert written["n"] == 2
    # Sub-threshold names must be kept -- they're the whole reason this exists.
    assert {r["ticker"] for r in written["rows"]} == {"A.NS", "B.NS"}


def test_record_cross_section_is_write_once(_tmp_paths):
    cs.record_cross_section("2026-07-22", [{"ticker": "A.NS", "score": 5, "close": 100.0}])
    cs.record_cross_section("2026-07-22", [{"ticker": "Z.NS", "score": 9, "close": 900.0}])

    written = json.loads((cs._CS_DIR / "2026-07-22.json").read_text())
    assert [r["ticker"] for r in written["rows"]] == ["A.NS"]  # not overwritten


def test_record_cross_section_skips_rows_without_a_price(_tmp_paths):
    cs.record_cross_section("2026-07-22", [
        {"ticker": "A.NS", "score": 5, "close": 100.0},
        {"ticker": "NOPRICE.NS", "score": 7, "close": None},
    ])
    written = json.loads((cs._CS_DIR / "2026-07-22.json").read_text())
    assert [r["ticker"] for r in written["rows"]] == ["A.NS"]


# ── IC computation against a planted answer ───────────────────────────────────

def _plant(monkeypatch, tmp_path, n_days: int, n_names: int, relationship: str):
    """Write `n_days` cross-sections and a fake yfinance whose forward returns
    follow `relationship` ('positive' | 'negative' | 'none') w.r.t. score."""
    scan_dates = [(date.today() - timedelta(days=40 + d * 3)).isoformat() for d in range(n_days)]
    tickers = [f"T{i}.NS" for i in range(n_names)]

    for sd in scan_dates:
        cs.record_cross_section(sd, [
            {"ticker": t, "score": i, "close": 100.0, "qualified": True}
            for i, t in enumerate(tickers)
        ])

    # Build one price frame per ticker: flat 100 up to every scan date, then a
    # step whose size encodes the planted relationship.
    idx = pd.date_range(end=pd.Timestamp(date.today()), periods=400, freq="D")
    frames = {}
    for i, t in enumerate(tickers):
        if relationship == "positive":
            fwd = 100.0 + i          # higher score -> higher forward price
        elif relationship == "negative":
            fwd = 100.0 + (n_names - i)
        else:
            fwd = 100.0 + ((i * 37) % 11)  # deterministic but score-unrelated
        frames[t] = pd.DataFrame({"Close": [fwd] * len(idx)}, index=idx)

    combined = pd.concat(frames, axis=1)  # MultiIndex (ticker, field)

    fake_yf = type("_FakeYF", (), {
        "download": staticmethod(lambda *a, **k: combined),
    })
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)


def test_ic_detects_a_planted_positive_ranking(_tmp_paths, monkeypatch):
    _plant(monkeypatch, _tmp_paths, n_days=3, n_names=25, relationship="positive")
    result = cs.compute_score_ic()

    assert result["n_days"] == 3
    # Score and forward return are perfectly co-ranked by construction.
    assert result["mean_ic"] == pytest.approx(1.0, abs=0.01)


def test_ic_detects_a_planted_inverse_ranking(_tmp_paths, monkeypatch):
    _plant(monkeypatch, _tmp_paths, n_days=3, n_names=25, relationship="negative")
    result = cs.compute_score_ic()

    assert result["mean_ic"] == pytest.approx(-1.0, abs=0.01)


def test_ic_reports_near_zero_for_unrelated_returns(_tmp_paths, monkeypatch):
    _plant(monkeypatch, _tmp_paths, n_days=3, n_names=25, relationship="none")
    result = cs.compute_score_ic()

    assert abs(result["mean_ic"]) < 0.5  # no systematic ordering


def test_thin_cross_sections_are_skipped_not_averaged(_tmp_paths, monkeypatch):
    """A day with fewer than _MIN_NAMES_PER_DAY names is too thin for its rank
    correlation to mean anything."""
    _plant(monkeypatch, _tmp_paths, n_days=2,
           n_names=cs._MIN_NAMES_PER_DAY - 1, relationship="positive")
    result = cs.compute_score_ic()

    assert result["n_days"] == 0
    assert result["verdict"] == "INSUFFICIENT"


def test_verdict_stays_insufficient_below_min_days(_tmp_paths, monkeypatch):
    """Even a perfect IC must not read as proven on a handful of days."""
    _plant(monkeypatch, _tmp_paths, n_days=3, n_names=25, relationship="positive")
    result = cs.compute_score_ic()

    assert result["mean_ic"] == pytest.approx(1.0, abs=0.01)
    assert result["verdict"] == "INSUFFICIENT"
    assert result["n_days"] < cs._MIN_DAYS


def test_unmatured_cross_sections_are_excluded(_tmp_paths, monkeypatch):
    """Today's scan has no forward return yet and must not be scored."""
    cs.record_cross_section(date.today().isoformat(), [
        {"ticker": f"T{i}.NS", "score": i, "close": 100.0} for i in range(25)
    ])
    result = cs.compute_score_ic()
    assert result["n_days"] == 0


# ── report line ───────────────────────────────────────────────────────────────

def test_report_line_shows_progress_while_insufficient(_tmp_paths):
    line = cs.ic_report_line({"n_days": 4, "min_days": 30, "verdict": "INSUFFICIENT"})
    assert "INSUFFICIENT" in line and "4/30" in line


def test_report_line_shows_verdict_once_measured(_tmp_paths):
    line = cs.ic_report_line({"n_days": 40, "min_days": 30, "verdict": "RANKS",
                              "mean_ic": 0.041, "t_stat": 2.4})
    assert "RANKS" in line and "0.041" in line
