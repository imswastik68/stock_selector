"""
Regression tests for scripts/backtest.py's _download_and_cache stale-tail
refresh (Phase 3): a cached ticker whose last bar predates end-STALE_DAYS
must be re-downloaded, not silently reused forever. A ticker whose
re-download fails keeps its stale data as a fallback rather than vanishing.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest import mock

import pandas as pd

import backtest as bt


def _write_csv(path, last_date: date, first_date: date | None = None):
    first_date = first_date or (last_date - timedelta(days=10))
    idx = pd.bdate_range(first_date, last_date)
    df = pd.DataFrame({
        "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1_000_000.0,
    }, index=idx)
    df.to_csv(path)


def test_stale_ticker_is_redownloaded_fresh_untouched(tmp_path, monkeypatch):
    cache_dir = tmp_path / "backtest_ohlcv"
    cache_dir.mkdir()
    monkeypatch.setattr(bt, "_CACHE_DIR2", cache_dir)
    monkeypatch.setattr(bt, "CACHE_DIR", tmp_path)

    end = date(2026, 7, 4)
    _write_csv(cache_dir / "FRESH_NS.csv", last_date=end - timedelta(days=1))
    _write_csv(cache_dir / "STALE_NS.csv", last_date=end - timedelta(days=30))

    requested_batches = []

    def fake_download(batch, **kwargs):
        requested_batches.append(list(batch))
        idx = pd.bdate_range(end - timedelta(days=5), end)
        cols = pd.MultiIndex.from_product([batch, ["Open", "High", "Low", "Close", "Volume"]])
        df = pd.DataFrame(1.0, index=idx, columns=cols)
        for t in batch:
            df[(t, "Close")] = 200.0  # distinguishable from the old cached value (100.0)
        return df

    with mock.patch.object(bt.yf, "download", side_effect=fake_download):
        result = bt._download_and_cache(["FRESH.NS", "STALE.NS"], date(2020, 1, 1), end)

    # Only the stale ticker triggered a network call
    all_requested = [t for batch in requested_batches for t in batch]
    assert all_requested == ["STALE.NS"]

    # Stale ticker's CSV got overwritten with fresh data
    assert result["STALE.NS"]["Close"].iloc[-1] == 200.0
    # Fresh ticker's cached data is untouched
    assert result["FRESH.NS"]["Close"].iloc[-1] == 100.0

    reloaded = pd.read_csv(cache_dir / "STALE_NS.csv", index_col=0, parse_dates=True)
    assert reloaded["Close"].iloc[-1] == 200.0


def test_stale_ticker_falls_back_to_stale_data_if_redownload_fails(tmp_path, monkeypatch):
    cache_dir = tmp_path / "backtest_ohlcv"
    cache_dir.mkdir()
    monkeypatch.setattr(bt, "_CACHE_DIR2", cache_dir)
    monkeypatch.setattr(bt, "CACHE_DIR", tmp_path)

    end = date(2026, 7, 4)
    _write_csv(cache_dir / "STALE_NS.csv", last_date=end - timedelta(days=30))

    with mock.patch.object(bt.yf, "download", side_effect=Exception("network error")):
        result = bt._download_and_cache(["STALE.NS"], date(2020, 1, 1), end)

    # Re-download failed -- the stale data must still be present, not dropped
    assert "STALE.NS" in result
    assert result["STALE.NS"]["Close"].iloc[-1] == 100.0


def test_force_ignores_cache_entirely(tmp_path, monkeypatch):
    cache_dir = tmp_path / "backtest_ohlcv"
    cache_dir.mkdir()
    monkeypatch.setattr(bt, "_CACHE_DIR2", cache_dir)
    monkeypatch.setattr(bt, "CACHE_DIR", tmp_path)

    end = date(2026, 7, 4)
    _write_csv(cache_dir / "FRESH_NS.csv", last_date=end - timedelta(days=1))

    requested_batches = []

    def fake_download(batch, **kwargs):
        requested_batches.append(list(batch))
        idx = pd.bdate_range(end - timedelta(days=5), end)
        cols = pd.MultiIndex.from_product([batch, ["Open", "High", "Low", "Close", "Volume"]])
        df = pd.DataFrame(1.0, index=idx, columns=cols)
        for t in batch:
            df[(t, "Close")] = 300.0
        return df

    with mock.patch.object(bt.yf, "download", side_effect=fake_download):
        result = bt._download_and_cache(["FRESH.NS"], date(2020, 1, 1), end, force=True)

    all_requested = [t for batch in requested_batches for t in batch]
    assert all_requested == ["FRESH.NS"]  # even the non-stale ticker was re-requested
    assert result["FRESH.NS"]["Close"].iloc[-1] == 300.0
