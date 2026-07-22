"""
Regression test: a failed .info fetch must NOT be written to the fundamentals
cache.

yfinance throttles .info with HTTP 429 when called in a tight loop. The old
code cached the resulting all-None "neutral" record stamped with today's date;
the 7-day TTL then served that non-answer for a week without ever retrying,
silently disabling fundamental_strong / fundamental_weak on exactly the tickers
that had failed. Observed live on 2026-07-22: all 20 candidates 429'd and all 20
were cached as data-less.

A fetch failure is an absence of data, not a measurement of it.
"""

from __future__ import annotations

import json

import pytest

from src.data import fundamentals


class _FakeTicker:
    """yfinance.Ticker stand-in: raises for tickers in `failing`."""

    def __init__(self, symbol: str, failing: set[str], payload: dict):
        self._symbol = symbol
        self._failing = failing
        self._payload = payload

    @property
    def info(self) -> dict:
        if self._symbol in self._failing:
            raise RuntimeError("Too Many Requests. Rate limited. Try after a while.")
        return self._payload


@pytest.fixture
def _isolated_cache(tmp_path, monkeypatch):
    """Point the module at a throwaway cache file and remove the inter-call
    sleep so the test stays fast."""
    cache_file = tmp_path / "fundamentals_cache.json"
    monkeypatch.setattr(fundamentals, "_CACHE_FILE", cache_file)
    monkeypatch.setattr(fundamentals, "_FETCH_DELAY_SECONDS", 0)
    return cache_file


def _install_fake_yf(monkeypatch, failing: set[str], payload: dict):
    fake_yf = type("_FakeYF", (), {
        "Ticker": staticmethod(lambda s: _FakeTicker(s, failing, payload)),
    })
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)


_GOOD_INFO = {
    "returnOnEquity": 0.24,
    "debtToEquity": 20.0,
    "earningsQuarterlyGrowth": 0.15,
    "trailingEps": 12.0,
    "profitMargins": 0.18,
}


def test_failed_fetch_is_not_cached(_isolated_cache, monkeypatch):
    _install_fake_yf(monkeypatch, failing={"BAD.NS"}, payload=_GOOD_INFO)

    result = fundamentals.fetch_fundamental_signals(["GOOD.NS", "BAD.NS"])

    # Both get a neutral-shaped answer for this run...
    assert result["BAD.NS"]["fundamental_strong"] is False
    assert result["BAD.NS"]["roe"] is None

    # ...but only the successful one is persisted, so BAD.NS retries next run
    # instead of being pinned to "no data" for the 7-day TTL.
    cached = json.loads(_isolated_cache.read_text())
    assert "GOOD.NS" in cached
    assert "BAD.NS" not in cached


def test_successful_fetch_is_cached_and_reused(_isolated_cache, monkeypatch):
    _install_fake_yf(monkeypatch, failing=set(), payload=_GOOD_INFO)
    fundamentals.fetch_fundamental_signals(["GOOD.NS"])

    # Second call with a now-always-failing yfinance still returns the real
    # cached values -- proving the cache is read, not silently refetched.
    _install_fake_yf(monkeypatch, failing={"GOOD.NS"}, payload=_GOOD_INFO)
    again = fundamentals.fetch_fundamental_signals(["GOOD.NS"])

    assert again["GOOD.NS"]["roe"] == 0.24


def test_all_failing_leaves_cache_empty(_isolated_cache, monkeypatch):
    """The 2026-07-22 scenario: every ticker 429s. Nothing should persist."""
    tickers = [f"T{i}.NS" for i in range(5)]
    _install_fake_yf(monkeypatch, failing=set(tickers), payload=_GOOD_INFO)

    result = fundamentals.fetch_fundamental_signals(tickers)

    assert all(result[t]["roe"] is None for t in tickers)
    cached = json.loads(_isolated_cache.read_text()) if _isolated_cache.exists() else {}
    assert cached == {}
