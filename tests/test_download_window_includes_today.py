"""yfinance's `end` is EXCLUSIVE -- the candidate feeds must ask for today+1.

Measured 2026-09-17 on 281 live Telegram picks: the price quoted in every EOD
alert (scans run 18:00-23:00 IST, NSE closes 15:30) matched the PREVIOUS day's
close 136/138 times. Cause: `end = date.today()` in the four yfinance feeds,
which drops today's bar, so `closes.iloc[-1]` was yesterday. Every "52-week
breakout" was a day-old breakout: picks gapped +0.34% overnight (t=+3.94) and
were up only 36.8% one day later.

These tests pin the window so the off-by-one cannot come back. They assert on
the `end` kwarg actually passed to yf.download rather than on downloaded data,
so they are deterministic and need no network.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

import src.data.breakdowns as breakdowns
import src.data.breakouts as breakouts
import src.data.reversal as reversal
import src.data.volume as volume


def _capture(monkeypatch, module) -> list[dict]:
    """Stub out yf.download on `module`, recording the kwargs it was called with."""
    calls: list[dict] = []

    def fake_download(*args, **kwargs):
        calls.append(kwargs)
        return pd.DataFrame()  # empty -> _parse_batch yields nothing, fetch returns []

    monkeypatch.setattr(module.yf, "download", fake_download)
    return calls


@pytest.fixture(autouse=True)
def _no_cache_no_premarket(monkeypatch, tmp_path):
    """Force the real download path: no same-day cache, not pre-market."""
    monkeypatch.delenv("SCAN_MODE", raising=False)
    for mod in (breakouts, breakdowns, reversal, volume):
        monkeypatch.setattr(mod, "save_today", lambda *a, **k: None, raising=False)
    monkeypatch.setattr("src.cache.load_today", lambda *a, **k: None)
    monkeypatch.setattr("src.cache.load_latest", lambda *a, **k: None)
    monkeypatch.setattr("src.cache.save_today", lambda *a, **k: None)


@pytest.mark.parametrize(
    "module, fetch_name",
    [
        (breakouts, "fetch_breakouts"),
        (breakdowns, "fetch_breakdowns"),
        (reversal, "fetch_reversal_candidates"),
    ],
)
def test_end_is_tomorrow_so_todays_bar_is_included(monkeypatch, module, fetch_name):
    fetch = getattr(module, fetch_name, None)
    if fetch is None:
        pytest.skip(f"{module.__name__} has no {fetch_name}")
    calls = _capture(monkeypatch, module)
    monkeypatch.setattr(module, "_load_universe", lambda: ["AAA.NS", "BBB.NS"])

    fetch()

    assert calls, "yf.download was never called -- test would pass vacuously"
    tomorrow = date.today() + timedelta(days=1)
    for kwargs in calls:
        assert kwargs["end"] == tomorrow, (
            f"{module.__name__} passed end={kwargs['end']}; yfinance's end is "
            f"exclusive so it must be {tomorrow} to include today's bar"
        )
        # start must still trail end, i.e. the window didn't collapse
        assert kwargs["start"] < kwargs["end"]


def test_volume_batch_download_includes_today(monkeypatch):
    calls = _capture(monkeypatch, volume)
    volume._batch_download(["AAA.NS"], lookback_days=36)
    assert calls, "yf.download was never called"
    assert calls[0]["end"] == date.today() + timedelta(days=1)


def test_yfinance_end_really_is_exclusive():
    """Pin the upstream behaviour this fix depends on. If yfinance ever makes
    `end` inclusive, the +1 becomes an off-by-one the other way and this fails."""
    import inspect

    import yfinance as yf

    doc = (inspect.getdoc(yf.download) or "").lower()
    if "exclusive" not in doc:
        pytest.skip("yfinance docstring does not state end-date semantics")
    assert "exclusive" in doc
