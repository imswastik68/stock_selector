"""
src/data/options.py, rebuilt on the EOD F&O bhavcopy (2026-07).

The live option-chain API is blocked for scripted clients -- the cookie
handshake 403s and api/option-chain-equities returns a bare "{}", verified by
hand against RELIANCE/INFY/SBIN, the three most liquid F&O names in India. The
old code reported that as "non-F&O or fetch error", which misattributed a hard
block as a property of the tickers.

The archive path (nsearchives.nseindia.com) is not blocked and is the
authoritative settlement record. One ~1.2 MB zip per trading day carries every
contract: strike, expiry, OI, OI change, volume, settlement, underlying price.

These tests use a synthetic bhavcopy frame so they neither hit the network nor
depend on a particular day's market state. What they pin:
  - PCR is computed from the NEAREST expiry only,
  - a genuinely non-F&O ticker returns empty signals rather than raising,
  - the returned dict keeps the exact shape downstream consumers index into,
  - illiquid chains are gated out,
  - one download serves the whole ticker list (it is a market-wide file).
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from src.data import options


def _bhav() -> "pd.DataFrame":
    """Two expiries for LIQUID, plus an illiquid name. Front expiry is put-heavy
    (PCR 2.0); the far expiry is deliberately call-heavy so a test can prove the
    far month is excluded."""
    rows = []
    for strike in (100, 110):
        rows += [
            {"TckrSymb": "LIQUID", "FinInstrmTp": "STO", "XpryDt": "2026-07-30",
             "OptnTp": "CE", "StrkPric": strike, "OpnIntrst": 5000,
             "ChngInOpnIntrst": 500, "UndrlygPric": 105.0},
            {"TckrSymb": "LIQUID", "FinInstrmTp": "STO", "XpryDt": "2026-07-30",
             "OptnTp": "PE", "StrkPric": strike, "OpnIntrst": 10000,
             "ChngInOpnIntrst": 800, "UndrlygPric": 105.0},
            # Far expiry: call-heavy, must NOT influence PCR.
            {"TckrSymb": "LIQUID", "FinInstrmTp": "STO", "XpryDt": "2026-08-27",
             "OptnTp": "CE", "StrkPric": strike, "OpnIntrst": 900000,
             "ChngInOpnIntrst": 0, "UndrlygPric": 105.0},
            {"TckrSymb": "LIQUID", "FinInstrmTp": "STO", "XpryDt": "2026-08-27",
             "OptnTp": "PE", "StrkPric": strike, "OpnIntrst": 1,
             "ChngInOpnIntrst": 0, "UndrlygPric": 105.0},
        ]
    # Below _MIN_CALL_OI -- signals must stay off.
    rows.append({"TckrSymb": "ILLIQUID", "FinInstrmTp": "STO", "XpryDt": "2026-07-30",
                 "OptnTp": "CE", "StrkPric": 50, "OpnIntrst": 10,
                 "ChngInOpnIntrst": 1, "UndrlygPric": 55.0})
    rows.append({"TckrSymb": "ILLIQUID", "FinInstrmTp": "STO", "XpryDt": "2026-07-30",
                 "OptnTp": "PE", "StrkPric": 50, "OpnIntrst": 40,
                 "ChngInOpnIntrst": 1, "UndrlygPric": 55.0})
    return pd.DataFrame(rows)


def test_pcr_uses_the_nearest_expiry_only():
    """Front expiry is 2:1 put-heavy; the far expiry is overwhelmingly
    call-heavy. Summing both would swamp the number that matters for a swing."""
    sig = options._signals_from_bhavcopy(_bhav(), "LIQUID", prev_close=100.0)
    assert sig["pcr"] == pytest.approx(2.0)


def test_oi_buildup_direction_uses_price_vs_prev_close():
    df = _bhav()
    # Underlying 105 > prev 100, and net OI change is positive -> long buildup.
    up = options._signals_from_bhavcopy(df, "LIQUID", prev_close=100.0)
    assert up["long_buildup"] is True
    assert up["short_buildup"] is False

    # Same OI expansion but price DOWN -> short buildup instead.
    down = options._signals_from_bhavcopy(df, "LIQUID", prev_close=110.0)
    assert down["short_buildup"] is True
    assert down["long_buildup"] is False


def test_missing_prev_close_leaves_buildup_signals_off():
    """Direction is unknowable without a reference price -- must not guess."""
    sig = options._signals_from_bhavcopy(_bhav(), "LIQUID", prev_close=None)
    for k in ("long_buildup", "short_buildup", "short_covering", "long_unwinding"):
        assert sig[k] is False


def test_illiquid_chain_is_gated_out():
    sig = options._signals_from_bhavcopy(_bhav(), "ILLIQUID", prev_close=50.0)
    assert sig["pcr_fear"] is False
    assert sig["long_buildup"] is False


def test_non_fo_ticker_returns_empty_signals():
    """An SME with no derivatives (e.g. GAUDIUMIVF) must read as absent, not
    raise and not fabricate a reading."""
    sig = options._signals_from_bhavcopy(_bhav(), "NOTLISTED", prev_close=100.0)
    assert sig == options._EMPTY
    assert sig["pcr"] is None


def test_signal_dict_shape_is_stable():
    """Downstream code indexes these keys directly."""
    sig = options._signals_from_bhavcopy(_bhav(), "LIQUID", prev_close=100.0)
    assert set(sig) == set(options._EMPTY)


def test_one_bhavcopy_download_serves_every_ticker(monkeypatch):
    """It's a market-wide file -- fetching per ticker would be the old mistake
    in a new costume."""
    calls = []

    def _fake_fetch(*a, **k):
        calls.append(1)
        return _bhav()

    monkeypatch.setattr(options, "fetch_fo_bhavcopy", _fake_fetch)
    monkeypatch.setattr(options, "_load_pcr_cache", lambda: {})
    monkeypatch.setattr(options, "_save_pcr_cache", lambda c: None)

    options.fetch_options_signals(["LIQUID.NS", "ILLIQUID.NS", "NOTLISTED.NS"])
    assert len(calls) == 1


def test_unavailable_bhavcopy_degrades_to_empty_signals(monkeypatch):
    """A failed archive fetch must not crash the scan."""
    monkeypatch.setattr(options, "fetch_fo_bhavcopy", lambda *a, **k: None)

    result = options.fetch_options_signals(["RELIANCE.NS", "SBIN.NS"])
    assert set(result) == {"RELIANCE.NS", "SBIN.NS"}
    assert all(v == options._EMPTY for v in result.values())


def test_options_signals_still_carry_zero_score_weight():
    """Deliberate: these are wired for DATA and backtesting, not scoring. They
    stay at 0 until scripts/backtest_events.py returns a SHIP verdict."""
    from src.scorer import SHORT_TERM_WEIGHTS

    for sig in ("options_pcr_fear", "options_long_buildup", "options_short_covering"):
        assert SHORT_TERM_WEIGHTS.get(sig, 0) == 0
