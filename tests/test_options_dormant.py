"""
src/data/options.py is dormant (OPTIONS_FETCH_LIVE = False).

NSE blocks the option-chain API from scripted clients -- the cookie handshake
403s and option-chain-equities returns a bare "{}", verified by hand against
RELIANCE/INFY/SBIN. The fetch therefore burned scan time on calls that could
never succeed while printing "non-F&O or fetch error", which misattributed the
cause.

These tests pin the two properties that make disabling it safe:
  - it performs NO network I/O while dormant,
  - it still returns the full signal dict for every ticker, so every downstream
    consumer (scorer's options_* lookups, agent's entry rendering) sees the same
    shape it always did and simply finds every signal False.
"""

from __future__ import annotations

import pytest

from src.data import options


def test_dormant_flag_is_off():
    assert options.OPTIONS_FETCH_LIVE is False


def test_dormant_fetch_makes_no_network_calls(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("options fetch attempted network I/O while dormant")

    monkeypatch.setattr(options.requests, "Session", _boom)
    monkeypatch.setattr(options, "_make_session", _boom)

    result = options.fetch_options_signals(["RELIANCE.NS", "SBIN.NS"])
    assert set(result) == {"RELIANCE.NS", "SBIN.NS"}


def test_dormant_fetch_returns_the_full_signal_shape():
    """Downstream code indexes these keys directly -- a truncated dict would
    turn a dead data source into a KeyError."""
    result = options.fetch_options_signals(["RELIANCE.NS"])
    sig = result["RELIANCE.NS"]

    assert set(sig) == set(options._EMPTY)
    # Every signal reads as "absent", not as a live reading.
    assert sig["pcr"] is None
    assert not any(sig[k] for k in sig if isinstance(sig[k], bool))


def test_empty_ticker_list_still_short_circuits():
    assert options.fetch_options_signals([]) == {}


def test_dormant_signals_contribute_zero_score():
    """The scoring rationale for leaving this off: no options signal that reads
    positive carries weight anyway."""
    from src.scorer import SHORT_TERM_WEIGHTS

    for sig in ("options_pcr_fear", "options_long_buildup", "options_short_covering"):
        assert SHORT_TERM_WEIGHTS.get(sig, 0) == 0
