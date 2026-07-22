"""
Tests for two 2026-07 changes to src/agent.py:

1. Alert-volume caps (MAX_ACTIONABLE / MAX_PHASE_B). The daily alert used to
   emit up to 10 actionable + 8 watch = 18 names to hand-check; cut to 5 + 3.
   These assert the caps are actually applied by _build_entries' truncation --
   the counts are the whole point of the change, so a silent revert should fail
   loudly here.

2. The Gemini backend + _caller_chain ordering. Gemini leads (free tier, better
   narrative prose than llama-3.3-70b), with Groq then Ollama as fallbacks.
   Two properties matter and are easy to break:
     - a keyless cloud backend must be SKIPPED, not crash the scan (the repo
       ran on Groq for months with no GEMINI_API_KEY set -- that must keep
       working untouched),
     - Ollama must never be attempted in CI, where there's no Ollama to reach.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from src import agent


# ── alert-volume caps ─────────────────────────────────────────────────────────

def test_caps_are_the_intended_counts():
    assert agent.MAX_ACTIONABLE == 5
    assert agent.MAX_PHASE_B == 3


def test_build_entries_truncates_actionable_list_to_max_actionable():
    """The real _build_entries, fed more qualifying buys than the cap, must
    return exactly MAX_ACTIONABLE of them -- highest score first."""
    n = 12
    candidates = [{"ticker": f"T{i}.NS", "score": 20 - i, "active_signals": []}
                  for i in range(n)]
    market_context = {
        "technicals": {
            f"T{i}.NS": {"wyckoff_phase": "MARKUP", "direction": "buy",
                         "today_close": 100.0, "rsi": 60}
            for i in range(n)
        },
        "atr_pct": {f"T{i}.NS": 2.0 for i in range(n)},
        "beta":    {f"T{i}.NS": 1.0 for i in range(n)},
    }

    buy_list, sell_list, _, _ = agent._build_entries(candidates, market_context, "uptrend")

    assert len(buy_list) + len(sell_list) == agent.MAX_ACTIONABLE == 5
    # Highest-scoring names survive the cut, in descending score order.
    assert [e["ticker"] for e in buy_list] == [f"T{i}.NS" for i in range(5)]


def test_build_entries_truncates_watch_list_to_max_phase_b():
    """Candidates in a non-actionable Wyckoff phase route to phase_b, capped."""
    n = 9
    candidates = [{"ticker": f"W{i}.NS", "score": 20 - i, "active_signals": []}
                  for i in range(n)]
    market_context = {
        "technicals": {
            f"W{i}.NS": {"wyckoff_phase": "ACCUMULATION_A", "direction": "watch",
                         "today_close": 100.0, "rsi": 50}
            for i in range(n)
        },
        "atr_pct": {f"W{i}.NS": 2.0 for i in range(n)},
        "beta":    {f"W{i}.NS": 1.0 for i in range(n)},
    }

    _, _, phase_b, _ = agent._build_entries(candidates, market_context, "uptrend")

    assert len(phase_b) == agent.MAX_PHASE_B == 3


# ── backend selection ─────────────────────────────────────────────────────────

def test_caller_chain_default_prefers_gemini_then_groq_then_ollama(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("GROQ_API_KEY", "q")
    assert [c[0] for c in agent._caller_chain("gemini", in_ci=False)] == ["gemini", "groq", "ollama"]


def test_caller_chain_respects_a_non_default_primary(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("GROQ_API_KEY", "q")
    assert [c[0] for c in agent._caller_chain("groq", in_ci=False)][0] == "groq"
    assert [c[0] for c in agent._caller_chain("ollama", in_ci=False)][0] == "ollama"


def test_ollama_is_never_attempted_in_ci(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("GROQ_API_KEY", "q")
    assert "ollama" not in [c[0] for c in agent._caller_chain("gemini", in_ci=True)]


def _fake_openai(contacted: list):
    class FakeClient:
        def __init__(self, base_url=None, api_key=None):
            contacted.append(base_url)
            msg = MagicMock()
            msg.choices = [MagicMock()]
            msg.choices[0].message.content = '[{"ticker": "X", "narrative": "ok"}]'
            self.chat = MagicMock()
            self.chat.completions.create = MagicMock(return_value=msg)

    mod = MagicMock()
    mod.OpenAI = FakeClient
    return mod


def test_keyless_gemini_falls_back_to_groq_instead_of_failing(monkeypatch):
    """The repo has no GEMINI_API_KEY set by default -- scans must keep running
    on Groq, and must not waste a call on the keyless Gemini endpoint."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "real-key")
    contacted: list = []
    with patch.dict(sys.modules, {"openai": _fake_openai(contacted)}):
        out = agent._llm_call("hi", "gemini", in_ci=True)

    assert out is not None
    assert contacted == [agent._GROQ_BASE]


def test_gemini_is_used_when_its_key_is_present(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "real-key")
    monkeypatch.setenv("GROQ_API_KEY", "also-real")
    contacted: list = []
    with patch.dict(sys.modules, {"openai": _fake_openai(contacted)}):
        agent._llm_call("hi", "gemini", in_ci=True)

    assert contacted == [agent._GEMINI_BASE]
