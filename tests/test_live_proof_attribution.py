"""The PROVEN badges must disclose that they are the same trades counted N times.

live_alpha_gate attributes one pick to ALL of its active_signals, which is the
right call for measuring a signal but makes the report read as N independent
confirmations. Measured 2026-09-17 on outputs/performance.json: five signals
each reported ✅ PROVEN with n=70/71/70/60/41 -- summing to 312 -- over 71
DISTINCT trades, pairwise Jaccard up to 0.99 (actual_52w_breakout and
near_52w_high are essentially the same set). The Telegram alert showed five
green ticks for one repeated momentum bet over ~6 weeks.

That also explains the apparent contradiction where every per-signal line says
PROVEN while AGGREGATE says NO-EDGE.

These tests pin the disclosure, not the verdicts.
"""

from __future__ import annotations

from src.gates import live_alpha_gate, live_proof_report


def _pick(signals: list[str], abnormal: float) -> dict:
    return {
        "direction": "buy",
        "eval_method": "next_day_zone_v2",
        "abnormal_10d": abnormal,
        "active_signals": signals,
    }


N_COHORT = 42  # > LIVE_ALPHA_MIN_N_PER_SIGNAL (30), spread over 3 months


def _date(i: int) -> str:
    """Spread picks over 3 calendar months so they clear LIVE_ALPHA_MIN_MONTHS."""
    return f"2026-{1 + i // 14:02d}-{1 + i % 14:02d}"


def _perf_one_cohort() -> dict:
    """Picks that ALL carry the same two signals -- maximal overlap."""
    return {
        _date(i): {f"T{i}.NS": _pick(["sig_a", "sig_b"], 3.0 + (i % 5))}
        for i in range(N_COHORT)
    }


def test_fixture_actually_reaches_proven():
    """Without this the overlap tests below pass vacuously -- they short-circuit
    when nothing is PROVEN, and an INSUFFICIENT fixture would hide a regression."""
    attr = live_alpha_gate(_perf_one_cohort())["attribution"]
    assert set(attr["proven_signals"]) == {"sig_a", "sig_b"}, (
        f"fixture must produce 2 PROVEN signals, got {attr['proven_signals']}"
    )


def test_overlap_is_reported_for_a_single_repeated_cohort():
    attr = live_alpha_gate(_perf_one_cohort())["attribution"]
    assert attr["n_distinct_picks"] == N_COHORT, "distinct trades miscounted"
    assert attr["n_claimed"] == 2 * N_COHORT, "two signals x N picks should claim 2N"
    assert attr["inflation_factor"] == 2.0
    assert attr["max_pairwise_jaccard"] == 1.0, "identical cohorts must show Jaccard 1.0"
    assert set(attr["most_overlapping_pair"]) == {"sig_a", "sig_b"}


def test_report_emits_a_warning_line_when_evidence_is_inflated():
    report = live_proof_report(_perf_one_cohort())
    assert report["attribution"]["inflation_factor"] >= 1.5
    warnings = [ln for ln in report["lines"] if "independent proofs" in ln]
    assert warnings, "inflated per-signal evidence was reported with no caveat line"
    assert "distinct trades" in warnings[0]
    assert str(2 * N_COHORT) in warnings[0] and str(N_COHORT) in warnings[0]


def test_distinct_count_equals_claimed_when_signals_do_not_overlap():
    """No overlap -> no inflation -> no warning. Guards against crying wolf."""
    perf = {
        _date(i): {f"A{i}.NS": _pick(["only_a"], 3.0 + (i % 5)),
                   f"B{i}.NS": _pick(["only_b"], 3.0 + (i % 5))}
        for i in range(N_COHORT)
    }
    attr = live_alpha_gate(perf)["attribution"]
    assert set(attr["proven_signals"]) == {"only_a", "only_b"}, "fixture went vacuous"
    assert attr["max_pairwise_jaccard"] == 0.0
    assert attr["inflation_factor"] == 1.0
    assert not [ln for ln in live_proof_report(perf)["lines"]
                if "independent proofs" in ln], "warned on non-overlapping evidence"


def test_attribution_key_is_always_present():
    """Downstream formatting reads gate['attribution'] unconditionally."""
    assert "attribution" in live_alpha_gate({})
    assert live_alpha_gate({})["attribution"]["n_distinct_picks"] == 0
