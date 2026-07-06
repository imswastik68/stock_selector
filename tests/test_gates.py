"""
Tests for src/gates.py -- the automated ship/no-ship gate status that
replaces daily human eyeballing of outputs/performance.json and
outputs/factor_backtest.json.
"""

from __future__ import annotations

import json
from unittest import mock

import src.gates as g


_V2 = "next_day_zone_v2"


def _month_data(month: str, wins: int, losses: int, opens: int = 0, eval_method: str | None = _V2) -> dict:
    """Build a {scan_date: {ticker: pick}} slice all falling in `month`
    ("YYYY-MM"), with the given win/loss/open counts. Tagged v2-methodology
    by default -- pass eval_method=None to simulate legacy (pre-fix) picks,
    which the gate must exclude from its win-rate verdict."""
    picks = {}
    day = 1
    for i in range(wins):
        picks[f"W{i}.NS"] = {"outcome": "t1_hit", "eval_method": eval_method}
    for i in range(losses):
        picks[f"L{i}.NS"] = {"outcome": "sl_hit", "eval_method": eval_method}
    for i in range(opens):
        picks[f"O{i}.NS"] = {"outcome": "open", "eval_method": eval_method}
    return {f"{month}-{day:02d}": picks}


def test_scanner_gate_cleared_when_three_months_pass():
    perf = {}
    for month, wins, losses in [("2026-04", 20, 20), ("2026-05", 19, 21), ("2026-06", 25, 15)]:
        perf.update(_month_data(month, wins, losses))
    result = g.scanner_gate(perf)
    assert result["cleared"] is True
    assert len(result["months"]) == 3
    assert all(m["passes"] for m in result["months"])


def test_scanner_gate_not_cleared_when_one_month_below_min_decided():
    perf = {}
    for month, wins, losses in [("2026-04", 20, 20), ("2026-05", 19, 21), ("2026-06", 15, 5)]:  # 20 decided < 40
        perf.update(_month_data(month, wins, losses))
    result = g.scanner_gate(perf)
    assert result["cleared"] is False
    june = next(m for m in result["months"] if m["month"] == "2026-06")
    assert june["decided"] == 20
    assert june["passes"] is False


def test_scanner_gate_not_cleared_when_win_rate_just_below_bar():
    # 44.9% < 45.0% -> should fail
    perf = {}
    for month in ["2026-04", "2026-05", "2026-06"]:
        perf.update(_month_data(month, wins=449, losses=551))  # 44.9%
    result = g.scanner_gate(perf)
    assert result["cleared"] is False
    for m in result["months"]:
        assert m["wr_pct"] == 44.9
        assert m["passes"] is False


def test_scanner_gate_timeout_and_open_excluded_from_decided():
    perf = _month_data("2026-06", wins=45, losses=10, opens=100)
    result = g.scanner_gate(perf)
    june = result["months"][0]
    assert june["decided"] == 55  # opens don't count
    assert june["wins"] == 45


def test_scanner_gate_insufficient_history_not_cleared():
    perf = {}
    for month, wins, losses in [("2026-05", 50, 10), ("2026-06", 50, 10)]:  # only 2 months
        perf.update(_month_data(month, wins, losses))
    result = g.scanner_gate(perf)
    assert result["cleared"] is False
    assert len(result["months"]) == 2


def test_scanner_gate_status_line_format():
    perf = _month_data("2026-06", wins=10, losses=30)
    result = g.scanner_gate(perf)
    assert "SCANNER GATE: NOT CLEARED" in result["status_line"]
    assert "25.0%" in result["status_line"]
    assert "40 decided" in result["status_line"]


# ── v2-methodology filtering (legacy pre-fix picks must not poison the gate) ─

def test_scanner_gate_legacy_picks_excluded_from_verdict():
    """Picks with no eval_method stamp (recorded before the next_day_zone_v2
    fix) must not count toward wins/losses/decided at all."""
    perf = _month_data("2026-06", wins=2, losses=14, eval_method=None)  # legacy: 12.5% over 16
    result = g.scanner_gate(perf)
    assert result["overall_decided"] == 0
    assert result["overall_wr_pct"] is None
    assert result["legacy_decided"] == 16
    assert result["legacy_wr_pct"] == 12.5
    assert result["cleared"] is False


def test_scanner_gate_mixed_v2_and_legacy_month_counts_only_v2():
    perf = _month_data("2026-06", wins=45, losses=5)  # v2: 90% over 50
    legacy = _month_data("2026-06", wins=2, losses=14, eval_method=None)
    # merge into the same scan_date bucket -- rename legacy tickers so they
    # don't collide with (and overwrite) the v2 tickers' identical W0/L0 names
    date_key = next(iter(perf))
    legacy_renamed = {f"LEGACY_{k}": v for k, v in legacy[date_key].items()}
    perf[date_key].update(legacy_renamed)
    result = g.scanner_gate(perf)
    june = result["months"][0]
    assert june["decided"] == 50   # legacy 16 excluded
    assert june["wins"] == 45
    assert result["legacy_decided"] == 16
    assert result["legacy_wr_pct"] == 12.5


def test_scanner_gate_status_line_shows_no_decided_yet_when_v2_empty():
    result = g.scanner_gate({})
    assert "v2: no decided picks yet" in result["status_line"]


def test_scanner_gate_status_line_shows_legacy_context_when_present():
    perf = _month_data("2026-06", wins=2, losses=14, eval_method=None)
    result = g.scanner_gate(perf)
    assert "legacy 12.5% over 16" in result["status_line"]
    assert "excluded" in result["status_line"]


def test_scanner_gate_no_legacy_context_when_no_legacy_picks():
    perf = _month_data("2026-06", wins=10, losses=30)
    result = g.scanner_gate(perf)
    assert "legacy" not in result["status_line"]
    assert result["legacy_decided"] == 0
    assert result["legacy_wr_pct"] is None


# ── momentum_gate ─────────────────────────────────────────────────────────────

def test_momentum_gate_missing_file():
    with mock.patch.object(g, "OUTPUTS", g.OUTPUTS.parent / "nonexistent_dir_xyz"):
        result = g.momentum_gate()
    assert result["live"] is False
    assert "no factor_backtest.json found" in result["reason"]


def test_momentum_gate_ships_true(tmp_path):
    bt_file = tmp_path / "factor_backtest.json"
    bt_file.write_text(json.dumps({
        "strategies": {"mom_12_1": {"ship_gate_multi_split": {"ships": True}}},
    }))
    with mock.patch.object(g, "OUTPUTS", tmp_path):
        result = g.momentum_gate()
    assert result["live"] is True
    assert "mom_12_1" in result["reason"]


def test_momentum_gate_ships_false(tmp_path):
    bt_file = tmp_path / "factor_backtest.json"
    bt_file.write_text(json.dumps({
        "strategies": {
            "mom_12_1": {"ship_gate_multi_split": {"ships": False}},
            "mom_gated": {"ship_gate_multi_split": {"ships": False}},
        },
    }))
    with mock.patch.object(g, "OUTPUTS", tmp_path):
        result = g.momentum_gate()
    assert result["live"] is False
    assert "no momentum strategy has passed" in result["reason"]


def test_momentum_gate_corrupt_json(tmp_path):
    bt_file = tmp_path / "factor_backtest.json"
    bt_file.write_text("{not valid json")
    with mock.patch.object(g, "OUTPUTS", tmp_path):
        result = g.momentum_gate()
    assert result["live"] is False
    assert "unreadable" in result["reason"]


def test_momentum_gate_status_line_format(tmp_path):
    bt_file = tmp_path / "factor_backtest.json"
    bt_file.write_text(json.dumps({"strategies": {}}))
    with mock.patch.object(g, "OUTPUTS", tmp_path):
        result = g.momentum_gate()
    assert result["status_line"].startswith("MOMENTUM GATE: PAPER-ONLY")


def test_factor_scan_gate_status_delegates_to_gates_module(tmp_path):
    """scripts/factor_scan.py's _gate_status must return the same tuple as
    src.gates.momentum_gate() -- regression against re-diverging the two."""
    import factor_scan

    bt_file = tmp_path / "factor_backtest.json"
    bt_file.write_text(json.dumps({
        "strategies": {"mom_gated": {"ship_gate_multi_split": {"ships": True}}},
    }))
    with mock.patch.object(g, "OUTPUTS", tmp_path):
        is_live, reason = factor_scan._gate_status()
    assert is_live is True
    assert "mom_gated" in reason


# ── live_alpha_gate (Live-Proof Round Phase 4) ───────────────────────────────
# Gate is FROZEN: LIVE_ALPHA_MIN_N_PER_SIGNAL=30, LIVE_ALPHA_MIN_N_AGGREGATE=100,
# LIVE_ALPHA_MIN_MONTHS=3, LIVE_ALPHA_SIGNIFICANCE=0.05. These tests pin the
# math and the gate logic against hand-computed fixtures -- never loosen the
# thresholds to make a test (or real data) pass.

def _rows(month_values: dict[str, list[float]]) -> list[tuple[str, float]]:
    return [(month, v) for month, values in month_values.items() for v in values]


def test_alpha_verdict_proven_low_variance_positive_mean_3_months():
    """n=30, mean=1.0 (hand-verified: t=32.98, p=0.0), spanning 3 months."""
    rows = _rows({
        "2026-01": [1.0] * 10,
        "2026-02": [1.2] * 10,
        "2026-03": [0.8] * 10,
    })
    result = g._alpha_verdict(rows, g.LIVE_ALPHA_MIN_N_PER_SIGNAL)
    assert result["n"] == 30
    assert result["months"] == 3
    assert abs(result["mean"] - 1.0) < 1e-6
    assert abs(result["t_stat"] - 32.98) < 0.1
    assert result["p_value"] < 0.0001
    assert result["verdict"] == "PROVEN"


def test_alpha_verdict_insufficient_when_n_below_min():
    """Same distribution as the PROVEN case but only 20 rows -- n < 30."""
    rows = _rows({"2026-01": [1.0] * 7, "2026-02": [1.2] * 7, "2026-03": [0.8] * 6})
    result = g._alpha_verdict(rows, g.LIVE_ALPHA_MIN_N_PER_SIGNAL)
    assert result["n"] == 20
    assert result["verdict"] == "INSUFFICIENT"


def test_alpha_verdict_no_edge_when_mean_positive_but_not_significant():
    """n=30, mean=+0.38 (hand-verified: t=0.54, p=0.588 > 0.05) -- a small
    positive mean swamped by variance is NOT proof of an edge."""
    values = [1.8236, -4.8999, 4.0523, 5.0028, -9.4552, -6.2109, 0.9392, -1.2812, 0.216, -3.9652,
              4.697, 4.189, 0.6302, 5.9362, 2.6375, -3.9965, 2.1438, -4.4944, 4.6923, 0.0504,
              -0.6243, -3.1046, 6.4127, -0.4726, -1.8416, -1.4607, 2.9615, 2.1272, 2.3637, 2.4541]
    rows = _rows({"2026-01": values[:10], "2026-02": values[10:20], "2026-03": values[20:]})
    result = g._alpha_verdict(rows, g.LIVE_ALPHA_MIN_N_PER_SIGNAL)
    assert result["n"] == 30
    assert abs(result["mean"] - 0.384) < 0.01
    assert abs(result["p_value"] - 0.588) < 0.01
    assert result["p_value"] >= g.LIVE_ALPHA_SIGNIFICANCE
    assert result["verdict"] == "NO-EDGE"


def test_alpha_verdict_insufficient_when_months_below_min_despite_n_and_mean():
    """Same n=30/mean=1.0 as the PROVEN case, but all crammed into ONE
    calendar month -- enough n is not the same as a proven edge over time."""
    rows = _rows({"2026-01": [1.0] * 10 + [1.2] * 10 + [0.8] * 10})
    result = g._alpha_verdict(rows, g.LIVE_ALPHA_MIN_N_PER_SIGNAL)
    assert result["n"] == 30
    assert result["months"] == 1
    assert result["verdict"] == "INSUFFICIENT"


def test_alpha_verdict_insufficient_at_two_months():
    rows = _rows({"2026-01": [1.0] * 15, "2026-02": [1.0] * 15})
    result = g._alpha_verdict(rows, g.LIVE_ALPHA_MIN_N_PER_SIGNAL)
    assert result["months"] == 2
    assert result["verdict"] == "INSUFFICIENT"


def test_two_tailed_p_value_matches_hand_computation():
    """t=32.98 (n=30) should produce an effectively-zero p-value; t=0 must
    produce p=1.0 exactly (no evidence against H0 whatsoever)."""
    assert g._two_tailed_p_value(0.0, 30) == 1.0
    assert g._two_tailed_p_value(32.98, 30) < 1e-6


# ── live_alpha_gate: bucketing (attribution, buy-only, reversal stress) ─────

def _alpha_pick(direction="buy", active_signals=None, abnormal_10d=1.0, abnormal_10d_stress=None):
    return {
        "direction": direction, "eval_method": "next_day_zone_v2",
        "active_signals": active_signals or [], "abnormal_10d": abnormal_10d,
        "abnormal_10d_stress": abnormal_10d_stress,
    }


def test_live_alpha_gate_attributes_one_pick_to_multiple_signals():
    """A pick firing 2 signals counts toward BOTH signals' buckets --
    attribution, not a partition of the sample."""
    perf = {"2026-01-05": {"MULTI.NS": _alpha_pick(active_signals=["sig_a", "sig_b"])}}
    result = g.live_alpha_gate(perf)
    assert "sig_a" in result["per_signal"]
    assert "sig_b" in result["per_signal"]
    assert result["per_signal"]["sig_a"]["n"] == 1
    assert result["per_signal"]["sig_b"]["n"] == 1
    assert result["aggregate"]["n"] == 1


def test_live_alpha_gate_excludes_sell_direction_picks():
    perf = {"2026-01-05": {"SHORT.NS": _alpha_pick(direction="sell", active_signals=["sig_a"])}}
    result = g.live_alpha_gate(perf)
    assert result["aggregate"]["n"] == 0
    assert "sig_a" not in result["per_signal"]


def test_live_alpha_gate_excludes_legacy_non_v2_picks():
    perf = {"2026-01-05": {"OLD.NS": {
        "direction": "buy", "eval_method": None, "active_signals": ["sig_a"], "abnormal_10d": 1.0,
    }}}
    result = g.live_alpha_gate(perf)
    assert result["aggregate"]["n"] == 0


def test_live_alpha_gate_excludes_picks_without_computed_abnormal_10d():
    perf = {"2026-01-05": {"PENDING.NS": _alpha_pick(abnormal_10d=None)}}
    result = g.live_alpha_gate(perf)
    assert result["aggregate"]["n"] == 0


def test_live_alpha_gate_reversal_stress_only_from_reversal_signal_picks():
    perf = {
        "2026-01-05": {
            "REV.NS": _alpha_pick(active_signals=["reversal_oversold_v2"],
                                    abnormal_10d=1.0, abnormal_10d_stress=0.7),
            "OTHER.NS": _alpha_pick(active_signals=["near_52w_high"], abnormal_10d=1.0),
        },
    }
    result = g.live_alpha_gate(perf)
    assert result["reversal_oversold_v2_stress"] is not None
    assert result["reversal_oversold_v2_stress"]["n"] == 1


def test_live_alpha_gate_no_reversal_picks_gives_none_stress_verdict():
    perf = {"2026-01-05": {"OTHER.NS": _alpha_pick(active_signals=["near_52w_high"])}}
    result = g.live_alpha_gate(perf)
    assert result["reversal_oversold_v2_stress"] is None


# ── live_proof_report (Live-Proof Round Phase 5) ─────────────────────────────

def test_live_proof_report_writes_outputs_file_and_lines(tmp_path):
    perf = {"2026-01-05": {"SIG.NS": _alpha_pick(active_signals=["near_52w_high"])}}
    with mock.patch.object(g, "OUTPUTS", tmp_path):
        report = g.live_proof_report(perf)

    out_file = tmp_path / "live_proof.json"
    assert out_file.exists()
    saved = json.loads(out_file.read_text())
    assert saved["per_signal"]["near_52w_high"]["n"] == 1
    assert any("near_52w_high" in line for line in report["lines"])
    assert any("AGGREGATE" in line for line in report["lines"])


def test_live_proof_report_insufficient_line_format():
    result = {"n": 5, "months": 1, "mean": 0.5, "win_pct": 60.0, "t_stat": None,
              "p_value": None, "verdict": "INSUFFICIENT"}
    line = g._format_alpha_line("sig_x", result, 30)
    assert "INSUFFICIENT" in line
    assert "5/30" in line


def test_live_proof_report_proven_line_shows_checkmark():
    result = {"n": 30, "months": 3, "mean": 1.0, "win_pct": 90.0, "t_stat": 32.98,
              "p_value": 0.0001, "verdict": "PROVEN"}
    line = g._format_alpha_line("sig_x", result, 30)
    assert "✅" in line
    assert "PROVEN" in line


def test_live_proof_report_no_edge_line_shows_cross():
    result = {"n": 30, "months": 3, "mean": 0.38, "win_pct": 50.0, "t_stat": 0.54,
              "p_value": 0.588, "verdict": "NO-EDGE"}
    line = g._format_alpha_line("sig_x", result, 30)
    assert "❌" in line
    assert "NO-EDGE" in line


def test_live_proof_report_includes_reversal_stress_line_when_present(tmp_path):
    perf = {"2026-01-05": {"REV.NS": _alpha_pick(
        active_signals=["reversal_oversold_v2"], abnormal_10d=1.0, abnormal_10d_stress=0.7)}}
    with mock.patch.object(g, "OUTPUTS", tmp_path):
        report = g.live_proof_report(perf)
    assert any("2x-cost stress" in line for line in report["lines"])
