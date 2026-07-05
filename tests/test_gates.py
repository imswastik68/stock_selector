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
