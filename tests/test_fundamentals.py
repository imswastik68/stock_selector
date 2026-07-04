"""
Regression test for the audit fix to src/data/fundamentals.py's debtToEquity
normalization. yfinance's debtToEquity is ALWAYS a percentage (e.g. 36.2
meaning a true ratio of 0.362x), never a raw ratio -- the old conditional
(`/100 if de_ratio > 10 else de_ratio`) left true D/E 0.03-0.10 (reported as
3-10) unscaled, misclassifying low-debt companies as fundamental_weak.
Confirmed live: TATAELXSI.NS reported D/E=5.3 (true ~0.05).
"""

from __future__ import annotations

import pytest

from src.data.fundamentals import _derive_signals


def _info(roe=0.10, de=None, eps_growth=0.05, eps=10.0, margins=0.10) -> dict:
    return {
        "returnOnEquity": roe, "debtToEquity": de,
        "earningsQuarterlyGrowth": eps_growth, "trailingEps": eps, "profitMargins": margins,
    }


def test_low_debt_company_not_misclassified_weak():
    """TATAELXSI.NS regression case: reported D/E=5.3 (true ~0.053) must NOT
    trigger fundamental_weak (threshold is de_normalized > 3.0)."""
    r = _derive_signals(_info(roe=0.213, de=5.3))
    assert r["de_ratio"] == pytest.approx(0.053, abs=1e-6)
    assert r["fundamental_weak"] is False


def test_high_percentage_debt_still_scales_correctly():
    """PIRAMALFIN.NS-style case: D/E=280 -> 2.8 true ratio (below the 3.0
    weak threshold, so 'neutral' not 'weak' -- confirms scaling, not the
    threshold itself)."""
    r = _derive_signals(_info(roe=0.05, de=280))
    assert r["de_ratio"] == pytest.approx(2.8, abs=1e-6)


def test_genuinely_high_debt_flagged_weak():
    """D/E=350 -> true ratio 3.5, above the weak threshold -- must flag weak."""
    r = _derive_signals(_info(roe=0.05, de=350))
    assert r["de_ratio"] == pytest.approx(3.5, abs=1e-6)
    assert r["fundamental_weak"] is True


def test_strong_classification_still_works_with_correct_scaling():
    """High ROE + low D/E (true ratio < 1.0) + positive EPS growth -> strong."""
    r = _derive_signals(_info(roe=0.20, de=40, eps_growth=0.10))  # 40 -> 0.4x
    assert r["de_ratio"] == pytest.approx(0.4, abs=1e-6)
    assert r["fundamental_strong"] is True


def test_missing_debt_to_equity_returns_none():
    r = _derive_signals(_info(de=None))
    assert r["de_ratio"] is None
