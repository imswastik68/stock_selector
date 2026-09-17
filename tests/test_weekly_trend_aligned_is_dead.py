"""`weekly_trend_aligned` is declared but never computed — pin that.

src/scorer.py carried `"weekly_trend_aligned": 1` from the start, but
src/technicals.py never sets the key, so every read site falls through to
`.get("weekly_trend_aligned", False)`. It fired on **0 of 291,314** rows in
outputs/backtest_trades.csv and on 0 live picks.

Zeroed 2026-09-17 rather than implemented: wiring up the weekly EMA10>EMA20
calculation would silently activate a +1 weight that has never been measured
against a holdout, which is exactly what this repo's discipline forbids.

These tests fail if someone implements the signal without also validating it,
turning a silent behaviour change into a visible one.
"""

from __future__ import annotations

from src.scorer import SHORT_TERM_WEIGHTS, _compute_score


def test_weight_is_zero_until_validated():
    assert SHORT_TERM_WEIGHTS["weekly_trend_aligned"] == 0, (
        "weekly_trend_aligned has a non-zero weight. Nothing computes this "
        "signal, so the weight is unearnable — and if you HAVE just implemented "
        "it, backtest it against a 70/30 holdout before giving it a weight."
    )


def test_scoring_it_on_changes_nothing():
    """Even if a caller passes it explicitly, it must not move the score."""
    base, _ = _compute_score({"rsi_momentum": True})
    withsig, _ = _compute_score({"rsi_momentum": True, "weekly_trend_aligned": True})
    assert withsig == base, (
        "weekly_trend_aligned moved the score; it is unvalidated and must not"
    )


def test_technicals_still_does_not_compute_it():
    """The guard that makes the two tests above meaningful.

    If this starts failing, the signal is now being computed — go validate it
    and only then give it a weight.
    """
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src" / "technicals.py"
    text = src.read_text()
    assert "weekly_trend_aligned" not in text, (
        "src/technicals.py now computes weekly_trend_aligned. Backtest it "
        "(70/30 holdout, date-clustered inference) before weighting it."
    )
