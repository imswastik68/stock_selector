"""Regenerate the auto section of SESSION_LOG.md from live outputs/ state.

Run manually or from CI after the EOD scan. Only rewrites the block
between the AUTO-GENERATED markers -- the rest of the file is
hand-maintained. Safe to run with missing outputs files (fresh checkout,
pre-first-scan) -- each section degrades to "not available yet" instead
of crashing.
"""
from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "SESSION_LOG.md"
BEGIN = "<!-- AUTO-GENERATED:BEGIN -- do not hand-edit between these markers.\n     Regenerated daily by scripts/update_session_log.py (runs in CI after\n     the EOD scan). Edit the script if the content needs to change. -->"
END = "<!-- AUTO-GENERATED:END -->"


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _score_ic_line() -> str:
    d = _load_json(ROOT / "outputs" / "score_ic.json")
    if not d:
        return "- **Score IC:** not available yet (outputs/score_ic.json missing)."
    return (
        f"- **Score IC:** {d.get('n_days')} days recorded (needs "
        f"{d.get('min_days')}), mean IC {d.get('mean_ic')}, "
        f"t={d.get('t_stat')}, verdict **{d.get('verdict')}**."
    )


def _momentum_gate_line() -> str:
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        from src.gates import momentum_gate
        g = momentum_gate()
        return f"- **Momentum gate:** {'LIVE' if g.get('live') else 'PAPER-ONLY'} -- {g.get('reason')}"
    except Exception as exc:  # pragma: no cover - defensive, CI must not die on this
        return f"- **Momentum gate:** could not evaluate ({exc})."


def _portfolio_line() -> str:
    d = _load_json(ROOT / "outputs" / "portfolio.json")
    if not d:
        return "- **Portfolio:** not available yet (outputs/portfolio.json missing)."
    holdings = list(d.get("holdings", {}).keys())
    return (
        f"- **Portfolio:** equity {d.get('equity')}, cash {d.get('cash')}, "
        f"{len(holdings)} open holdings: {', '.join(holdings) or 'none'}."
    )


def _ci_line() -> str:
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%h %ad", "--date=short"],
            cwd=ROOT, capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception:
        out = "unknown"
    return f"- **Last commit at log time:** {out}."


def build_section() -> str:
    lines = [
        f"_(as of {date.today().isoformat()}, auto-generated)_",
        "",
        _ci_line(),
        _score_ic_line(),
        _momentum_gate_line(),
        _portfolio_line(),
    ]
    return "\n".join(lines)


def main() -> None:
    text = LOG_PATH.read_text()
    if BEGIN not in text or END not in text:
        raise SystemExit("SESSION_LOG.md is missing the AUTO-GENERATED markers")
    pre, rest = text.split(BEGIN, 1)
    _, post = rest.split(END, 1)
    new_text = pre + BEGIN + "\n" + build_section() + "\n" + END + post
    LOG_PATH.write_text(new_text)


if __name__ == "__main__":
    main()
