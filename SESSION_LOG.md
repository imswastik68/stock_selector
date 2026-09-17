# SESSION_LOG.md

Living log for `stock_selector`. Read this first in any new session. Update
the **Last updated** entry (top) whenever you ship a change or learn
something that changes the state below. Keep it short — this is a state
dump, not a diary. Full history is in `git log`.

---

## Read order for a new session
1. This file (current state, open questions, do-not-repeat list).
2. `git log --oneline -20` — what actually shipped since this was written.
3. `outputs/score_ic.json`, `outputs/factor_backtest.json` — current
   evidence, not what's described below (numbers move every scan day).
4. `tests/test_*.py` docstrings for shipped signals — each one documents
   the backtest/holdout numbers that justified it.

## Working rules (do not violate)
- **Ship gate:** any scoring change must be backtested with a 70/30
  chronological holdout against `outputs/backtest_trades.csv` /
  `cache/backtest_ohlcv/`, and be sign-consistent train+holdout. In-sample
  wins alone are not enough — several strong in-sample results were
  rejected for failing holdout (see "Rejected on evidence" below).
- **"Make it the best" is not a license to add alphas.** Default response
  to that request is: propose, backtest, holdout-check, ship only if it
  survives.
- Mutation-check new tests (verify they fail against the pre-fix code /
  a wrong-threshold mutant).
- `git pull --rebase --autostash` before pushing — CI commits back daily
  (`archive:` / `live-proof:` commits, `[skip ci]`).
- Use `~/Downloads/miniconda3/envs/venv/bin/python` for anything needing
  pandas/yfinance.
- Never claim "fixed" from a local run only — CI is the real environment
  and lands ~1.5-3h after the cron on schedule, or check `gh run list`.

## Current state
<!-- AUTO-GENERATED:BEGIN -- do not hand-edit between these markers.
     Regenerated daily by scripts/update_session_log.py (runs in CI after
     the EOD scan). Edit the script if the content needs to change. -->
_(as of 2026-09-17, auto-generated)_

- **Last commit at log time:** f53f51f 2026-09-16.
- **Score IC:** 16 days recorded (needs 30), mean IC 0.0303, t=1.08, verdict **INSUFFICIENT**.
- **Momentum gate:** PAPER-ONLY -- no momentum strategy has passed the multi-split ship gate — PAPER-ONLY (outputs/factor_backtest.json)
- **Portfolio:** equity 101224.11, cash 11412.507370000005, 5 open holdings: PIRAMALFIN.NS, BHARATFORG.NS, KPIL.NS, NYKAA.NS, POLYCAB.NS.
<!-- AUTO-GENERATED:END -->

**Known unresolved (human-tracked, not auto-updated):**
- **Portfolio ₹1L vs ₹10L reconciliation.** `outputs/portfolio.json` is
  still on the old ~₹1L lineage, holdings from June 2026
  (PIRAMALFIN, BHARATFORG, KPIL, NYKAA, POLYCAB), `exit_policy: "static"`.
  Flagged as an open decision since 2026-08-25, still unanswered — do not
  silently reconcile it, ask first. If the user has answered this since,
  remove this bullet.
- Repo was briefly private (~2026-08-10 to ~2026-08-25), which silently
  hit a GitHub Actions billing block (jobs failed in 3-5s, "recent account
  payments have failed"). Fixed by making the repo public. Not a code bug
  — keep the repo public.

## Shipped, holdout-validated
- `rsi_overbought` = RSI>80, weight -1 (2026-08). RSI>75 and a graduated
  -1/-2/-3 variant were tried and rejected (train t=5.16/5.59 but holdout
  t=-0.82/-0.85). See `tests/test_rsi_overbought.py` docstring.
- Options data via EOD F&O bhavcopy (`src/data/options.py`) — live NSE
  option-chain API is blocked (403). Bhavcopy archive works, nearest
  expiry only.
- `options_long_unwinding` weight zeroed (was -1, backwards-signed —
  backtest showed +0.193 aggregate ret_lift, i.e. rewarding what it
  claimed to penalize).
- `options_pcr_greed` (-1) and `options_short_buildup` (+1) validated,
  weights unchanged.
- F&O watchlist bug: watch-grade candidates weren't reaching the F&O
  section (early `continue` skipped them before they joined `all_entries`).
  Fixed — they're now appended to `all_entries` too.
- Cross-section IC tracking (`src/cross_section.py`): full scored
  universe (~250-375 names/day) persisted daily so IC is computed
  honestly, not just on the range-restricted emitted picks.
- CI persistence fixes: `git pull --rebase --autostash` (dirty
  `outputs/*.json` was aborting rebase), `git add -A outputs/` instead of
  naming files explicitly (was erroring on missing pathspec).

## Rejected on evidence (don't re-propose without new data)
- Momentum-cluster cap (cap picks from the same momentum bucket) — paired
  per-day IC test, t=1.26 on 291k trades. Not significant.
- RSI>75 penalty and graduated RSI penalty — see above.
- New IVOL/Amihud alphas — would add noise at current IC, not tested
  further than that judgment.
- Buy-side options signals beyond PCR/short-buildup — NO-SHIP.

## Key files
- `src/scorer.py` — deterministic weighted score, all signal weights.
- `src/technicals.py` — technical signal computation (RSI, phase, etc).
- `src/data/options.py` — F&O bhavcopy fetch + PCR/OI signals.
- `src/cross_section.py` — full-universe score capture + rank IC.
- `src/gates.py` — momentum_gate() ship/no-ship logic.
- `main.py` — orchestration, alert building, gate wiring.
- `src/telegram_alert.py` — message formatting.
- `.github/workflows/daily_scan.yml` — CI cron, commit-back steps.
- `scripts/backtest_events.py` — per-signal backtest event collection.
