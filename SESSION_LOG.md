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

- **Last commit at log time:** 57dac1e 2026-09-17.
- **Score IC:** 16 days recorded (needs 30), mean IC 0.0303, t=1.08, verdict **INSUFFICIENT**.
- **Momentum gate:** PAPER-ONLY -- no momentum strategy has passed the multi-split ship gate — PAPER-ONLY (outputs/factor_backtest.json)
- **Portfolio:** equity 101224.11, cash 11412.507370000005, 5 open holdings: PIRAMALFIN.NS, BHARATFORG.NS, KPIL.NS, NYKAA.NS, POLYCAB.NS.
<!-- AUTO-GENERATED:END -->

**Known unresolved (human-tracked, not auto-updated):**
- **Portfolio ₹1L vs ₹10L mismatch — the paper book is DEAD.** Measured
  2026-09-17: `outputs/portfolio.json` is the old ~₹1L lineage (equity
  ~101k, cash ₹11.4k) but `RISK_CAPITAL=1000000` in CI, so
  `src/risk.py:size_position` returns ₹125k-150k notional per pick.
  `src/portfolio.py:open_positions` then skips every pick on
  `state["cash"] < notional`. **Zero positions opened since 2026-07-17
  while 57 buys were emitted in September alone.** The 5 remaining
  holdings are from June with `exit_policy: "static"` (no time cap), so
  they never free the capital. The live-proof book has not tracked the
  system for two months. Needs a decision: reset the book to ₹10L, or set
  RISK_CAPITAL to match the ~₹1L book. Do not pick one silently — ask.
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

## Measured live performance (2026-09-17, 248 Telegram picks)
Source: `scripts/fetch_telegram_history.py` → `scripts/analyse_picks.py`.
Re-run both to refresh; numbers below are point-in-time.

- **Raw SL/T1 result:** 219 closed, 43% win rate, avg win +12.2% / avg
  loss -9.7%, expectancy **-0.33%/trade gross**, ≈**-0.63% net** of the
  repo's own 0.30% cost model (`src/costs.py`).
- **Beta-adjusted alpha vs NIFTY (fwd 10d, regression intercept):**
  - all 215 matured picks: beta 0.35, **alpha +0.00%, t=0.00**
  - pre-2026-08: beta 0.96, alpha -1.23%, t=-1.55
  - 2026-08 onward: beta 0.25, alpha +2.21%, **t=+1.21 (n=32)**
  The post-August improvement is real in direction but **not
  significant**, and roughly half of the naive +3.9% "alpha" was just low
  beta during a -5% NIFTY stretch. By the repo's own bar this is
  INSUFFICIENT, not a proven edge.
- **Exit policy is NOT the problem.** Paired test (pure 10-day hold minus
  SL/T1 realised, same trades): pre-Aug +0.24pp t=+0.32, Aug+ -0.32pp
  t=-0.20. Neither significant. All six policies in
  `outputs/backtest_exits.json` have negative OOS expectancy. Do not
  re-litigate exits without new evidence — the entry signal is the
  binding constraint.
- **Pick diversity still bad:** last 40 buys had 11 distinct signal
  fingerprints; the single most common set appeared 15×. Still one
  momentum bet repeated.

## Directional accuracy (measured 2026-09-17, buy next day's open)
Answers "does a pick actually go the predicted way over 3 days / a week?"
— **no, not to any significant degree.**

| horizon | up% (all) | up% (Aug+) | beat NIFTY (Aug+) |
|---|---|---|---|
| 1d | 41.0% | 36.8% | 42.6% |
| 3d | 46.8% | 33.8% | 50.8% |
| 5d | 50.6% | 51.8% | 60.7% |
| 10d | 53.4% | 60.5% | 73.7% |

All t-stats between -1.51 and +0.87 — nothing significant. The short
horizons were actively bad (picks fall for the first ~3 days) because of
the stale-data bug below; re-measure once post-fix picks accumulate.

## Fixed 2026-09-17: scans ran one trading day late
`end = date.today()` in the four yfinance feeds (breakouts, breakdowns,
reversal, volume) — **yfinance's `end` is exclusive**, so today's bar was
never downloaded and `closes.iloc[-1]` was the previous session's close.
EOD scans run 18:00-23:00 IST (NSE closes 15:30) yet quoted the previous
day's close in 136/138 picks. Every "52-week breakout" was a day-old
breakout; picks gapped +0.34% overnight (t=+3.94) before you could buy.
Fixed to `date.today() + timedelta(days=1)`, pinned by
`tests/test_download_window_includes_today.py` (mutation-checked).
**All performance numbers above predate this fix** — they measure a
system running a day late, so they are a floor, not the system's ceiling.
Mid-day scans now see a partial current-session bar (understates
volume_ratio until close); that is conservative and still better than
quoting yesterday.

## Known reporting defects (not yet fixed)
- **`live_alpha_gate` over-counts evidence.** The 5 signals reporting
  ✅ PROVEN in `outputs/live_proof.json` (actual_52w_breakout,
  near_52w_high, weekly_trend_aligned, rs_vs_nifty, rs_quality_strong)
  are pairwise Jaccard 0.56-0.99 overlapping — **71 distinct trades
  reported as n=70+71+70+60+41=312**. The Telegram alert therefore shows
  five independent-looking proofs for what is one cohort over ~6 weeks.
  Also why per-signal says PROVEN while AGGREGATE says NO-EDGE: the
  aggregate additionally includes 57 older picks whose `active_signals`
  was never recorded (empty list), which average -2.76%.
- **`score` is never recorded on live picks.** `record_picks` in
  `src/performance.py:104-133` stores active_signals/regime/big_mover but
  not `score`, so all 285 picks in `outputs/performance.json` have
  `score: None` and live score→outcome attribution is impossible from the
  audit trail. (`analyse_picks.py` only has score because it re-parses the
  Telegram text.) One-line fix, not yet made.
- `active_signals` recording was broken before 2026-08 (0/80 in June,
  33/107 in July, 41/41 Aug, 57/57 Sep). Now fixed; historical gap
  permanently contaminates any pre-August aggregate.

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
