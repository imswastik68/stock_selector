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
_(as of 2026-09-23, auto-generated)_

- **Last commit at log time:** f43baa4 2026-09-23.
- **Score IC:** 21 days recorded (needs 30), mean IC 0.0474, t=1.94, verdict **INSUFFICIENT**.
- **Momentum gate:** PAPER-ONLY -- no momentum strategy has passed the multi-split ship gate — PAPER-ONLY (outputs/factor_backtest.json)
- **Portfolio:** equity 101224.11, cash 11412.507370000005, 5 open holdings: PIRAMALFIN.NS, BHARATFORG.NS, KPIL.NS, NYKAA.NS, POLYCAB.NS.
<!-- AUTO-GENERATED:END -->

**Known unresolved (human-tracked, not auto-updated):**
- ~~Portfolio ₹1L vs ₹10L mismatch~~ **RESOLVED 2026-09-17** — user chose
  ₹10L. `scripts/reset_portfolio.py` archived the dead ~₹1L book to
  `outputs/portfolio_archive/portfolio_2026-09-17.json` (5 June holdings
  flattened at last mark into `closed` with outcome `"reset"`, 11 closed
  trades retained) and wrote a flat ₹10,00,000 book. Sizing now fits:
  ₹155k-248k notional per pick against ₹1M cash, and the 6% portfolio risk
  budget caps it at ~6 concurrent positions. The book will actually open
  trades again — it had opened **zero since 2026-07-17** while 57 buys were
  emitted in September, because `open_positions` skipped every pick on
  `state["cash"] < notional`.
- Repo was briefly private (~2026-08-10 to ~2026-08-25), which silently
  hit a GitHub Actions billing block (jobs failed in 3-5s, "recent account
  payments have failed"). Fixed by making the repo public. Not a code bug
  — keep the repo public.

## Shipped 2026-09-17: REGIME_WEIGHTS pruned (15 of 37 entries were wrong-signed)
`REGIME_WEIGHTS` re-weights signals by NIFTY trend. Unlike SHORT_TERM_WEIGHTS
it was **never validated** when generated (2026-07 Phase 5) — and 15 entries
had a sign the backtest contradicts. Worst: `ranging` set
**rsi_momentum to −2**, while rsi_momentum is the ONLY signal in scorer.py
sign-consistently positive across the 70/30 holdout at every horizon
(fwd_10d train +1.02 t=9.09 / holdout +0.60 t=2.55). Ranging is ~22% of the
sample, so for a fifth of its life the scorer penalised its best predictor.
`uptrend` likewise rewarded actual_52w_breakout (+1) and rs_vs_nifty (+1)
whose train lifts are −0.61 and −0.76.

Rule (`scripts/regime_override_audit.py`, re-runnable): drop the entry when
sign(TRAIN lift) ≠ sign(override); leave entries with <200 train obs ALONE
rather than guess. Holdout per-day rank IC **0.0241 → 0.0356 (+48%)**,
paired t=+1.70, positive in **12 of 12** split-point × horizon cells,
improving monotonically with train size. No single cell reaches p<0.05
(best t=+1.97) — what justifies it is the consistency plus the fact that the
dropped entries never had evidence behind them. Pinned by
`tests/test_regime_overrides_match_evidence.py` (17/30 fail against the old
table).

**Do NOT delete this table wholesale.** An earlier pass this session compared
all-overrides vs no-overrides and appeared to show deletion helped — that was
an artifact of a reconstruction that omitted the bearish penalties
(distribution_signal, heavy_selling, volume_5x). Those are the CORRECT part
and carry most of the table's value; with them included, wholesale deletion
measures as a wash (t=−0.50/−0.00/+0.60).

**Also beware:** several scorer.py weight comments are Simpson's-paradox
artifacts. `near_52w_high` is commented "BEST validated signal: +1.01 ret
lift" and weighted +3 — its **pooled** return_pct lift is +0.138 but its
**within-year** lift is −0.230. The signal fires more in bull years, so a
pooled signal-on-vs-signal-off comparison credits it with the year. Same for
`actual_52w_breakout` (+3, within-year −0.680) and `rsi_bearish_div` (+3,
within-year −0.249). Always control for period before believing a lift.
Rebalancing the base weights on this was tested and did **not** beat the
existing weights out of sample, so the base table was left alone — the
finding is about the comments being wrong, not a shippable reweight.

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
  binding constraint. NB: any pre-2026-09-17 claim that "most losers were
  deeply green first" came from the MAE/MFE window bug (fixed 59bc3fa)
  and was an artifact of post-exit bars — do not revive it.
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

## Fixed 2026-09-17: the alert advised a 1-2 day hold, the worst horizon
`src/scorer.py` hardcoded `timeframe = "1-2d"` and **66 of 70 live buy picks
carried it**, while `WINNER_POLICY="time_10d"` force-exits at 10 bars. The
alert advised a hold the system never performs — and per the table above,
1-2 days is the one horizon that loses money net of costs.

`timeframe` is display-only (only `src/telegram_alert.py` reads it), so this
mis-instructed the reader rather than breaking a trade, which is why it
survived. Now derived from the exit policy via
`src.trade_sim.horizon_label()`, which also backs the defaults in
`response_parser`, `agent` and the LLM prompt schema — **four** places
independently told the user a holding period and three were short. Pinned by
`tests/test_timeframe_matches_exit_policy.py` (mutation-checked).

## Open lead 2026-09-17: the 10-day time stop may be cutting winners early
On the SAME `score>=4` trades (paired, so this is not a cohort difference),
holding 20 bars instead of 10:

| split | excess@10d vs NIFTY | excess@20d | diff | paired t |
|---|---|---|---|---|
| train | +0.70% | +1.59% | **+0.89pp** | +12.12 |
| holdout | +0.85% | +1.87% | **+1.03pp** | +13.71 |

p<1e-4 both sides, sign-consistent, and **not beta** — NIFTY itself returned
+0.06%/+0.31% over those windows, and the NIFTY-excess version is *stronger*
than the raw one. This is the largest clean effect found this session.

**Caveat that stops it being shipped on this number alone:** `fwd_20d` is
exit-agnostic buy-and-hold. The live book exits on SL/T1 first, so a longer
time cap only affects trades still open at day 10 — stops and the time cap
interact. `time_15d`/`time_20d` were added to `src/trade_sim.py`
(`_TIME_STOP_BARS`) and `scripts/backtest_exits.py`'s POLICIES so the real
comparison can be run; `WINNER_POLICY` is **unchanged at `time_10d`** until
that backtest says otherwise. This is also the one legitimate reason to
reopen the "exits" question, which the note below otherwise forbids.

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

## Why the system is long-only (asked 2026-09-17, re-verified)
`SHORT_PIPELINE_LIVE = False` in `src/agent.py:115`. Sells are scored and
routed to phase_b as informational only. This is evidence-backed, not an
oversight — re-measured on `outputs/backtest_trades.csv` (291,314 trades,
2020-11 to 2026-06, of which 79,972 are sell-direction):

| cohort (fwd_10d, direction-adjusted) | n | mean | t |
|---|---|---|---|
| all BUY rows | 203,162 | **+1.22%** | +23.6 |
| all SELL rows | 79,972 | **−1.64%** | **−39.6** |
| short a 52w breakdown | 4,976 | −5.49% | −25.2 |
| short a heavy-selling name | 5,470 | −4.91% | −24.0 |
| short near 52w low | 10,886 | −3.34% | −26.7 |
| short a distribution name | 6,000 | −3.17% | −18.7 |

Every bearish signal loses money on the short side, and the effect is
huge and stable (t between −17 and −40). **Mechanism:** un-flipping the
sign shows breakdown names *rise* +5.49% over the next 10 days (t=+25.2,
59.7% up) — Indian small/midcaps that break to 52-week lows mean-revert
violently, so shorting them fights a strong bounce.

Structural constraint on top of that: India has no overnight short in the
cash segment. A held short needs stock futures (~190 F&O names) or long
puts, so even a real edge would only be executable on the F&O subset.

**Trap for future sessions:** `fwd_5d/10d/20d` in `backtest_trades.csv`
and in `src/trade_sim.py` are ALREADY direction-adjusted (positive =
the trade profited, for both buys and sells). Reading them as raw price
moves makes the short book look like a huge *winner* when it is the
opposite. Split by `direction` before interpreting.

## What this system can and cannot do (measured 2026-09-17)
Asked for "sure shot" 3-day / 1-week / 1-month calls. **That does not exist
here and the numbers say so plainly.** Best measured directional accuracy of
any cohort is **52.3% up**. Everything below is holdout-only (2024-08 to
2026-06, n=210,072 buy rows), net of the repo's 0.30% round-trip cost, cut by
the live actionable-buy bar (`score >= 4`, `src/agent.py:346`):

| horizon | net/trade @score≥4 | up% | verdict |
|---|---|---|---|
| **fwd_5d (3d-1wk)** | **−0.09%** | 47% | **no edge — cost eats it** |
| fwd_10d (2 wks) | +0.82% | 50.6% | real, thin |
| **fwd_20d (~1 month)** | **+1.88%** | 52% | **best horizon** |

Two things follow, and they are the opposite of the intuition:
1. **The short end is the weak end.** 3-day/1-week calls are net-negative at
   every score threshold below 6. Do not market or trade this as a 3-day
   system.
2. **Positional (~1 month) is where the edge lives**, and it is monotone in
   score (`>=2` +0.25% → `>=4` +0.82% → `>=6` +1.06% at 10d, all
   sign-consistent train+holdout). The `score >= 4` emission bar is already
   correctly placed — do NOT raise `MIN_SCORE` (2) to match it; that is a
   Pass-1 pre-enrichment funnel filter, and raising it would drop candidates
   before options/SAST enrichment can lift their score.

Expressed honestly: at its best configuration this is a ~52% win rate with a
~+1.9% net edge per one-month trade. That is a real edge and it is worth
having. It is not a guarantee about any individual stock, and no amount of
further work will make it one.

## Known reporting defects
- ~~`live_alpha_gate` over-counts evidence~~ **FIXED 2026-09-17.** The 5
  ✅ PROVEN signals were **71 distinct trades reported as n=312** (4.39x),
  actual_52w_breakout/near_52w_high overlapping 99%. Attribution is still
  one-pick-to-all-its-signals (correct for measuring a signal), but
  `live_alpha_gate` now returns an `attribution` block and
  `live_proof_report` emits a caveat line whenever inflation ≥1.5x:
  `⚠️ NOT 5 independent proofs: n=312 ... is only 71 distinct trades`.
  Pinned by `tests/test_live_proof_attribution.py` (mutation-checked).
  Still true and unfixed: AGGREGATE says NO-EDGE partly because it also
  includes 57 older picks whose `active_signals` was never recorded.
- ~~`score` is never recorded on live picks~~ **FIXED 2026-09-17.**
  `record_picks` now stores `score`. All 285 pre-fix picks keep
  `score: None` permanently — live score→outcome attribution only becomes
  possible for picks recorded from here on. Pinned by
  `tests/test_record_picks_stores_score.py` (incl. a score-of-0 case, since
  `entry.get("score") or None` would silently lose it).
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
