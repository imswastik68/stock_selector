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

## THE GOAL (adopted 2026-11-11) — supersedes "alpha vs NIFTY"
Telegram alerts that pick stocks about to become **top-50 movers**, in
stocks the user can actually trade; the user places trades manually.
Same bar as stock-news-alerts. Scored by `scripts/top50_scorecard.py`
(BSE bhavcopy, method in its docstring, pinned by
`tests/test_top50_scorecard.py`):

| | 1 day | 1 week (5 bars) | 1 month (21 bars) |
|---|---|---|---|
| bullish | +5.71% | +12.63% | +27.54% |
| bearish (F&O only) | −4.44% | −9.65% | −19.71% |

Entry by send time: before 09:15 → that day's open; market hours → that
day's close; after 15:30/holiday → next open. Tradable = BSE group
A/B/X, ≥ ₹20, ≥ ₹1 cr/day. **PASS** a horizon only with ≥100 tradable
alerts on ≥40 dates AND touched ≥30% AND ≥2× a random tradable stock on
the same days AND P&L/alert > 0 (limit at bar, else window close, 0.30%).

**Baseline 2026-11-11 (453 alerts, May→Nov): FAIL at every horizon.**
Tradable bullish touched 8.2% / 6.8% / 6.9% (1d/1w/1m) vs random
5.2% / 6.3% / 5.5% → 1.1-1.6× random, P&L/alert −0.44% / −0.62% / +1.01%.
19 of 280 one-week alerts touched +12.63%. Only 3 tradable F&O shorts
ever sent. Score 15+ touched 13.2% at 1w (n=38) but still lost money —
volatility, not skill. Need ~4× more hits to reach 30%.

## BACKTEST 2026-11-12: 1-month model replayed with live rules
Monthly walk-forward (44 LightGBM fits, each trained only on labels matured
37+ days before the month), signals 2023-01 → 2026-08. Rules as proposed for
live: EOD signal → buy next open; top 5/day among ≥₹5cr/day, ≥₹20; each
stock once per 21-bar window; skip entry days locked at upper circuit (4%);
limit sell at +27.54% else close of bar 21; 0.30% cost. Scratch scripts
(not committed): wf_monthly.py, bt_live.py.

| | 2023 | 2024 | 2025 | 2026 (to Aug) | ALL |
|---|---|---|---|---|---|
| alerts (days) | 271 (171) | 302 (176) | 256 (163) | 178 (115) | 1007 (625) |
| hit 1-month bar | 20.3% | 21.9% | 18.0% | 27.5% | **21.4%** |
| random ≥5cr stock | 6.7% | 8.8% | 4.6% | 8.2% | 7.1% |
| P&L/trade | +3.84% | +2.76% | +0.41% | +4.33% | **+2.73%** |
| beyond random (NW t) | +0.86 (1.0) | +1.27 (1.2) | +1.32 (1.8) | +3.27 (1.8) | **+1.53 (2.75)** |

Median trade +0.43%, win 52%; median in-trade drawdown −9.5%, worst 10%
< −24.6%. At 0.60% cost: +2.43%/trade. **Verdict: FAIL** (hit 21.4% < 30%);
passes sample size, ≥2x random (3.0x), P&L > 0.
**₹10L portfolio** (equity/30 per trade, ~19 open): ₹21.8L by 2026-09-16,
CAGR +23.5%, max DD −25.7%; years +31.8 / +33.0 / +1.4 / +23.0%. To
2026-08-17: model ₹21.3L vs **equal-weight basket of all ≥₹5cr stocks ₹19.8L**
(CAGR 20.7%, DD −25.2%) vs NIFTY ₹13.3L. 20 slots: ₹24.0L, DD −31%.
Most of the money is small-cap beta; the model's own edge is ~+1.5%/trade.
Survivor-only cache inflates model AND basket alike. No T2T/Z filter before
Nov 2025 (no BSE group data).
**Data bug found:** yfinance writes exchange-holiday rows (e.g. 2026-05-01,
05-28, 06-26) with no volume; any 20-day volume window touching one goes NaN
→ the 11-11 research panel silently lost May–Jul 2026 (1.53M → 1.67M rows
after dropping them). Unchecked: whether scripts/backtest.py /
factor_backtest.py, which read the same cache, have the same hole.

## RESEARCH 2026-11-11: reverse-engineering top-50 movers
Panel: 1.53M tradable stock-days (2021-01→2026-08, ₹20+, ₹1cr+/day), ~25
price/volume features, labels = touched the top-50 bar from next open.
Scripts were scratch-only (not committed). Random touch rate 7.7% (1w).

**Single traits (vol-controlled, train vs holdout):** volatility dominates
(top ATR decile 13-15% vs 1-2%) but loses money; volume spike today
(x1.2 both halves), small turnover/low price (x1.25-1.3); momentum traits
(r20/r60/MA/RSI) work in train and vanish in holdout. No single-trait
decile makes money.

**LightGBM, yearly walk-forward (retrain on all prior years, purged),
top-5/day:**

| 1-month bar (+27.54%) | 2023 | 2024 | 2025 | 2026 (82d) |
|---|---|---|---|---|
| hit (random) | 19.5% (8.1%) | 23.0% (8.7%) | 23.3% (4.6%) | 22.2% (9.7%) |
| P&L/pick (random) | +3.90% (+3.29) | +2.20% (+0.71) | +3.04% (−1.43) | +2.81% (+1.99) |

Positive every year, beats random every year. 1-week model hits 21-30%
(3-4x random) but P&L ≈ 0 — misses eat the hits. Verified on official
BSE prices: 97% label agreement, same P&L.
Picks = **year's leaders under accumulation**: 12m +85-140%, r20 > 0,
ATR 5-6%, vr5 1.25-1.6x, small (₹3-5cr/day, ~₹150). Weak market: buys
leaders ~19% off highs (RSI ~55); normal/strong: near highs (RSI ~72).
**A transparent rule distilled from this FAILED holdout** (train t=3.06,
holdout 11.7% hit, −0.46%) — the edge lives in interactions.

**Fundamentals (Nov 2025→Aug 2026 only, point-in-time results):** low
institutional ownership is the strongest (12.1% vs 3.8%), small cap,
revenue growth mildly +; profit growth is U-shaped (volatility, not
direction); loss-makers hit MORE; no fundamental filter reliably improves
the model's picks on this 9-month sample.

**Against the goal: still FAIL on hit rate** (19.5-23% < 30%); passes
≥2x random and P&L>0 at 1 month. Caveats: 15% of entries open locked at
upper circuit; 76% of picks repeat a held name — first-signal-only is
19.5% hit / +1.60%; median intra-month drawdown −10%, worst 10% < −24%;
survivor-only price cache; ₹3-5cr/day names → slippage > 0.30%.

## READ FIRST 2026-11-11: live is losing; most backtest "alpha" is size beta
**Paper book was dead for 2 months (fixed).** The 2026-09-17 ₹10L reset
(8b9e200) was reverted 23 min later: `actions/cache` restores `outputs/`
over the checkout, so CI committed the stale cached ₹1L June book back and
kept it (5 June holdings, ₹11k cash, zero new trades). Workflow now
prefers git when a tracked output changed since the cache was saved;
reset re-applied (old book → `portfolio_archive/portfolio_2026-11-11.json`).

**Live Telegram buys (427 picks, 95 alert dates, May→Nov 5)**, entry at
next open, net 0.30%, clustered by alert date, NW lag = horizon:

| | 10d vs NIFTY | 20d vs NIFTY | 20d vs random universe name |
|---|---|---|---|
| before 2026-09-18 (64 dates) | −0.18% | +0.92% (t=+1.09) | −0.33% (t=+0.35) |
| since 2026-09-18 (16-26 dates) | −6.44% | **−13.89%** | **−4.58%** |

As-advertised trades since 9/18 (zone/SL/T1/10d): 38 SL, 10 T1, 13
timeout, mean **−4.45%**. Post-9/18 dates overlap heavily (~2 independent
20d windows) — treat as one bad episode, not 16 failures. ~2/3 of it is
size: random universe names fell −5.4% while NIFTY rose +3.9% (V-bottom
mid-Oct). The rest is selection: DOWNTREND REGIME_WEIGHTS pushed scores
to 14-19 on small-cap momentum names that kept falling; SMCGLOBAL was
alerted 5x in 3 weeks. Regime label itself is correct by its EMA rule.
Before 9/18 the picks were **indistinguishable from a random universe
name**.

**Daily backtest (1,401 dates, 2021-01→2026-09) settles two questions**
(score>=4 buys, gross, 70/30 cut 2025-01-08, NW t):

| 20d | train | holdout | full |
|---|---|---|---|
| admitted − NIFTY | +2.29% (4.51) | +0.54% (0.88) | +1.79% (4.21) |
| universe EW − NIFTY | +1.72% (3.25) | −0.24% | +1.16% (2.48) |
| admitted − universe EW | +0.57% (2.72) | +0.78% (1.92) | +0.63% (3.30) |
| admitted − same-day pool | +0.15% (0.99) | +0.70% (3.27) | +0.31% (2.38) |
| **top-5 by score − admitted** | −0.30% (−1.52) | −0.26% (−0.59) | −0.29% (−1.51) |

1. **~2/3 of "alpha vs NIFTY" is the small-cap universe beating NIFTY**,
   not selection. Selection edge vs the universe is ~+0.6% gross (~+0.3%
   net) per 20d trade. Measure future alpha vs universe EW, not NIFTY.
   (Universe is survivor-only; adm−universe also mixes limit-zone entry vs
   next-open, so it is an upper bound. adm−pool is the clean comparison.)
2. **Score does not rank above the bar.** Top-5 by score is slightly
   *worse* than the average admitted name in both halves (n.s.); within-
   admitted rank IC ≈ 0. Taking the 5 highest scores adds nothing.
   Not shipped: no alternative selection has been tested.

## READ FIRST 2026-09-17: every t-stat in this repo is inflated ~10x
`backtest_trades.csv` is **weekly-sampled**: 280 scan dates, ~750 trades
per date. Trades sharing a scan date are massively cross-correlated (same
market, same day), so a per-trade t-test overstates significance by
roughly √(trades-per-date). **The unit of independence is the scan date,
not the trade.**

Long side, `score>=4`, NIFTY-excess fwd_20d, the same numbers three ways:

| | mean | naive t | date-clustered t | Newey-West(3) t |
|---|---|---|---|---|
| train (219 dates) | +0.945% | +10.07 | +3.79 | **+2.89** |
| holdout (**55 dates**) | +2.472% | +19.06 | +1.62 | **+1.49 (n.s.)** |
| full (274 dates) | +1.374% | +17.90 | +4.12 | **+3.24** |

Newey-West lag 3 because a 20-day horizon overlaps ~3 weekly scans.

**What this means, stated honestly:** the long edge is *sign-consistent*
(positive in both halves) and *full-sample significant* (t=+3.24), but it
is **not established on the holdout alone** (p=0.11). Only **50.9%** of
holdout scan dates have positive mean excess — a coin flip. The holdout
is 55 observations, not 10,743.

**Rules that follow:**
- Never quote a per-trade t-stat from this file again. Cluster by
  `as_of`, then Newey-West for horizon overlap.
- A four-figure `n` in any older table in this document is a trade count,
  not a sample size. Historic claims here carrying t>10 (e.g. "+31.96",
  "+13.71", "+14.52") are inflated by this bug and should be re-derived
  before being relied on.
- Resolving anything about *ranking* needs a denser (daily) backtest.
  55 dates cannot settle it.

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
_(as of 2026-11-10, auto-generated)_

- **Last commit at log time:** d9a00e0 2026-11-10.
- **Score IC:** 55 days recorded (needs 30), mean IC 0.0159, t=1.05, verdict **WEAK**.
- **Momentum gate:** PAPER-ONLY -- no momentum strategy has passed the multi-split ship gate — PAPER-ONLY (outputs/factor_backtest.json)
- **Portfolio:** equity 104246.91, cash 11412.507370000005, 5 open holdings: PIRAMALFIN.NS, BHARATFORG.NS, KPIL.NS, NYKAA.NS, POLYCAB.NS.
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

## RESOLVED 2026-09-17: the 10-day time stop is CORRECT — keep `time_10d`
The lead below said a 20-bar hold beats 10. **It does not survive contact
with the actual exit rules.** `scripts/backtest_exits.py` re-simulated all
280,799 trades across 8 policies (~35 min). OOS from 2024-10-22:

| policy | OOS expectancy | OOS WR% |
|---|---|---|
| **time_10d** | **−0.814%** | 43.1% |
| time_15d | −0.916% | 41.9% |
| time_20d | −0.979% | 40.9% |
| static | −1.035% | 38.8% |
| partial_t1_be | −1.199% | 38.8% |
| breakeven_1r | −1.426% | 29.6% |
| partial_t1_trail | −1.659% | 31.8% |
| trail_atr3 | −1.874% | 30.4% |

`time_10d` wins and the ordering is **monotone the wrong way** for the
lead: 10 > 15 > 20. **Mechanism:** `fwd_20d` is exit-agnostic
buy-and-hold, so it rewards holding through drawdowns that the live book
would have stopped out. Once SL/T1 are live, a longer cap just means more
trades reach their stop. `WINNER_POLICY` stays **`time_10d`** — this is
exactly the caveat below, and it was the right call not to ship on the
paired number.

**Note the levels are all negative** because this run filters by neither
score nor direction: it pools sub-threshold trades and the short book
(which carries −2.12% alpha, see above). It ranks policies correctly but
does not estimate the live system's expectancy.

**The cohort-restricted re-run (buy + `score>=4`) does NOT confirm this
and should not be read as overturning it either** — it is a weaker test,
not a better one:

| policy | OOS exp | OOS WR% | delta vs time_10d | clustered t | p |
|---|---|---|---|---|---|
| static | +1.455% | 50.2% | +0.66pp | +1.73 | 0.098 |
| **time_10d** | +0.799% | 48.4% | — | — | — |
| time_15d | +1.016% | 48.9% | +0.22pp | +1.01 | 0.32 |
| time_20d | +1.159% | 49.2% | +0.36pp | +1.13 | 0.27 |
| partial_t1_be | +1.577% | 50.2% | +0.78pp | +2.50 | **0.020** |

n=41,857 trades but only **24 OOS scan dates**. Filtering to `score>=4`
pushes the 70/30 cut to 2025-12-30, so this OOS window is entirely
**Jan-Jun 2026** — exactly the flood period documented above where
`score>=4` admits 94.5% of candidates in the prevailing downtrend regime.
It is one regime, not a mix, and 24 dates is a third of the 84-date OOS
window the full-pool result above was measured on. Only `partial_t1_be`
clears p<0.05, and barely.

**Verdict: `WINNER_POLICY` stays `time_10d`.** This is a real lead
(`partial_t1_be` — scale out at T1, move stop to breakeven, let the rest
run — deserves a proper look) but not evidence strong enough to act on:
single regime, low date count, borderline significance. Needs the daily
backtest to get real power before changing anything.

**Do not re-open the "longer time stop" question without a new
mechanism** — that specific idea is dead on both runs. The
`partial_t1_be` lead is a different, open question.

## Original lead (kept for the reasoning) — the 10-day stop may cut winners early
On the SAME `score>=4` trades (paired, so this is not a cohort difference),
holding 20 bars instead of 10:

| split | excess@10d vs NIFTY | excess@20d | diff | paired t (naive) |
|---|---|---|---|---|
| train | +0.70% | +1.59% | **+0.89pp** | +12.12 |
| holdout | +0.85% | +1.87% | **+1.03pp** | +13.71 |

Re-derived with date-clustered inference (the naive t above is inflated,
see the note at the top of this file) the effect **does** hold up on its
own terms — train +0.88pp (Newey-West t=+3.17), holdout +1.23pp (NW
t=+2.22, p=0.023), positive on 58.4% of scan dates. It survives
clustering because it is *paired* on identical trades, so the
market-wide move cancels and the date-to-date variance is small.

It is nonetheless **not** an argument for a longer time stop — see the
resolution above. Both things are true: holding longer helps if you never
stop out, and hurts once you do.

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
`SHORT_PIPELINE_LIVE = False` in `src/agent.py:116`. Sells are scored and
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

### The short cohort is beta, not inverted alpha (measured 2026-09-17)
The mean-reversion above is tempting to trade the other way — "if the
sell list rises, go long the sell list." **It does not work: the sell
cohort is simply high-beta.** OLS of direction-adjusted net trade P&L on
the matching NIFTY move, fwd_20d, sell cost 0.15% futures per
`src/costs.py`:

| cohort (fwd_20d) | alpha | t (naive, INFLATED) | beta |
|---|---|---|---|
| BUY score≥4, train | **+1.93%** | +16.7 | 0.67 |
| BUY score≥4, holdout | **+1.77%** | +15.4 | 0.64 |
| SELL score≥4, train | −1.10% | −1.9 | 1.30 |
| SELL score≥4, holdout | **−2.12%** | −5.1 | 1.19 |

⚠️ The t column is per-trade and therefore **inflated ~10x** — see
"every t-stat in this repo is inflated" at the top. Date-clustered, the
long side is t=+2.89 train / **+1.49 holdout (not significant)**. The
alpha *point estimates* and the beta estimates stand; the significance
does not. The short side's negative sign is what matters there, and it
is negative in both halves at any inference standard.

The short book carries beta ~1.2-1.3 and *negative* alpha on top of it,
so going long the sell list is a leveraged index bet, not an edge. The
long book carries beta **0.64** with ~+1.8% alpha — less market exposure
than the index and a real excess on top. Sign-consistent across the
split.

Because beta is 0.64 and not 1.0, a plain NIFTY-excess subtraction
(which implicitly assumes beta=1) **understates** the long alpha. Use
the regression.

**Method warning:** an earlier version of this test split the cohorts by
NIFTY's *forward* return ("NIFTY up" vs "NIFTY down" buckets). That
conditions on the future and is not a valid alpha estimate, though it is
fine as a beta diagnostic. The regression above replaces it. Results are
robust to entry misalignment (fills land 1-2 bars after `as_of`; lags of
0/1/2 days move the buy figure only +1.40/+1.37/+1.34%).

Pooled NIFTY-excess at the live bar, both splits sign-consistent:

| horizon | train | holdout |
|---|---|---|
| fwd_5d | −0.05% (p=0.36) | +0.07% (p=0.25) — **no alpha** |
| fwd_10d | +0.35% (t=+4.8) | +0.69% (t=+8.8) |
| fwd_20d | +1.21% (t=+11.8) | +1.73% (t=+15.4) — **best** |

Note the 5d row: the earlier raw-return table showed fwd_5d as merely
weak; with beta removed it is **zero in both splits**. The 3-day/1-week
horizon has no alpha at all, not just a thin one. This matches the
published record — one-month cross-sectional momentum is a documented
effect, while sub-weekly horizons are where costs and reversal dominate
(Jegadeesh-Titman line of work; "short-term momentum (almost)
everywhere", J. Int. Financial Markets 2019).

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

## 2026-09-17: the score is a proven FILTER; whether it RANKS is undetermined
Asked to surface "max expectancy" picks. Measured on beta-adjusted fwd_20d
alpha, buy rows, 70/30 split.

**The bar itself.** `score>=4` earns alpha=**+1.93%** train / **+1.77%**
holdout at beta 0.64 — sign-consistent, and full-sample significant once
clustered by date (t=+3.24). On the holdout alone it is **not**
significant (clustered t=+1.49). Best current read: a likely edge of
roughly +1 to +2% per one-month trade, not a settled one.

**Undetermined — the ordering.** Correctly measured (per-day paired,
top-6 vs 6 drawn at random from the same day's eligible pool, 400 draws):

| split | top-6 by score | random-6 | random − top | p |
|---|---|---|---|---|
| train | +1.168% | +1.670% | +0.502pp | 0.24 |
| holdout | +0.874% | +0.594% | −0.281pp | 0.61 |

Sign-flips across the split and neither is significant → **no evidence
either way.** Do not conclude that score-ranking helps, and do not
conclude it hurts.

**A pooled test here is misleading and I ran one first.** Pooling all
top-5 rows across dates gave "top-5 = −0.21% vs pool +1.57%", which
looks like the push is actively picking the worst names. That comparison
mixes dates with very different market conditions; the per-day paired
version above is the correct test and it shows no such effect. Earlier
Spearman/rank-IC numbers (train −0.041, holdout +0.039) have the same
pooling flaw.

**Power limit — read this before running another ranking test.**
`backtest_trades.csv` is sampled **weekly**, not daily: 280 scan dates
over 5.7 years, of which only ~253 have ≥6 eligible names and **75 land
in the holdout**. Any per-day ranking claim is therefore low-powered by
construction. Getting a real answer needs a denser backtest, not another
pass over this file.

**Consequence:** there is currently no evidence-backed reason to change
the top-5 Telegram selection. The value is in the filter. Leave the
selection alone until a daily-sampled backtest can actually resolve it.

### The edge is concentrated in 3 of 7 years
Alpha of the admitted cohort (fwd_20d, per-trade t shown — inflated, use
for sign only):

| yr | alpha | t | dominant regime |
|---|---|---|---|
| 2020 | **−6.12** | −4.07 | uptrend |
| 2021 | +4.40 | +15.73 | uptrend |
| 2022 | +0.06 | +0.28 | downtrend |
| 2023 | +4.49 | +20.10 | uptrend |
| 2024 | −0.33 | −1.48 | uptrend |
| 2025 | −0.18 | −0.77 | downtrend |
| 2026 | +2.99 | +20.56 | downtrend |

2021, 2023 and 2026 carry the whole result; 2020 is strongly negative
and 2022/2024/2025 are flat. This is the real reason the holdout fails
to reach significance — it is not a power artifact alone, the effect
genuinely switches off for years at a time.

**The 70/30 split is nearly a regime split**, which confounds every
train-vs-holdout comparison in this document: train is mostly uptrend
(14,632 admitted rows) and holdout mostly downtrend (5,060). Splitting
*within* each regime on its own timeline:

| regime | train alpha | holdout alpha | |
|---|---|---|---|
| uptrend | +2.81% | **−0.75%** | SIGN FLIP |
| ranging | +1.66% | −0.78% | SIGN FLIP |
| downtrend | +2.27% | +2.01% | consistent |

Only the downtrend cohort holds up. Note this is the cohort the scorer
floods (94.5% admit) — so the regime override that looks like a bug by
admit-rate is, by alpha, the part that works.

### The score's scale is not stable across regimes
Monthly mean score on buy rows swings from **−2.96** (2026-02) to
**+9.12** (2026-04) with no matching change in any `sig_*` firing rate —
the swing comes from `REGIME_WEIGHTS`, which reprices every signal when
the NIFTY regime flips. So the fixed `score>=4` bar admits a wildly
varying share of candidates:

    2024Q4: 4%   2025Q1: 46%   2025Q3: 6%   2026Q2: 95%

A 95%-admit quarter is not a filter. This is the likely mechanism behind
the ranking instability above, and it means `score>=4` does not mean the
same thing in two different months. Unresolved.

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
