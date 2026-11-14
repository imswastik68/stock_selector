# Replay the proposed live rules on past data, using only month-by-month walk-forward predictions.
#  signal at close t (EOD alert after 15:30) -> buy at open t+1
#  top 5 by model prob among stocks with prior-20d turnover >= Rs 5 cr, price >= 20
#  first signal only: a stock is not re-alerted while its previous 21-day window is open
#  skip if the entry day is locked at upper circuit (opened at high, closed at high, gap >= 1.9%)
#  exit: limit sell at +27.54% if the high touches it within 21 bars, else close of bar 21
#  cost 0.30% round trip (0.60% sensitivity)
import sys, numpy as np, pandas as pd
S = '/tmp/claude-1000/-home-swastik-Downloads-Projects/b82e419c-da14-438d-a197-0eaa67d2362f/scratchpad/re/'
TOP, MIN_CR, HOLD, BAR, COST = 5, 5.0, 21, 27.54, 0.30
SLOTS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
P = pd.read_pickle(S + 'panel.pkl')[['date', 'ticker', 'to20', 'up1', 'up5', 'up21', 'end21', 'pnl21', 'mae21']]
R = pd.read_pickle(S + 'wf_monthly_pred.pkl').merge(P, on=['date', 'ticker'])
R['cr'] = 10 ** R.to20 / 1e7
E = R[R.cr >= MIN_CR].copy()
E['rk'] = E.groupby('date').p.rank(ascending=False, method='first')
cand = E[E.rk <= TOP].sort_values(['date', 'rk'])

# raw prices for circuit check + portfolio marking
need = set(cand.ticker)
O, H, L, C, V = {}, {}, {}, {}, {}
for t in need:
    d = pd.read_csv(f'cache/backtest_ohlcv/{t}.csv', index_col=0, parse_dates=True)
    O[t], H[t], L[t], C[t], V[t] = (pd.to_numeric(d[k], errors='coerce') for k in ('Open', 'High', 'Low', 'Close', 'Volume'))
O, H, L, C, V = (pd.DataFrame(x).sort_index() for x in (O, H, L, C, V))
good = (V > 0).sum(axis=1) >= 0.5 * (V > 0).sum(axis=1).median()     # same holiday-row drop as panel.py
O, H, L, C = (x[good] for x in (O, H, L, C))
cal = C.index
pos = {d: i for i, d in enumerate(cal)}

last, rows = {}, []
for r in cand.itertuples():
    i = pos.get(r.date)
    if i is None or i + HOLD >= len(cal): continue
    if r.ticker in last and i - last[r.ticker] <= HOLD: continue      # still inside previous alert's window
    last[r.ticker] = i
    e = i + 1; o, h, c, prev = O.iat[e, O.columns.get_loc(r.ticker)], H.iat[e, H.columns.get_loc(r.ticker)], C.iat[e, C.columns.get_loc(r.ticker)], C.iat[i, C.columns.get_loc(r.ticker)]
    locked = bool(o >= h * 0.999 and c >= h * 0.999 and o / prev - 1 >= 0.019)
    rows.append(dict(date=r.date, i=i, ticker=r.ticker, p=r.p, cr=r.cr, up1=r.up1, up5=r.up5, up21=r.up21,
                     pnl=r.pnl21, end=r.end21, mae=r.mae21, locked=locked, entry=o))
A = pd.DataFrame(rows)
T = A[~A.locked].copy()

def nw_t(x, lag):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x); u = x - x.mean(); g = u @ u / n
    for l in range(1, min(lag, n - 1) + 1): g += 2 * (1 - l / (lag + 1)) * (u[l:] @ u[:-l]) / n
    return x.mean() / np.sqrt(g / n)

base = E.groupby('date').agg(r_hit=('up21', 'mean'), r_pnl=('pnl21', 'mean'))
T = T.merge(base, left_on='date', right_index=True)
print(f"signals {A.date.min():%Y-%m-%d}..{A.date.max():%Y-%m-%d}: top-{TOP} candidates {len(cand)}, first-signal alerts {len(A)}, "
      f"locked at upper circuit {A.locked.sum()} ({A.locked.mean():.0%}) -> tradable alerts {len(T)} on {T.date.nunique()} days "
      f"({len(T) / E.date.nunique():.2f}/trading day)")
print(f"locked ones would have done: hit {A[A.locked].up21.mean():.1%}  P&L {A[A.locked].pnl.mean():+.2f}%")

def summary(g):
    pd_ = g.groupby('date').pnl.mean().sort_index()
    return pd.Series({'alerts': len(g), 'days': g.date.nunique(), 'hit_1m': g.up21.mean(), 'random_hit': g.r_hit.mean(),
                      'x_random': g.up21.mean() / g.r_hit.mean(), 'pnl': g.pnl.mean(), 'pnl_0.6cost': g.pnl.mean() - 0.30,
                      'random_pnl': g.r_pnl.mean(), 'median_pnl': g.pnl.median(), 'win%': (g.pnl > 0).mean(),
                      'NW_t': nw_t(pd_.values, HOLD), 'hit_1d': g.up1.mean(), 'hit_1w': g.up5.mean(),
                      'mae_median': g.mae.median(), 'mae_p10': g.mae.quantile(.1)})
T['yr'] = T.date.dt.year
tab = pd.concat({**{y: summary(g) for y, g in T.groupby('yr')}, 'ALL': summary(T)}, axis=1)
pd.set_option('display.width', 200)
fmt = tab.T.copy()
for c in ('hit_1m', 'random_hit', 'win%', 'hit_1d', 'hit_1w'): fmt[c] = (fmt[c] * 100).round(1)
print(fmt.round(2).to_string())

# scorecard verdict on the whole replay
v = summary(T)
ok = {'>=100 alerts on >=40 days': v.alerts >= 100 and v.days >= 40, '>=30% hit': v.hit_1m >= .30,
      '>=2x random': v.x_random >= 2, 'P&L > 0': v.pnl > 0}
print("VERDICT:", "PASS" if all(ok.values()) else "FAIL", ok)

# ---------- Rs 10L portfolio ----------
N = pd.read_csv('cache/backtest_nifty.csv', skiprows=[1, 2], index_col=0); N.index = pd.to_datetime(N.index)
nC = pd.to_numeric(N['Close'], errors='coerce')
cash, eq, open_, curve, skipped = 1_000_000.0, 1_000_000.0, [], [], 0
byday = T.groupby('i')
first, lastd = T.i.min() + 1, T.i.max() + HOLD   # last exit = close of bar i+21
for d in range(first, lastd + 1):
    # exits scheduled for today (limit touch or window end) -- handled by marking per position
    still = []
    for ps in open_:
        tk, ent, sh, e0 = ps
        j = C.columns.get_loc(tk); hi = H.iat[d, j]; cl = C.iat[d, j]
        tgt = ent * (1 + BAR / 100)
        if np.isfinite(hi) and hi >= tgt:
            cash += sh * tgt * (1 - COST / 200); continue
        if d - e0 >= HOLD - 1:
            px = cl if np.isfinite(cl) else ent
            cash += sh * px * (1 - COST / 200); continue
        still.append(ps)
    open_ = still
    # entries: alerts from signal day d-1 buy at today's open
    if d - 1 in byday.groups:
        for r in byday.get_group(d - 1).itertuples():
            size = eq / SLOTS
            if cash < size: skipped += 1; continue
            j = C.columns.get_loc(r.ticker); o = O.iat[d, j]
            if not np.isfinite(o) or o <= 0: continue
            sh = size * (1 - COST / 200) / o; cash -= size
            open_.append((r.ticker, o, sh, d))
            # same-day touch
            if H.iat[d, j] >= o * (1 + BAR / 100):
                cash += sh * o * (1 + BAR / 100) * (1 - COST / 200); open_.pop()
    mv = sum(sh * (C.iat[d, C.columns.get_loc(tk)] if np.isfinite(C.iat[d, C.columns.get_loc(tk)]) else ent) for tk, ent, sh, _ in open_)
    eq = cash + mv
    curve.append((cal[d], eq, len(open_)))
K = pd.DataFrame(curve, columns=['date', 'equity', 'positions']).set_index('date')
yrs = (K.index[-1] - K.index[0]).days / 365.25
dd = K.equity / K.equity.cummax() - 1
n = nC.reindex(K.index).ffill(); n = n / n.iloc[0] * 1e6
ndd = n / n.cummax() - 1
print(f"\nRs 10L portfolio {K.index[0]:%Y-%m-%d}..{K.index[-1]:%Y-%m-%d} ({yrs:.1f} yrs), {SLOTS} slots of equity/{SLOTS}, skipped for no cash: {skipped}")
print(f"  model : Rs {K.equity.iloc[-1] / 1e5:.1f}L  CAGR {(K.equity.iloc[-1] / 1e6) ** (1 / yrs) - 1:+.1%}  max drawdown {dd.min():.1%}  avg positions {K.positions.mean():.0f}")
print(f"  NIFTY : Rs {n.iloc[-1] / 1e5:.1f}L  CAGR {(n.iloc[-1] / 1e6) ** (1 / yrs) - 1:+.1%}  max drawdown {ndd.min():.1%}")
yr_ret = K.equity.resample('YE').last().pct_change(); yr_ret.iloc[0] = K.equity.resample('YE').last().iloc[0] / 1e6 - 1
nyr = n.resample('YE').last().pct_change(); nyr.iloc[0] = n.resample('YE').last().iloc[0] / 1e6 - 1
print(pd.DataFrame({'model': (yr_ret * 100).round(1), 'NIFTY': (nyr * 100).round(1)}).to_string())
mo = K.equity.resample('ME').last().pct_change()
print(f"  months up {(mo > 0).mean():.0%}, worst month {mo.min():.1%}, best month {mo.max():.1%}")
K.to_pickle(S + 'bt_live_curve.pkl'); T.to_pickle(S + 'bt_live_trades.pkl')

# ---------- benchmark: equal-weight basket of every >= Rs 5 cr stock (same survivorship bias as the model) ----------
U = pd.read_pickle(S + 'panel.pkl')[['date', 'ticker', 'to20', 'r1']]
R1 = U.pivot(index='date', columns='ticker', values='r1'); EL = U.assign(ok=10 ** U.to20 / 1e7 >= MIN_CR).pivot(index='date', columns='ticker', values='ok').fillna(False).astype(bool)
ew = R1.where(EL.shift(1, fill_value=False)).mean(axis=1)
end = min(K.index[-1], ew.index[-1])
w = ew.loc[K.index[0]:end].iloc[1:]
uni = (1 + w).cumprod() * 1e6; udd = uni / uni.cummax() - 1
y2 = (uni.index[-1] - K.index[0]).days / 365.25
print(f"\nsame window to {end:%Y-%m-%d} ({y2:.1f} yrs):")
print(f"  model         : Rs {K.equity.loc[end] / 1e5:.1f}L  CAGR {(K.equity.loc[end] / 1e6) ** (1 / y2) - 1:+.1%}")
print(f"  >=5cr EW basket: Rs {uni.iloc[-1] / 1e5:.1f}L  CAGR {(uni.iloc[-1] / 1e6) ** (1 / y2) - 1:+.1%}  max drawdown {udd.min():.1%}")
print(f"  NIFTY         : Rs {n.loc[end] / 1e5:.1f}L  CAGR {(n.loc[end] / 1e6) ** (1 / y2) - 1:+.1%}")
uy = uni.resample('YE').last(); uy = pd.concat([pd.Series([1e6], index=[K.index[0]]), uy]).pct_change().dropna()
print("  EW basket by year %:", (uy * 100).round(1).to_dict())
