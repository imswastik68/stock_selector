import itertools, numpy as np, pandas as pd
S = '/tmp/claude-1000/-home-swastik-Downloads-Projects/b82e419c-da14-438d-a197-0eaa67d2362f/scratchpad/re/'
P = pd.read_pickle(S + 'panel.pkl'); cut = pd.Timestamp('2025-01-08')
P['breadth50'] = pd.to_numeric(P.breadth50, errors='coerce'); TO = 10 ** P.to20 / 1e7
def nw_t(x, lag):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x); u = x - x.mean(); g = u @ u / n
    for l in range(1, min(lag, n - 1) + 1): g += 2 * (1 - l / (lag + 1)) * (u[l:] @ u[:-l]) / n
    return x.mean() / np.sqrt(g / n)
weak = P.breadth50 < .35
def run(r250_min, vr5_min, near_hi_strong, near_hi_weak, to_min, rank_by):
    m = ((P.r250 >= r250_min) & (P.r20 > 0) & P.atr14.between(.035, .09) & (P.vr5 >= vr5_min) & (TO >= to_min)
         & np.where(weak, P.dist_52wh >= -near_hi_weak, P.dist_52wh >= -near_hi_strong))
    x = P[m].copy(); x['rk'] = x.groupby('date')[rank_by].rank(ascending=False); return x[x.rk <= 5]
grid = list(itertools.product([0.5, 0.8, 1.0], [1.0, 1.2, 1.5], [0.10, 0.20], [0.25, 0.35], [2, 5], ['r250', 'vr5']))
res = []
for g in grid:
    x = run(*g); tr = x[x.date < cut]
    if tr.date.nunique() < 300: continue
    res.append((g, tr.up21.mean(), tr.pnl21.mean(), nw_t(tr.groupby('date').pnl21.mean().values, 21), tr.date.nunique()))
R = pd.DataFrame(res, columns=['g', 'hit', 'pnl', 't', 'days']).sort_values('t', ascending=False)
print("best 5 on TRAIN (ranked by date-clustered t of P&L):"); print(R.head(5).to_string(index=False))
best = R.iloc[0].g
x = run(*best); ho = x[x.date >= cut]; tr = x[x.date < cut]
print(f"\nCHOSEN on train: r250>={best[0]}, vr5>={best[1]}, within {best[2]:.0%} of 52wH (weak mkt: {best[3]:.0%}), turnover>=Rs{best[4]}cr, rank by {best[5]}")
for lab, d in (('train', tr), ('HOLDOUT', ho)):
    q = d.groupby('date').pnl21.mean()
    print(f"{lab:<8} days {d.date.nunique():>4}  picks/day {len(d)/d.date.nunique():.1f}  hit(1m bar) {d.up21.mean():.1%}  hit(1w bar) {d.up5.mean():.1%}  P&L/pick {d.pnl21.mean():+.2f}%  NW t {nw_t(q.values, 21):+.2f}  worst-1m-loss median {d.mae21.median():+.1f}%")
print("holdout by quarter:"); print(ho.assign(q=ho.date.dt.to_period('Q')).groupby('q').agg(n=('up21', 'size'), hit=('up21', 'mean'), pnl=('pnl21', 'mean')).round(3).to_string())
