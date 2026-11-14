import numpy as np, pandas as pd, glob
S = '/tmp/claude-1000/-home-swastik-Downloads-Projects/b82e419c-da14-438d-a197-0eaa67d2362f/scratchpad/re/'
P = pd.read_pickle(S + 'panel.pkl'); R = pd.read_pickle(S + 'gbm_holdout.pkl')
def nw_t(x, lag):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x); u = x - x.mean(); g = u @ u / n
    for l in range(1, min(lag, n - 1) + 1): g += 2 * (1 - l / (lag + 1)) * (u[l:] @ u[:-l]) / n
    return x.mean() / np.sqrt(g / n)
# raw prices to check circuit locks: entry day opened at its high and never traded lower
O, Hh, Lw, C, V = ({}, {}, {}, {}, {})
need = set(pd.concat([R[k][R[k].p_rank <= 5].ticker for k in R]))
for t in need:
    d = pd.read_csv(f'cache/backtest_ohlcv/{t}.csv', index_col=0, parse_dates=True)
    O[t], Hh[t], Lw[t], C[t], V[t] = d.Open, d.High, d.Low, d.Close, d.Volume
O, Hh, Lw, C, V = (pd.DataFrame(x) for x in (O, Hh, Lw, C, V))
idx = C.index
for y, k in (('up1', 1), ('up5', 5), ('up21', 21)):
    H = R[y]; top = H[H.p_rank <= 5].merge(P[['date', 'ticker', 'to20', 'price', 'dist_52wl', 'brk20', 'brk250', 'vr1', 'r20', 'r250', 'max_ret_20']], on=['date', 'ticker'])
    # entry day = next trading day; locked = open==high==low (no trade possible at a sane price) or open==high & close==high & gap>=4.9%
    locked = []
    for r in top.itertuples():
        i = idx.searchsorted(r.date, side='right')
        if i >= len(idx): locked.append(False); continue
        e = idx[i]; o, h, l, c = O.at[e, r.ticker], Hh.at[e, r.ticker], Lw.at[e, r.ticker], C.at[e, r.ticker]
        prev = C[r.ticker].iloc[i - 1]
        locked.append(bool((h - l) / o < 0.002 or (o >= h * 0.999 and o / prev - 1 >= 0.049)))
    top['locked'] = locked
    pdm = top.groupby('date')[f'pnl{k}'].mean()
    q = top.assign(q=top.date.dt.to_period('Q')).groupby('q').agg(n=(y, 'size'), hit=(y, 'mean'), pnl=(f'pnl{k}', 'mean'))
    base_q = H.assign(q=H.date.dt.to_period('Q')).groupby('q')[y].mean()
    q['random'] = base_q; q['x'] = q.hit / q.random
    print(f"\n=== {y} top-5/day: hit {top[y].mean():.1%}, P&L {top[f'pnl{k}'].mean():+.2f}%  (NW t on per-date P&L {nw_t(pdm.sort_index().values, k):+.2f})")
    print(f"  entry day opened at upper-circuit/locked: {top.locked.mean():.1%};  hit excluding locked: {top[~top.locked][y].mean():.1%}, P&L {top[~top.locked][f'pnl{k}'].mean():+.2f}%")
    tr = 10 ** top.to20 / 1e7
    print(f"  pick turnover Rs cr/day: median {tr.median():.1f}, 25% {tr.quantile(.25):.1f};  share <Rs 3cr {(tr<3).mean():.0%}; price median Rs {10**top.price.median():.0f}")
    big = top[tr >= 5]; print(f"  only picks with >= Rs 5cr/day turnover: n={len(big)} hit {big[y].mean():.1%} P&L {big[f'pnl{k}'].mean():+.2f}%")
    print("  by quarter:"); print(q.round(3).to_string())
    print("  what the picks look like (median) vs all tradable:")
    allm = P[P.date >= '2025-01-08'][['atr14', 'dist_52wl', 'brk20', 'brk250', 'vr1', 'r20', 'r250', 'max_ret_20']].median()
    print(pd.DataFrame({'picks': top[allm.index].median(), 'all': allm, 'picks_mean': top[allm.index].mean()}).round(3).to_string())
