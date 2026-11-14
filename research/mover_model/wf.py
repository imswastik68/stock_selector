import numpy as np, pandas as pd, lightgbm as lgb
S = '/tmp/claude-1000/-home-swastik-Downloads-Projects/b82e419c-da14-438d-a197-0eaa67d2362f/scratchpad/re/'
P = pd.read_pickle(S + 'panel.pkl')
feats = [c for c in P.columns if c not in ('date','ticker') and not c[:2] in ('up','dn') and not c.startswith(('end','pnl','mae'))]
mkt = ['nifty_r20', 'breadth50', 'univ_r5']
X = P[feats].copy()
for f in feats:
    if f not in mkt: X[f] = P.groupby('date')[f].rank(pct=True)
X['month'] = P.date.dt.month; X = X.apply(pd.to_numeric, errors='coerce').astype('float32')
TO = 10 ** P.to20 / 1e7
def nw_t(x, lag):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x); u = x - x.mean(); g = u @ u / n
    for l in range(1, min(lag, n - 1) + 1): g += 2 * (1 - l / (lag + 1)) * (u[l:] @ u[:-l]) / n
    return x.mean() / np.sqrt(g / n)
out = []
for y, k in (('up21', 21), ('up5', 5)):
    print(f"\n=== {y}: walk-forward, retrained each year on all prior data (purged) ===")
    print(f"{'test year':<10}{'days':>5}{'random hit':>11}{'top5 hit':>9}{'x':>6}{'P&L/pick':>10}{'random P&L':>11}{'NW t':>7}{'  >=5cr: hit / P&L':>20}")
    for yr in (2023, 2024, 2025, 2026):
        start = pd.Timestamp(f'{yr}-01-01'); end = pd.Timestamp(f'{yr + 1}-01-01')
        tr = P.date < start - pd.Timedelta(days=int(k * 1.6) + 3); te = (P.date >= start) & (P.date < end)
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=500,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
        m.fit(X[tr], P.loc[tr, y])
        T = P.loc[te, ['date', 'ticker', y, f'pnl{k}']].copy(); T['p'] = m.predict_proba(X[te])[:, 1]; T['to'] = TO[te]
        T['rk'] = T.groupby('date').p.rank(ascending=False); top = T[T.rk <= 5]
        L = T[T.to >= 5].copy(); L['rk'] = L.groupby('date').p.rank(ascending=False); topL = L[L.rk <= 5]
        q = top.groupby('date')[f'pnl{k}'].mean()
        print(f"{yr:<10}{T.date.nunique():>5}{T[y].mean():>11.1%}{top[y].mean():>9.1%}{top[y].mean()/T[y].mean():>6.1f}{top[f'pnl{k}'].mean():>+9.2f}%{T[f'pnl{k}'].mean():>+10.2f}%{nw_t(q.values, k):>7.2f}"
              f"{topL[y].mean():>12.1%} / {topL[f'pnl{k}'].mean():+.2f}%")
        out.append(top.assign(model=y, yr=yr))
pd.concat(out).to_pickle(S + 'wf_picks.pkl')
