import numpy as np, pandas as pd, lightgbm as lgb
S = '/tmp/claude-1000/-home-swastik-Downloads-Projects/b82e419c-da14-438d-a197-0eaa67d2362f/scratchpad/re/'
P = pd.read_pickle(S + 'panel.pkl')
cut = pd.Timestamp('2025-01-08')
feats = [c for c in P.columns if c not in ('date','ticker') and not c[:2] in ('up','dn') and not c.startswith(('end','pnl','mae'))]
# rank-normalise per date so the model learns cross-sectional position, not market level (keep market cols raw)
mkt = ['nifty_r20', 'breadth50', 'univ_r5']
X = P[feats].copy()
for f in feats:
    if f not in mkt: X[f] = P.groupby('date')[f].rank(pct=True)
X['month'] = P.date.dt.month
X = X.apply(pd.to_numeric, errors='coerce').astype('float32')
res = {}
for y in ('up1', 'up5', 'up21'):
    k = {'up1': 1, 'up5': 5, 'up21': 21}[y]
    # purge: drop train rows whose label window overlaps the holdout
    tr = P.date < cut - pd.Timedelta(days=int(k * 1.6) + 3); ho = P.date >= cut
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=500,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m.fit(X[tr], P.loc[tr, y])
    H = P.loc[ho, ['date', 'ticker', y, f'pnl{k}', f'end{k}', 'atr14']].copy()
    H['p'] = m.predict_proba(X[ho])[:, 1]
    H['atr_rank'] = P.loc[ho].groupby('date').atr14.rank(ascending=False)
    H['p_rank'] = H.groupby('date').p.rank(ascending=False)
    print(f"\n=== {y}: holdout {H.date.nunique()} days, base {H[y].mean():.1%} ===")
    print(f"{'pick each day':<34}{'hit':>7}{'x base':>8}{'P&L/pick':>10}{'dates>0 P&L':>12}")
    for lab, sel in [('random (all tradable)', H.index == H.index),
                     ('top-5 most volatile (ATR)', H.atr_rank <= 5),
                     ('top-5 by model', H.p_rank <= 5), ('top-20 by model', H.p_rank <= 20),
                     ('top-50 by model', H.p_rank <= 50)]:
        s = H[sel]; pd_ = s.groupby('date')[f'pnl{k}'].mean()
        print(f"{lab:<34}{s[y].mean():>7.1%}{s[y].mean()/H[y].mean():>8.2f}{s[f'pnl{k}'].mean():>+9.2f}%{(pd_>0).mean():>11.0%}")
    # calibration: does p>=30% ever happen, and does it hold?
    for th in (0.2, 0.3, 0.4):
        s = H[H.p >= th]
        if len(s): print(f"  model prob >= {th:.0%}: {len(s):>6} picks on {s.date.nunique():>3} days  hit {s[y].mean():.1%}  P&L {s[f'pnl{k}'].mean():+.2f}%")
    imp = pd.Series(m.booster_.feature_importance('gain'), index=X.columns).sort_values(ascending=False)
    print("  top drivers:", ', '.join(f"{a}" for a in imp.index[:10]))
    res[y] = H
pd.to_pickle(res, S + 'gbm_holdout.pkl')
