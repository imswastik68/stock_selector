# Monthly walk-forward of the 1-month (up21) LightGBM model: each month is scored
# by a model trained only on data whose 21-day labels had fully matured before it.
import numpy as np, pandas as pd, lightgbm as lgb, time
S = '/tmp/claude-1000/-home-swastik-Downloads-Projects/b82e419c-da14-438d-a197-0eaa67d2362f/scratchpad/re/'
P = pd.read_pickle(S + 'panel.pkl')
feats = [c for c in P.columns if c not in ('date','ticker') and not c[:2] in ('up','dn') and not c.startswith(('end','pnl','mae'))]
mkt = ['nifty_r20', 'breadth50', 'univ_r5']
X = P[feats].copy()
for f in feats:
    if f not in mkt: X[f] = P.groupby('date')[f].rank(pct=True)
X['month'] = P.date.dt.month; X = X.apply(pd.to_numeric, errors='coerce').astype('float32')
out = []
for m in pd.period_range('2023-01', '2026-08', freq='M'):
    t0 = time.time(); start = m.start_time; end = (m + 1).start_time
    tr = P.date < start - pd.Timedelta(days=37)          # 21 trading days of label + margin
    te = (P.date >= start) & (P.date < end)
    if not te.any(): continue
    mdl = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=500,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
    mdl.fit(X[tr], P.loc[tr, 'up21'])
    out.append(pd.DataFrame({'date': P.date[te].values, 'ticker': P.ticker[te].values, 'p': mdl.predict_proba(X[te])[:, 1]}))
    print(m, int(tr.sum()), int(te.sum()), f"{time.time() - t0:.0f}s", flush=True)
pd.concat(out).to_pickle(S + 'wf_monthly_pred.pkl')
print("done")
