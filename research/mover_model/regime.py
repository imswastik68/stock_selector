import numpy as np, pandas as pd
S = '/tmp/claude-1000/-home-swastik-Downloads-Projects/b82e419c-da14-438d-a197-0eaa67d2362f/scratchpad/re/'
P = pd.read_pickle(S + 'panel.pkl'); R = pd.read_pickle(S + 'gbm_holdout.pkl')
P['breadth50'] = pd.to_numeric(P.breadth50, errors='coerce')
cols = ['r250', 'r60', 'r20', 'r5', 'dist_52wh', 'dist_52wl', 'atr14', 'vr1', 'vr5', 'to20', 'price', 'brk20', 'brk250', 'max_ret_20', 'rsi']
H = R['up21']; top = H[H.p_rank <= 5].drop(columns=['atr14']).merge(P[['date', 'ticker', 'breadth50', 'nifty_r20'] + cols], on=['date', 'ticker'])
allho = P[P.date >= '2025-01-08']
top['regime'] = pd.cut(top.breadth50, [0, .35, .6, 1], labels=['weak mkt (<35% above 50DMA)', 'normal', 'strong mkt (>60%)'])
allho = allho.assign(regime=pd.cut(allho.breadth50, [0, .35, .6, 1], labels=['weak mkt (<35% above 50DMA)', 'normal', 'strong mkt (>60%)']))
for reg, g in top.groupby('regime', observed=True):
    a = allho[allho.regime == reg]
    print(f"\n### {reg}: {g.date.nunique()} days | picks hit {g.up21.mean():.1%} (random {a.up21.mean():.1%}) | P&L {g.pnl21.mean():+.2f}% (random {a.pnl21.mean():+.2f}%)")
    t = pd.DataFrame({'picks median': g[cols].median(), 'all stocks median': a[cols].median()})
    t.loc['turnover Rs cr'] = [10 ** g.to20.median() / 1e7, 10 ** a.to20.median() / 1e7]; t.loc['price Rs'] = [10 ** g.price.median(), 10 ** a.price.median()]
    print(t.drop(['to20', 'price']).round(3).to_string())
# winners vs losers WITHIN the model's picks
print("\n### within model picks: hit vs miss (median)")
print(pd.DataFrame({'hit': top[top.up21][cols].median(), 'miss': top[~top.up21][cols].median()}).round(3).to_string())
