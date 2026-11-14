import numpy as np, pandas as pd
S = '/tmp/claude-1000/-home-swastik-Downloads-Projects/b82e419c-da14-438d-a197-0eaa67d2362f/scratchpad/re/'
P = pd.read_pickle(S + 'panel.pkl'); cut = pd.Timestamp('2025-01-08')
P['breadth50'] = pd.to_numeric(P.breadth50, errors='coerce')
def nw_t(x, lag):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x); u = x - x.mean(); g = u @ u / n
    for l in range(1, min(lag, n - 1) + 1): g += 2 * (1 - l / (lag + 1)) * (u[l:] @ u[:-l]) / n
    return x.mean() / np.sqrt(g / n)
TO = 10 ** P.to20 / 1e7
rules = {
  'top-5 by 12m return (r250)':                       (P.index == P.index, 'r250'),
  'top-5 by r250, ATR>=4%':                           (P.atr14 >= .04, 'r250'),
  'top-5 by r250, ATR>=4%, r20>0':                    ((P.atr14 >= .04) & (P.r20 > 0), 'r250'),
  'top-5 by r250, ATR>=4%, r20>0, >=Rs5cr/day':       ((P.atr14 >= .04) & (P.r20 > 0) & (TO >= 5), 'r250'),
  'top-5 by r250, ATR 4-8%, r20>0, within 15% of 52wH, >=5cr': ((P.atr14.between(.04, .08)) & (P.r20 > 0) & (P.dist_52wh > -.15) & (TO >= 5), 'r250'),
  'top-5 by 6m return (r120), ATR>=4%, r20>0, >=5cr': ((P.atr14 >= .04) & (P.r20 > 0) & (TO >= 5), 'r120'),
  'top-5 by dist from 52w low, ATR>=4%, >=5cr':       ((P.atr14 >= .04) & (TO >= 5), 'dist_52wl'),
}
for y, k in (('up21', 21), ('up5', 5)):
    print(f"\n=== {y} (bar {'+27.54% in 1 month' if k == 21 else '+12.63% in 1 week'}) — train 2021-01..2025-01 | holdout 2025-01..2026-08 ===")
    print(f"{'rule':<62}{'hit tr':>7}{'P&L tr':>8}{'hit ho':>8}{'P&L ho':>8}{'NW t ho':>8}{'days ho':>8}")
    for lab, (mask, by) in rules.items():
        x = P[mask].copy(); x['rk'] = x.groupby('date')[by].rank(ascending=False); x = x[x.rk <= 5]
        tr, ho = x[x.date < cut], x[x.date >= cut]
        print(f"{lab:<62}{tr[y].mean():>7.1%}{tr[f'pnl{k}'].mean():>+7.2f}%{ho[y].mean():>8.1%}{ho[f'pnl{k}'].mean():>+7.2f}%"
              f"{nw_t(ho.groupby('date')[f'pnl{k}'].mean().values, k):>8.2f}{ho.date.nunique():>8}")
    print(f"{'random tradable':<62}{P[P.date<cut][y].mean():>7.1%}{P[P.date<cut][f'pnl{k}'].mean():>+7.2f}%{P[P.date>=cut][y].mean():>8.1%}{P[P.date>=cut][f'pnl{k}'].mean():>+7.2f}%")
