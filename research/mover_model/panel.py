# Build a long panel: one row per (date, tradable stock) with features known at close t
# and labels for touching the top-50 bar from the next open.
import glob, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
S = '/tmp/claude-1000/-home-swastik-Downloads-Projects/b82e419c-da14-438d-a197-0eaa67d2362f/scratchpad/re/'
M = {k: {} for k in ('Open','High','Low','Close','Volume')}
for f in glob.glob('cache/backtest_ohlcv/*.csv'):
    d = pd.read_csv(f, index_col=0, parse_dates=True)
    t = f.split('/')[-1][:-4]
    for k in M: M[k][t] = pd.to_numeric(d[k], errors='coerce')
O, H, L, C, V = (pd.DataFrame(M[k]).sort_index().loc['2019-06-01':].astype('float32') for k in ('Open','High','Low','Close','Volume'))
V = V.where(V > 0)
# yfinance emits exchange-holiday rows with no volume; they blank every 20-day window they touch
good = V.notna().sum(axis=1) >= 0.5 * V.notna().sum(axis=1).median()
print('dropping no-volume dates:', [str(d.date()) for d in V.index[~good]])
O, H, L, C, V = (x[good] for x in (O, H, L, C, V))
N = pd.read_csv('cache/backtest_nifty.csv', skiprows=[1,2], index_col=0); N.index = pd.to_datetime(N.index)
nC = pd.to_numeric(N['Close'], errors='coerce').reindex(C.index).ffill()
ret = C.pct_change(fill_method=None)
TO = (C * V)
F = {}
for k in (1, 5, 20, 60, 120, 250): F[f'r{k}'] = C / C.shift(k) - 1
hh250 = H.rolling(250, min_periods=120).max(); ll250 = L.rolling(250, min_periods=120).min()
F['dist_52wh'] = C / hh250 - 1; F['dist_52wl'] = C / ll250 - 1
tr = pd.concat([(H - L), (H - C.shift()).abs(), (L - C.shift()).abs()]).groupby(level=0).max() if False else np.maximum(H - L, np.maximum((H - C.shift()).abs(), (L - C.shift()).abs()))
F['atr14'] = tr.rolling(14).mean() / C
F['atr_ratio'] = tr.rolling(5).mean() / tr.rolling(50).mean()          # range expansion/contraction
F['vol20'] = ret.rolling(20).std()
F['vr1'] = V / V.rolling(20).mean().shift(1)
F['vr5'] = V.rolling(5).mean() / V.rolling(60).mean().shift(5)
F['to20'] = np.log10(TO.rolling(20).mean())                              # log turnover (size proxy)
F['price'] = np.log10(C)
F['clv'] = ((C - L) / (H - L)).where(H > L)
F['gap'] = O / C.shift() - 1
F['day_ret'] = ret
up = (ret > 0).astype('float32'); F['up_streak'] = up.groupby((up != up.shift()).cumsum(), axis=0).cumsum() if False else up.rolling(5).sum()
for w in (20, 50, 200): F[f'ma{w}'] = C / C.rolling(w).mean() - 1
d = C.diff(); g = d.clip(lower=0).ewm(alpha=1/14).mean(); l = (-d.clip(upper=0)).ewm(alpha=1/14).mean(); F['rsi'] = 100 - 100 / (1 + g / l)
bbw = C.rolling(20).std() / C.rolling(20).mean(); F['bbw_rank'] = bbw.rolling(120, min_periods=60).rank(pct=True)
F['brk20'] = (C >= H.rolling(20).max().shift(1)).astype('float32')
F['brk250'] = (C >= hh250.shift(1)).astype('float32')
F['days_since_hh'] = (H.rolling(250, min_periods=120).apply(lambda x: len(x) - 1 - np.argmax(x), raw=True))
F['max_ret_20'] = ret.rolling(20).max()                                  # lottery-ness: biggest 1-day jump last month
F['rel_r20'] = F['r20'].sub(nC / nC.shift(20) - 1, axis=0)
# market context (same for all stocks on a day)
breadth = (C > C.rolling(50).mean()).where(C.notna()).mean(axis=1)
mkt = pd.DataFrame({'nifty_r20': nC / nC.shift(20) - 1, 'breadth50': breadth, 'univ_r5': ret.rolling(5).sum().median(axis=1)})
# tradable mask: price >= 20 and prior-20d avg turnover >= 1 cr
trad = (C >= 20) & (TO.rolling(20).mean().shift(1) >= 1e7)
# labels from next open
E = O.shift(-1)
def fmax(X, k): return pd.concat([X.shift(-j) for j in range(1, k + 1)], axis=1, keys=range(k)).T.groupby(level=1).max().T if False else np.fmax.reduce([X.shift(-j).values for j in range(1, k + 1)])
def fmin(X, k): return np.fmin.reduce([X.shift(-j).values for j in range(1, k + 1)])
lab = {}
for k, up_bar, dn_bar in ((1, 5.71, -4.44), (5, 12.63, -9.65), (21, 27.54, -19.71)):
    hi = fmax(H, k) / E.values * 100 - 100; lo = fmin(L, k) / E.values * 100 - 100
    end = C.shift(-k).values / E.values * 100 - 100
    lab[f'up{k}'] = hi >= up_bar; lab[f'dn{k}'] = lo <= dn_bar; lab[f'end{k}'] = end
    lab[f'pnl{k}'] = np.where(hi >= up_bar, up_bar, end) - 0.30            # limit-sell at bar, else window close
    lab[f'mae{k}'] = lo
valid = trad.values & np.isfinite(E.values) & np.isfinite(C.shift(-21).values)
dates = np.repeat(C.index.values[:, None], C.shape[1], 1)[valid]
tick = np.repeat(C.columns.values[None, :], C.shape[0], 0)[valid]
P = pd.DataFrame({'date': dates, 'ticker': tick})
for k, X in F.items(): P[k] = X.values[valid].astype('float32')
for k, X in lab.items(): P[k] = X[valid]
P = P.merge(mkt, left_on='date', right_index=True, how='left')
P = P[P.date >= '2021-01-01'].reset_index(drop=True)
P.to_pickle(S + 'panel.pkl')
print(P.shape, P.date.min(), P.date.max(), "stocks/day median", int(P.groupby('date').size().median()))
print("base rates:", {k: round(P[k].mean() * 100, 2) for k in ('up1', 'up5', 'up21', 'dn1', 'dn5', 'dn21')})
