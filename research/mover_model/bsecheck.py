import sys, numpy as np, pandas as pd
from datetime import date
sys.path.insert(0, 'scripts'); import top50_scorecard as sc
S = '/tmp/claude-1000/-home-swastik-Downloads-Projects/b82e419c-da14-438d-a197-0eaa67d2362f/scratchpad/re/'
R = pd.read_pickle(S + 'gbm_holdout.pkl')
p = sc.load_panel(date(2025, 9, 15), date(2026, 11, 11)); nse = sc.nse_symbol_to_isin(); cal = p['cal']
for y, k in (('up21', 21), ('up5', 5), ('up1', 1)):
    H = R[y]; top = H[(H.p_rank <= 5) & (H.date >= '2025-11-12')].copy()
    out = []
    for r in top.itertuples():
        sym = r.ticker.replace('_NS', '').replace('_BO', ''); j = p['isin_col'].get(nse.get(sym) or p['bse_symbol_to_isin'].get(sym))
        e = cal.searchsorted(r.date, side='right')
        if j is None or e + k > len(cal): out.append(None); continue
        entry = p['O'][e, j]; w = slice(e, e + k)
        if not np.isfinite(entry): out.append(None); continue
        s = sc._score(entry, p['L'][w.stop - 1, j], np.nanmax(p['H'][w, j]), 'bullish', k)
        out.append((bool(s['touch']), float(s['pnl']), bool(p['ok'][e, j])))
    top['bse'] = out; t = top.dropna(subset=['bse'])
    t['bse_hit'] = t.bse.str[0]; t['bse_pnl'] = t.bse.str[1]; t['bse_ok'] = t.bse.str[2]
    print(f"{y}: {len(t)}/{len(top)} picks matched on BSE | yfinance hit {t[y].mean():.1%} P&L {t[f'pnl{k}'].mean():+.2f}%  ->  BSE hit {t.bse_hit.mean():.1%} P&L {t.bse_pnl.mean():+.2f}%"
          f" | label agreement {(t[y] == t.bse_hit).mean():.1%} | BSE-tradable (A/B/X) {t.bse_ok.mean():.0%}, among them hit {t[t.bse_ok].bse_hit.mean():.1%} P&L {t[t.bse_ok].bse_pnl.mean():+.2f}%")

# which filter fails? group from bhav on entry day
import glob
fr = pd.concat(pd.read_csv(f, usecols=['TradDt', 'ISIN', 'SctySrs', 'ClsPric']) for f in sorted(glob.glob('cache/bse_bhav/*.csv')) if f[-12:-4] >= '20251101')
fr['TradDt'] = pd.to_datetime(fr.TradDt); fr = fr.drop_duplicates(['ISIN', 'TradDt']); G = fr.set_index(['ISIN', 'TradDt']).SctySrs
H = R['up21']; top = H[(H.p_rank <= 5) & (H.date >= '2025-11-12')].copy()
grp, hit = [], []
for r in top.itertuples():
    sym = r.ticker.replace('_NS', ''); isin = nse.get(sym) or p['bse_symbol_to_isin'].get(sym)
    e = cal.searchsorted(r.date, side='right')
    grp.append(G.get((isin, cal[e]), 'n/a') if e < len(cal) else 'n/a')
top['grp'] = grp
print("\n1-month picks by BSE group (A/B = normal, X = BSE-only, T/XT = trade-to-trade, Z = non-compliant, M/MT = SME):")
print(top.groupby('grp').agg(n=('up21', 'size'), hit=('up21', 'mean'), pnl=('pnl21', 'mean')).round(3).to_string())
