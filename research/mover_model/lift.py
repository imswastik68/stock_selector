import numpy as np, pandas as pd
S = '/tmp/claude-1000/-home-swastik-Downloads-Projects/b82e419c-da14-438d-a197-0eaa67d2362f/scratchpad/re/'
P = pd.read_pickle(S + 'panel.pkl'); cut = pd.Timestamp('2025-01-08')
feats = [c for c in P.columns if c not in ('date','ticker') and not c[:2] in ('up','dn') and not c.startswith(('end','pnl','mae'))]
y = 'up5'; base = P[y].mean()
# 1) anatomy: winners vs everyone else (median)
w = P[P[y]]; nw = P[~P[y]]
print(f"ONE-WEEK WINNERS (touched +12.63%): {len(w):,} of {len(P):,} stock-days ({base:.1%})\n")
print(f"{'feature':<14}{'winners':>10}{'others':>10}  | top-decile hit rate  train / holdout | same, within ATR quintile (vol-controlled) train / holdout")
P['atrq'] = P.groupby('date').atr14.transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False))
rows = []
for f in feats:
    x = P[[f, y, 'date', 'atrq', 'pnl5']].dropna()
    dec = x.groupby('date')[f].transform(lambda s: pd.qcut(s.rank(method='first'), 10, labels=False))
    x = x.assign(dec=dec)
    def top_vs(d):
        hi = d[d.dec == 9][y].mean(); lo = d[d.dec == 0][y].mean(); return hi, lo
    tr, ho = x[x.date < cut], x[x.date >= cut]
    (h_tr, l_tr), (h_ho, l_ho) = top_vs(tr), top_vs(ho)
    # vol-controlled: within each ATR quintile, rank feature into deciles; top-decile hit / quintile base, averaged
    xv = x.assign(vdec=x.groupby(['date', 'atrq'])[f].transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False, duplicates='drop') if len(s) >= 5 else s * np.nan))
    def vc(d):
        g = d.groupby('atrq'); return np.nanmean([(gg[gg.vdec == 4][y].mean() / gg[y].mean()) for _, gg in g]), np.nanmean([(gg[gg.vdec == 0][y].mean() / gg[y].mean()) for _, gg in g])
    vt, vh = vc(xv[xv.date < cut]), vc(xv[xv.date >= cut])
    rows.append((f, w[f].median(), nw[f].median(), h_tr, l_tr, h_ho, l_ho, vt, vh,
                 x[x.dec == 9].pnl5.mean()))
R = pd.DataFrame(rows, columns=['f','win_med','oth_med','top_tr','bot_tr','top_ho','bot_ho','vc_tr','vc_ho','pnl_top'])
R['spread'] = (R.top_tr - R.bot_tr).abs()
for r in R.sort_values('spread', ascending=False).itertuples():
    print(f"{r.f:<14}{r.win_med:>10.3f}{r.oth_med:>10.3f}  | top {r.top_tr:5.1%}/{r.top_ho:5.1%}  bottom {r.bot_tr:5.1%}/{r.bot_ho:5.1%} | "
          f"top-quintile x{r.vc_tr[0]:.2f}/x{r.vc_ho[0]:.2f}  bottom x{r.vc_tr[1]:.2f}/x{r.vc_ho[1]:.2f} | P&L top-dec {r.pnl_top:+.2f}%")
print(f"\nbase train {P[P.date<cut][y].mean():.1%}  holdout {P[P.date>=cut][y].mean():.1%};  ATR-quintile base rates:",
      P.groupby('atrq')[y].mean().round(3).to_dict(), " P&L by ATR quintile:", P.groupby('atrq').pnl5.mean().round(2).to_dict())
R.to_pickle(S + 'lift.pkl')
