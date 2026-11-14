import json, glob, os, numpy as np, pandas as pd
S = '/tmp/claude-1000/-home-swastik-Downloads-Projects/b82e419c-da14-438d-a197-0eaa67d2362f/scratchpad/re/'
P = pd.read_pickle(S + 'panel.pkl'); P = P[P.date >= '2025-11-15'].copy()
P['breadth50'] = pd.to_numeric(P.breadth50, errors='coerce')
Q, I = {}, {}
for f in glob.glob(S + 'fund/*.json'):
    t = os.path.basename(f)[:-5]; d = json.load(open(f))
    if d.get('q'): Q[t] = {pd.Timestamp(k): v for k, v in d['q'].items()}
    I[t] = d.get('info') or {}
def known(qend):  # results deadline: 45 days, 60 for the March (annual) quarter
    return qend + pd.Timedelta(days=60 if qend.month == 3 else 45)
def fund_row(t, day):
    q = Q.get(t)
    if not q: return {}
    ks = sorted(k for k in q if known(k) <= day)
    if not ks: return {}
    cur = q[ks[-1]]; prev = q[ks[-2]] if len(ks) > 1 else {}
    yago = q.get(min(q, key=lambda k: abs((k - (ks[-1] - pd.DateOffset(years=1))).days))) if len(q) > 1 else None
    if yago is not None and abs((min(q, key=lambda k: abs((k - (ks[-1] - pd.DateOffset(years=1))).days)) - (ks[-1] - pd.DateOffset(years=1))).days) > 20: yago = None
    def g(a, b, key):
        x, y = (a or {}).get(key), (b or {}).get(key)
        return (x - y) / abs(y) if x is not None and y not in (None, 0) else np.nan
    rev, ni, oi = cur.get('Total Revenue'), cur.get('Net Income'), cur.get('Operating Income')
    return {'profit_pos': float(ni > 0) if ni is not None else np.nan,
            'ni_qoq': g(cur, prev, 'Net Income'), 'rev_qoq': g(cur, prev, 'Total Revenue'),
            'ni_yoy': g(cur, yago, 'Net Income'), 'rev_yoy': g(cur, yago, 'Total Revenue'),
            'op_margin': oi / rev if oi is not None and rev else np.nan,
            'turnaround': float((ni or 0) > 0 and (prev.get('Net Income') or 0) <= 0) if ni is not None and prev else np.nan}
rows = [fund_row(t, d) for t, d in zip(P.ticker, P.date)]
Fd = pd.DataFrame(rows, index=P.index); P = P.join(Fd)
for k in ('heldPercentInsiders', 'heldPercentInstitutions', 'debtToEquity', 'marketCap', 'trailingPE', 'priceToBook'):
    P[k] = pd.to_numeric(P.ticker.map(lambda t: I.get(t, {}).get(k)), errors='coerce')
P['sector'] = P.ticker.map(lambda t: I.get(t, {}).get('sector'))
P['log_mcap'] = np.log10(P.marketCap)
P.to_pickle(S + 'panel_fund.pkl')
fcols = ['profit_pos', 'turnaround', 'ni_qoq', 'rev_qoq', 'ni_yoy', 'rev_yoy', 'op_margin', 'heldPercentInsiders', 'heldPercentInstitutions', 'debtToEquity', 'log_mcap', 'trailingPE', 'priceToBook']
print(f"stock-days {len(P):,} ({P.date.min().date()} -> {P.date.max().date()}), with results data {P.ni_qoq.notna().mean():.0%}, base 1m-bar hit {P.up21.mean():.1%}\n")
P['atrq'] = P.groupby('date').atr14.transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False))
print(f"{'fundamental':<24}{'winners med':>12}{'others med':>11} | hit by quintile (Q1 low .. Q5 high), vol-controlled x-lift Q5 / Q1")
for f in fcols:
    x = P.dropna(subset=[f])
    if x[f].nunique() <= 2:
        g = x.groupby(f).up21.mean(); print(f"{f:<24}{x[x.up21][f].mean():>12.2f}{x[~x.up21][f].mean():>11.2f} | " + '  '.join(f"{k:.0f}: {v:.1%}" for k, v in g.items())); continue
    x = x.assign(qq=x.groupby('date')[f].transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False)))
    hq = x.groupby('qq').up21.mean()
    vc = x.assign(v=x.groupby(['date', 'atrq'])[f].transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False, duplicates='drop') if len(s) >= 5 else s * np.nan))
    lift = lambda b: np.nanmean([gg[gg.v == b].up21.mean() / gg.up21.mean() for _, gg in vc.groupby('atrq')])
    print(f"{f:<24}{x[x.up21][f].median():>12.3f}{x[~x.up21][f].median():>11.3f} | " + '  '.join(f"{v:5.1%}" for v in hq.values) + f" | x{lift(4):.2f} / x{lift(0):.2f}")
print("\nsector hit rate (1m bar):"); print(P.groupby('sector').agg(n=('up21', 'size'), hit=('up21', 'mean'), pnl=('pnl21', 'mean')).sort_values('hit', ascending=False).round(3).to_string())
