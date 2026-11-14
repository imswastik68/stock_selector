import json, os, time, pandas as pd, yfinance as yf, warnings
from concurrent.futures import ThreadPoolExecutor
warnings.simplefilter("ignore")
S = '/tmp/claude-1000/-home-swastik-Downloads-Projects/b82e419c-da14-438d-a197-0eaa67d2362f/scratchpad/re/'
os.makedirs(S + 'fund', exist_ok=True)
P = pd.read_pickle(S + 'panel.pkl')
tickers = sorted(P[P.date >= '2025-11-01'].ticker.unique())
print(len(tickers), "tickers", flush=True)
ROWS = ('Total Revenue', 'Net Income', 'Operating Income', 'Diluted EPS')
INFO = ('marketCap', 'sharesOutstanding', 'heldPercentInsiders', 'heldPercentInstitutions', 'debtToEquity', 'returnOnEquity', 'sector', 'industry', 'trailingPE', 'priceToBook')
def get(t):
    out = S + f'fund/{t}.json'
    if os.path.exists(out): return
    sym = t.replace('_NS', '.NS').replace('_BO', '.BO')
    for attempt in range(3):
        try:
            tk = yf.Ticker(sym); q = tk.quarterly_income_stmt
            qd = {str(c.date()): {r: (None if pd.isna(q.at[r, c]) else float(q.at[r, c])) for r in ROWS if r in q.index} for c in q.columns} if q is not None and not q.empty else {}
            info = tk.info or {}
            json.dump({'q': qd, 'info': {k: info.get(k) for k in INFO}}, open(out, 'w')); return
        except Exception as e:
            time.sleep(5 * (attempt + 1))
    json.dump({'q': {}, 'info': {}, 'err': True}, open(out, 'w'))
with ThreadPoolExecutor(6) as ex: list(ex.map(get, tickers))
print("done", len(os.listdir(S + 'fund')), flush=True)
