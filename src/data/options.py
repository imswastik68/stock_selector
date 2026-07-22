"""
NSE options chain reader: Put-Call Ratio (PCR) and OI buildup signals.

PCR interpretation (contrarian extremes only):
  > 1.5 : Extreme fear / put-heavy → contrarian bullish signal (pcr_fear)
  < 0.5 : Extreme complacency / call-heavy → warning signal (pcr_greed)
  0.5-1.5: Normal range — no signal

OI buildup:
  price_up  + OI_up   → long_buildup  (bullish — fresh longs entering)
  price_up  + OI_down → short_covering (bullish — shorts exiting)
  price_down + OI_up  → short_buildup  (bearish — fresh shorts entering)
  price_down + OI_down→ long_unwinding (bearish — longs exiting)

For scoring we use:
  pcr_fear       : PCR > 1.5 (extreme fear = contrarian bullish)     → +2
  long_buildup   : price up + OI up = fresh longs entering            → +1
  short_buildup  : bearish OI signal                                  → −2
  pcr_greed      : PCR < 0.5 (extreme complacency = warning)         → −1

All NSE API calls require a valid cookie session — obtained by hitting the
homepage first. Fails gracefully: all signals default to False on any error.
"""

from __future__ import annotations

import io
import json
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import requests

from src.cache import load_today, save_today

_OUTPUTS_DIR   = Path(__file__).parent.parent.parent / "outputs"
_PCR_CACHE_FILE = _OUTPUTS_DIR / "pcr_cache.json"
_PCR_HISTORY_LEN = 60   # rolling window: 60 trading days ≈ 3 months
_PCR_MIN_POINTS  = 10   # fall back to fixed thresholds below this count

_NSE_HOME = "https://www.nseindia.com/"
_OC_EQUITIES = "https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"

_MIN_CALL_OI = 5000  # ignore options signals on illiquid chains (SME / thinly-traded stocks)

# DATA SOURCE (rewritten 2026-07): NSE's EOD F&O bhavcopy, not the live API.
#
# The live API (api/option-chain-equities) is blocked for scripted clients --
# the cookie handshake 403s and the endpoint returns a bare "{}", verified
# against RELIANCE/INFY/SBIN, the three most liquid F&O names in India. That is
# why the old fetch reported "0 with options data" for every ticker while
# blaming "non-F&O".
#
# The ARCHIVE path is not blocked and is the authoritative settlement record --
# the same nsearchives.nseindia.com host src/data/delivery.py already reads
# successfully. One ~1.2 MB zip per trading day carries every contract: 38,175
# rows / 215 underlyings on 2026-07-21 (32,393 stock options, 5,142 index
# options, 625 stock futures) with strike, expiry, OI, OI change, volume,
# settlement and underlying price.
#
# EOD rather than intraday. For swing horizons (5-10 days) that is the right
# resolution -- and it is what finally makes these signals BACKTESTABLE, since
# the same archive exists historically. They stay at weight 0 until
# scripts/backtest_events.py returns a SHIP verdict, like every other signal here.
_FO_BHAV_URL = ("https://nsearchives.nseindia.com/content/fo/"
                "BhavCopy_NSE_FO_0_0_0_{date}_F_0000.csv.zip")
_FO_BHAV_LOOKBACK_DAYS = 6  # walk back past weekends/holidays to the last published file

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": _NSE_HOME,
}

_EMPTY = {
    "pcr": None,
    "pcr_fear": False,    # PCR > 1.5 — extreme fear = contrarian bullish
    "pcr_greed": False,   # PCR < 0.5 — extreme complacency = warning
    "long_buildup": False,
    "short_buildup": False,
    "short_covering": False,
    "long_unwinding": False,
    "total_put_oi": 0,
    "total_call_oi": 0,
}


def _load_pcr_cache() -> dict[str, list[float]]:
    if _PCR_CACHE_FILE.exists():
        try:
            return json.loads(_PCR_CACHE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_pcr_cache(cache: dict[str, list[float]]) -> None:
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    _PCR_CACHE_FILE.write_text(json.dumps(cache))


def _percentile(values: list[float], p: float) -> float:
    """Linear interpolation percentile — no numpy needed."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def _make_session() -> requests.Session:
    """Plain NSE-headered session. The archive host needs no cookie handshake."""
    s = requests.Session()
    s.headers.update(_HEADERS)
    return s


def fetch_fo_bhavcopy(as_of: date | None = None):
    """
    Download and parse the most recent published F&O bhavcopy as a DataFrame.

    Walks back up to _FO_BHAV_LOOKBACK_DAYS (weekends/holidays have no file, and
    the current day's is not published until after the close). Cached per
    calendar day -- the file is ~1.2 MB and never changes once published.
    Returns None if nothing could be fetched.
    """
    import pandas as pd

    cached = load_today("fo_bhavcopy")
    if cached is not None:
        return cached

    session = _make_session()
    start = as_of or date.today()
    for back in range(0, _FO_BHAV_LOOKBACK_DAYS + 1):
        d = start - timedelta(days=back)
        url = _FO_BHAV_URL.format(date=d.strftime("%Y%m%d"))
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code != 200 or len(resp.content) < 1000:
                continue
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                df = pd.read_csv(z.open(z.namelist()[0]))
            df.columns = [c.strip() for c in df.columns]
            print(f"[options] F&O bhavcopy {d}: {len(df)} contracts, "
                  f"{df['TckrSymb'].nunique()} underlyings")
            save_today("fo_bhavcopy", df)
            return df
        except Exception as exc:
            print(f"[options] bhavcopy {d} failed: {type(exc).__name__} {exc}")
            continue

    print(f"[options] no F&O bhavcopy found in the last {_FO_BHAV_LOOKBACK_DAYS} days")
    return None


def _signals_from_bhavcopy(df, symbol: str, prev_close: float | None) -> dict:
    """
    Derive the same signal dict the option-chain parser produced, from this
    symbol's rows in the F&O bhavcopy.

    PCR and OI-change use the NEAREST expiry only. That's where swing-relevant
    positioning sits; summing every expiry mixes a 3-month-out strike nobody is
    trading into the same number as the front month.
    """
    opts = df[(df["TckrSymb"] == symbol) & (df["FinInstrmTp"] == "STO")]
    if opts.empty:
        return dict(_EMPTY)

    # Nearest expiry present for this symbol.
    expiries = sorted(opts["XpryDt"].dropna().unique())
    if not expiries:
        return dict(_EMPTY)
    front = opts[opts["XpryDt"] == expiries[0]]

    calls = front[front["OptnTp"] == "CE"]
    puts  = front[front["OptnTp"] == "PE"]

    total_call_oi = float(calls["OpnIntrst"].sum())
    total_put_oi  = float(puts["OpnIntrst"].sum())
    total_call_chg = float(calls["ChngInOpnIntrst"].sum())
    total_put_chg  = float(puts["ChngInOpnIntrst"].sum())

    pcr = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else None

    # Underlying price today, straight from the settlement record.
    ul_series = front["UndrlygPric"].dropna()
    ul = float(ul_series.iloc[0]) if not ul_series.empty else 0.0

    price_up = bool(prev_close and prev_close > 0 and ul > prev_close)
    price_dn = bool(prev_close and prev_close > 0 and ul < prev_close)
    oi_up = (total_call_chg + total_put_chg) > 0

    liquid = total_call_oi >= _MIN_CALL_OI

    return {
        "pcr": pcr,
        # Cold-start fixed thresholds; fetch_options_signals overrides these with
        # percentile-based ones once enough history accumulates.
        "pcr_fear":  bool(liquid and pcr is not None and pcr > 1.5),
        "pcr_greed": bool(liquid and pcr is not None and pcr < 0.5),
        "long_buildup":   bool(liquid and price_up and oi_up),
        "short_covering": bool(liquid and price_up and not oi_up),
        "short_buildup":  bool(liquid and price_dn and oi_up),
        "long_unwinding": bool(liquid and price_dn and not oi_up),
        "total_put_oi": total_put_oi,
        "total_call_oi": total_call_oi,
    }


def _parse_option_chain(data: dict, prev_close: float | None) -> dict:
    """
    Parse NSE option-chain response and return signal dict.
    prev_close is used for OI buildup direction (today close vs yesterday close).
    """
    records = data.get("records", {}).get("data", [])
    if not records:
        return dict(_EMPTY)

    total_put_oi  = sum(r.get("PE", {}).get("openInterest", 0) for r in records if "PE" in r)
    total_call_oi = sum(r.get("CE", {}).get("openInterest", 0) for r in records if "CE" in r)
    total_put_chg = sum(r.get("PE", {}).get("changeinOpenInterest", 0) for r in records if "PE" in r)
    total_call_chg = sum(r.get("CE", {}).get("changeinOpenInterest", 0) for r in records if "CE" in r)

    pcr = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else None

    # Underlying close vs prev close
    ul = data.get("records", {}).get("underlyingValue") or 0
    # We rely on caller to pass prev_close; fallback to OI change direction only
    price_up = (prev_close is not None and prev_close > 0 and ul > prev_close)
    price_dn = (prev_close is not None and prev_close > 0 and ul < prev_close)

    # OI expanding or contracting
    net_oi_chg = total_call_chg + total_put_chg
    oi_up = net_oi_chg > 0

    long_buildup   = bool(price_up and oi_up)
    short_covering = bool(price_up and not oi_up)
    short_buildup  = bool(price_dn and oi_up)
    long_unwinding = bool(price_dn and not oi_up)

    # Gate all signals on minimum liquidity — illiquid option chains produce noise
    liquid = total_call_oi >= _MIN_CALL_OI

    # Contrarian extremes only; normal PCR range (0.5-1.5) carries no signal
    pcr_fear  = bool(liquid and pcr is not None and pcr > 1.5)
    pcr_greed = bool(liquid and pcr is not None and pcr < 0.5)

    # OI signals only meaningful on liquid chains
    long_buildup   = bool(liquid and long_buildup)
    short_buildup  = bool(liquid and short_buildup)
    short_covering = bool(liquid and short_covering)
    long_unwinding = bool(liquid and long_unwinding)

    return {
        "pcr": pcr,
        "pcr_fear": pcr_fear,
        "pcr_greed": pcr_greed,
        "long_buildup": long_buildup,
        "short_buildup": short_buildup,
        "short_covering": short_covering,
        "long_unwinding": long_unwinding,
        "total_put_oi": total_put_oi,
        "total_call_oi": total_call_oi,
    }


def _fetch_one(ticker: str, session: requests.Session, prev_close: float | None) -> tuple[str, dict]:
    symbol = ticker.replace(".NS", "").replace("[PENNY]", "")
    try:
        url = _OC_EQUITIES.format(symbol=symbol)
        resp = session.get(url, timeout=15)
        if resp.status_code == 403:
            # Session expired — return empty (caller handles retry)
            return ticker, dict(_EMPTY)
        resp.raise_for_status()
        data = resp.json()
        return ticker, _parse_option_chain(data, prev_close)
    except Exception as exc:
        print(f"[options] {ticker}: {exc}")
        return ticker, dict(_EMPTY)


def fetch_options_signals(
    tickers: list[str],
    prev_closes: dict[str, float] | None = None,
    max_workers: int = 4,
) -> dict[str, dict]:
    """
    Fetch NSE options data for a list of tickers.

    tickers     : list of NSE tickers e.g. ["SBIN.NS", "RELIANCE.NS"]
    prev_closes : {ticker: yesterday_close} — used for OI buildup direction
    Returns     : {ticker: signal_dict}

    Only F&O-eligible stocks have options chains. Non-F&O tickers silently return _EMPTY.

    Sourced from the EOD F&O bhavcopy (see _FO_BHAV_URL) -- ONE download per
    day for the whole market, then a local lookup per ticker, rather than the
    old per-ticker API calls.
    """
    if not tickers:
        return {}

    prev_closes = prev_closes or {}
    results: dict[str, dict] = {}

    df = fetch_fo_bhavcopy()
    if df is None:
        print(f"[options] no bhavcopy available — {len(tickers)} tickers get empty signals")
        return {t: dict(_EMPTY) for t in tickers}

    for t in tickers:
        symbol = t.replace(".NS", "").replace("[PENNY]", "")
        try:
            results[t] = _signals_from_bhavcopy(df, symbol, prev_closes.get(t))
        except Exception as exc:
            print(f"[options] {t}: {type(exc).__name__} {exc}")
            results[t] = dict(_EMPTY)

    # Percentile-based PCR signals: update rolling history, override fixed-threshold flags
    cache = _load_pcr_cache()
    for ticker, sig in results.items():
        pcr = sig.get("pcr")
        if pcr is None:
            continue
        history = cache.get(ticker, [])
        history.append(float(pcr))
        cache[ticker] = history[-_PCR_HISTORY_LEN:]

        liquid = sig.get("total_call_oi", 0) >= _MIN_CALL_OI
        if liquid and len(history) >= _PCR_MIN_POINTS:
            p80 = _percentile(history, 80)
            p20 = _percentile(history, 20)
            sig["pcr_fear"]  = bool(pcr > p80)
            sig["pcr_greed"] = bool(pcr < p20)
        # else: keep fixed-threshold values from _parse_option_chain (cold-start fallback)

        print(f"[options] {ticker}: PCR={pcr} "
              f"long_buildup={sig['long_buildup']} pcr_fear={sig['pcr_fear']} "
              f"pcr_greed={sig['pcr_greed']} history_len={len(history)}")

    _save_pcr_cache(cache)

    no_data = sum(1 for v in results.values() if v["pcr"] is None)
    print(f"[options] {len(tickers)} tickers, {len(tickers) - no_data} with options data, "
          f"{no_data} not F&O-listed")
    return results
