"""BSE results calendar fetcher — board meetings / EPS announcements in next 5 days."""

import time
import requests
import pandas as pd
from datetime import date, timedelta

BSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.bseindia.com/",
}

RESULT_CATEGORIES = {"results", "quarterly results", "financial results", "board meeting"}


def _bse_session() -> requests.Session:
    """Prime a BSE session by visiting the corporate actions page first."""
    s = requests.Session()
    s.headers.update(BSE_HEADERS)
    s.get("https://www.bseindia.com", timeout=15)
    time.sleep(1)
    s.get("https://www.bseindia.com/corporates/ann.html", timeout=15)
    time.sleep(0.5)
    return s


def _fetch_bse_calendar(from_date: date, to_date: date) -> list[dict]:
    fmt = "%Y%m%d"
    # Try two known BSE API endpoint patterns
    urls = [
        (
            "https://api.bseindia.com/BseIndiaAPI/api/DefaultData/w"
            f"?scripcode=&segment=0&strdate={from_date.strftime(fmt)}"
            f"&enddate={to_date.strftime(fmt)}&type=BC"
        ),
        (
            "https://api.bseindia.com/BseIndiaAPI/api/CorporateAction/w"
            f"?scripcode=&from_date={from_date.strftime('%d/%m/%Y')}"
            f"&to_date={to_date.strftime('%d/%m/%Y')}&type=BC&flag=C"
        ),
    ]
    session = _bse_session()
    for url in urls:
        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            rows = data.get("Table", data.get("Table1", data)) if isinstance(data, dict) else data
            if isinstance(rows, list) and rows:
                return rows
        except Exception as exc:
            print(f"[results] BSE endpoint failed: {exc}")
    return []


def fetch_results_calendar(days_ahead: int = 5) -> list[dict]:
    """
    Return upcoming earnings announcements for the next `days_ahead` trading days.

    Each dict: {ticker, result_date, company_name, analyst_eps_estimate}
    analyst_eps_estimate is None — BSE API doesn't provide consensus estimates;
    the Claude agent will mark this signal [UNCONFIRMED] unless external data is fed in.
    """
    today = date.today()
    end = today + timedelta(days=days_ahead + 2)  # small buffer for weekends

    rows = _fetch_bse_calendar(today, end)

    results = []
    seen = set()
    for row in rows:
        purpose = (row.get("PURPOSE") or row.get("purpose") or "").lower()
        if not any(cat in purpose for cat in RESULT_CATEGORIES):
            continue

        scrip = (row.get("SCRIP_CD") or row.get("scripcd") or "").strip()
        short_name = (row.get("SCRIP_SHORT_NAME") or row.get("short_name") or scrip).strip()
        ex_date_raw = row.get("EX_DATE") or row.get("exdate") or row.get("DT_TM") or ""

        try:
            ex_date = pd.to_datetime(ex_date_raw).date()
        except Exception:
            continue

        if ex_date < today or ex_date > end:
            continue

        # BSE uses scrip code; map to NSE ticker is non-trivial — use scrip code as placeholder
        # Users can supply a scrip→NSE mapping CSV in data/bse_nse_map.csv
        ticker = f"{short_name}.NS"

        key = (scrip, ex_date)
        if key in seen:
            continue
        seen.add(key)

        results.append({
            "ticker": ticker,
            "bse_scrip_code": scrip,
            "company_name": short_name,
            "result_date": ex_date.isoformat(),
            "analyst_eps_estimate": None,  # requires external consensus data
        })

    results.sort(key=lambda x: x["result_date"])
    print(f"[results] {len(results)} upcoming result announcements")
    return results
