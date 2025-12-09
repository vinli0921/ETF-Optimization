"""
Historical news fetcher using the free GDELT 2 Docs API.

Why GDELT?
- Covers 2015–present.
- Free to query with start/end datetimes.
- Returns stable URLs (unlike GoogleNews scraping).

Usage:
    python src/fetch_news.py --ticker AAPL --start-year 2015 --end-year 2016 --out data/AAPL_2015_2016.csv
"""

import argparse
from datetime import datetime, timedelta
import time
import random
from typing import List, Optional

import pandas as pd
import requests

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 250  # GDELT max per page


def iso_to_gdelt(dt: datetime) -> str:
    """Convert datetime to GDELT's YYYYMMDDhhmmss format."""
    return dt.strftime("%Y%m%d%H%M%S")


def fetch_gdelt_page(
    query: str,
    start_dt: datetime,
    end_dt: datetime,
    offset: int = 0,
    maxrecords: int = MAX_RECORDS,
) -> List[dict]:
    """Fetch a single page of GDELT results."""
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "startdatetime": iso_to_gdelt(start_dt),
        "enddatetime": iso_to_gdelt(end_dt),
        "maxrecords": maxrecords,
        "offset": offset,
        "sort": "DateDesc",
    }
    resp = requests.get(GDELT_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("articles", [])


def fetch_gdelt_month(query: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Fetch all pages for a ticker within [start_dt, end_dt].
    Returns DataFrame with date, title, url, source, lang, and ticker.
    """
    rows = []
    offset = 0
    while True:
        articles = fetch_gdelt_page(query, start_dt, end_dt, offset=offset)
        if not articles:
            break

        for art in articles:
            rows.append(
                {
                    "date": art.get("seendate"),
                    "title": art.get("title"),
                    "url": art.get("url"),
                    "source": art.get("sourceurl"),
                    "lang": art.get("language"),
                    "ticker": query,
                }
            )

        if len(articles) < MAX_RECORDS:
            break
        offset += MAX_RECORDS
        time.sleep(random.uniform(0.5, 1.5))  # polite pause

    return pd.DataFrame(rows)


def batch_fetch_gdelt(
    ticker: str, start_year: int, end_year: int, sleep_range=(1.0, 2.5)
) -> pd.DataFrame:
    """
    Loop month-by-month from start_year to end_year inclusive.
    """
    all_months = []
    current = datetime(start_year, 1, 1)
    end_dt = datetime(end_year, 12, 31, 23, 59, 59)

    while current <= end_dt:
        next_month = (current + timedelta(days=32)).replace(day=1)
        period_end = min(next_month - timedelta(days=1), end_dt)

        print(
            f"Fetching {ticker} {current.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}"
        )
        try:
            df = fetch_gdelt_month(ticker, current, period_end)
            if not df.empty:
                all_months.append(df)
                print(f"  {len(df)} articles")
            else:
                print("  0 articles")
        except Exception as exc:
            print(f"  Error: {exc}")

        # polite pause
        time.sleep(random.uniform(*sleep_range))
        current = next_month

    if all_months:
        return pd.concat(all_months, ignore_index=True)
    return pd.DataFrame()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch historical news via GDELT.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g., AAPL")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = batch_fetch_gdelt(args.ticker, args.start_year, args.end_year)
    print(f"Total fetched: {len(df)} articles")
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")
    else:
        print(df.head())


if __name__ == "__main__":
    main()

