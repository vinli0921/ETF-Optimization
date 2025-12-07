from data import ETFDataLoader
import yfinance as yf
import json
import pandas as pd
import feedparser
import pandas as pd
import requests
def get_recent_news(ticker):
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}"
    feed = feedparser.parse(url)

    rows = []
    for entry in feed.entries:
        dt = pd.to_datetime(entry.published)
        text = entry.title + " " + entry.summary
        rows.append((dt, text))

    return pd.DataFrame(rows, columns=["date", "text"])

def print_raw(ticker):
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL"
    xml = requests.get(url).text
    print(xml)

'''def get_recent_news(ticker):
    tk = yf.Ticker(ticker)
    raw = tk.news
    print(len(raw))
    rows = []
    for item in raw:
        content = item.get("content", {})

        title = content.get("title", "")
        summary = content.get("summary", "")

        pub = content.get("pubDate")
        if not pub:
            continue

        dt = pd.to_datetime(pub).date()

        text = (title + " " + summary).strip()
        rows.append((dt, text))

    df = pd.DataFrame(rows, columns=["date", "text"])
    return df'''

if __name__ == "__main__":
    df = get_recent_news("META")
    print(df)