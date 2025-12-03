from data import ETFDataLoader
import yfinance as yf
import json
import pandas as pd


ticker = yf.Ticker("NVDA")
news = ticker.news

import yfinance as yf
import pandas as pd

def get_recent_news(ticker):
    tk = yf.Ticker(ticker)
    raw = tk.news

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
    return df

if __name__ == "__main__":
    df = get_recent_news("AAPL")
    print(df)

'''print("\n================ RAW NEWS ==================\n")
print(json.dumps(news, indent=4))  # Pretty-print JSON'''




'''for i, item in enumerate(news):
    print(f"--- News #{i+1} ---")
    print(list(item.keys()))
    print()'''
'''for item in news:
    raw = item.get("providerPublishTime")
    dt = pd.to_datetime(raw, unit="s")
    print(dt)'''