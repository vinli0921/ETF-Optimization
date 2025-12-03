from data import ETFDataLoader
import yfinance as yf
import json

ticker = "SPY"
tk = yf.Ticker(ticker)

news = tk.news
print("Total news items:", len(news))
print(json.dumps(news[0], indent=2))   # print first article
