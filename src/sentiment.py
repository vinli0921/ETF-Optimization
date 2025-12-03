import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import datetime

ETF_HOLDINGS={
    "SPY": ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"],
    "QQQ": ["AAPL", "MSFT", "NVDA", "AMZN", "META"],
    "VTI": ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"],
    "TLT": ["TLT"],  # Treasuries → use ETF news directly
    "BND": ["BND"],  # Bond news
    "GLD": ["GLD"]   # Gold news
}
class FinBertSentiment:
    def __init__(self):
        self.tokenizer=AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model=AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()

    def score(self, text):
        if not text or text.strip()=="":
            return 0.0

        tokens=self.tokenizer(text, return_tensors="pt", truncation=True)

        with torch.no_grad():
            logits=self.model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"]
            ).logits

        probs=torch.softmax(logits, dim=1).numpy()[0]
        return float(probs[2] - probs[0])

def get_recent_news(ticker):
    tk=yf.Ticker(ticker)
    raw_news=tk.news or []

    if len(raw_news)==0:
        return []

    rows=[]

    for item in raw_news:
        content=item.get("content", {})
        title=content.get("title", "") or ""
        summary=content.get("summary", "") or ""
        text = (title + ". " + summary).strip()
        if not text:
            continue

        # Job to extract the timestamp
        ts=item.get("providerPublishTime")

        # Try pubDate if "providerPublishTime" doesn't work
        if ts is None:
            ts=item.get("pubDate")

        # Last fallback → treat as now
        if ts is None:
            dt=datetime.datetime.now().date()
            rows.append((dt, text))
            continue

        # Try UNIX timestamp
        try:
            dt=pd.to_datetime(ts, unit="s")
        except:
            # Try normal timestamp
            try:
                dt=pd.to_datetime(ts)
            except:
                dt=datetime.datetime.now()

        rows.append((dt.date(), text))

    return rows


def compute_etf_sentiment(etf, finbert):
    holdings=ETF_HOLDINGS.get(etf, [])
    results=[]

    print(f"\n Computing sentiment for ETF {etf} using holdings {holdings}")

    for h in holdings:
        print(f"   → Fetching news for {h}...")
        news_items=get_recent_news(h)

        if len(news_items)==0:
            print(f"No news for {h}. Skipping.")
            continue

        for date, text in news_items:
            score=finbert.score(text)
            results.append((date, score))

    if len(results)==0:
        print(f"No sentiment data found for {etf}. Returning zeros.")
        return pd.Series(dtype=float)

    df=pd.DataFrame(results, columns=["date", "sentiment"])
    df=df.groupby("date")["sentiment"].mean().sort_index()

    return df

if __name__ == "__main__":
    ETFs=["SPY", "QQQ", "VTI", "TLT", "BND", "GLD"]

    finbert=FinBertSentiment()

    all_sent=pd.DataFrame()

    for etf in ETFs:
        s = compute_etf_sentiment(etf, finbert)

        if s.empty:
            # Create placeholder so DataFrame aligns
            all_sent[f"{etf}_sentiment"] = 0
        else:
            all_sent[f"{etf}_sentiment"] = s

    print("\nFinal ETF Sentiment DataFrame:")
    print(all_sent.tail())
    print("\nDone.")

''''class FinBertSentiment:
    def __init__(self):
        self.tokenizer=AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model=AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()
        
    def score(self,text):
        tokenized_inputs=self.tokenizer(text,return_tensors="pt",truncation=True)
        with torch.no_grad():
            logits = self.model(input_ids=tokenized_inputs["input_ids"],attention_mask=tokenized_inputs["attention_mask"]).logits
            
        tensor_probability=torch.softmax(logits,dim=1)
        numpy_probs=tensor_probability.detach().numpy()
        probabilities=numpy_probs[0]
        return float(probabilities[2]-probabilities[0])
    
    def daily_sentiment(self, ticker, start, end):
        tk=yf.Ticker(ticker)
        news=tk.news or []

        start_dt=pd.to_datetime(start).date()
        end_dt=pd.to_datetime(end).date()
        rows=[]
        for item in news:
            raw_dt=(item.get("providerPublishTime") or item.get("pubDate") or item.get("displayTime"))

            if raw_dt is None:
                continue
            try:
                dt=pd.to_datetime(raw_dt)
            except:
                continue
            d=dt.date()
            if not (start_dt <= d <= end_dt):
                continue
            title=item.get("title","")
            summary=item.get("summary","")
            text=(title+" "+summary).strip()
            if not text:
                continue

            score=self.score(text)
            rows.append((d, score))

    # Return empty series if no data
        if not rows:
            return pd.Series(dtype=float)

        df=pd.DataFrame(rows, columns=["date", "sentiment"])
        return df.groupby("date")["sentiment"].mean()
    
    

def generate_etf_sentiment_features(etf_prices):
    """
    Produces a sentiment feature for each ETF using its proxy ticker.
    Returns a DataFrame with columns:
        VTI_sentiment, SPY_sentiment, QQQ_sentiment, BND_sentiment, TLT_sentiment, GLD_sentiment
    """
    finbert=FinBertSentiment()
    sentiment_df=pd.DataFrame(index=etf_prices.index)

    start=str(etf_prices.index[0].date())
    end=str(etf_prices.index[-1].date())

    for etf in etf_prices.columns:
        proxy=ETF_SENTIMENT_PROXIES.get(etf, "AAPL")

        series=finbert.daily_sentiment(proxy, start, end)

        if series.empty:
            print(f"  No news for proxy {proxy}. Filling with zeros.\n")
            sentiment_df[f"{etf}_sentiment"] = 0.0
            continue

        aligned=series.reindex(etf_prices.index.date).fillna(0.0)
        aligned.index=etf_prices.index  # convert Date to DatetimeIndex
        sentiment_df[f"{etf}_sentiment"]=aligned
    print(sentiment_df.head())

    return sentiment_df



if __name__=="__main__":
    from data import load_default_etfs

    prices = load_default_etfs()
    sentiment_features = generate_etf_sentiment_features(prices)
    print(sentiment_features.head())'''