import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

class FinBertSentiment:
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
    
    
ETF_SENTIMENT_PROXIES={
    "VTI": "AAPL",   # US market proxy
    "SPY": "MSFT",   # S&P 500 proxy
    "QQQ": "NVDA",   # Tech/NASDAQ proxy
    "BND": "JPM",    # Bond/macro sentiment proxy
    "TLT": "GS",     # Treasury/macro sentiment proxy
    "GLD": "AAPL"    # fallback proxy
}

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
    print(sentiment_features.head())