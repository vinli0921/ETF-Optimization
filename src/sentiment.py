"""
Sentiment analysis for ETF portfolio optimization.

Uses GDELT for historical news and FinBERT for sentiment scoring.
Supports caching to avoid repeated API calls during backtesting.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# Mapping from ETF tickers to search keywords for GDELT
ETF_KEYWORDS = {
    # US Equity ETFs
    "SPY": ["S&P 500", "SP500", "stock market", "Wall Street"],
    "QQQ": ["NASDAQ", "tech stocks", "technology sector", "QQQ"],
    "VTI": ["stock market", "US equities", "equity market"],
    "IWM": ["small cap stocks", "Russell 2000", "small cap"],

    # Bond ETFs
    "TLT": ["treasury bonds", "interest rates", "Fed rates", "long term bonds"],
    "BND": ["bond market", "fixed income", "bonds"],

    # Commodity ETFs
    "GLD": ["gold price", "gold market", "precious metals", "gold ETF"],

    # International ETFs
    "VEA": ["international stocks", "developed markets", "European stocks", "Japan stocks"],
    "VWO": ["emerging markets", "EM stocks", "developing markets", "emerging market"],

    # Sector ETFs
    "XLE": ["energy sector", "oil stocks", "oil price", "energy stocks"],
}

# Holdings for fallback to individual stock news
ETF_HOLDINGS = {
    "SPY": ["AAPL", "MSFT", "NVDA", "AMZN", "META"],
    "QQQ": ["MSFT", "NVDA", "AMZN", "META", "AVGO"],
    "VTI": ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"],
    "TLT": ["TLT"],
    "BND": ["BND"],
    "GLD": ["GLD"],
    "VEA": ["VEA"],
    "VWO": ["VWO"],
    "IWM": ["IWM"],
    "XLE": ["XOM", "CVX"],
}


class FinBertSentiment:
    """
    FinBERT-based sentiment scorer for financial text.

    Returns sentiment scores in range [-1, 1] where:
    - Positive values indicate bullish/positive sentiment
    - Negative values indicate bearish/negative sentiment
    - Zero indicates neutral sentiment
    """

    def __init__(self, device: str = None):
        """
        Initialize FinBERT model.

        Args:
            device: 'cuda', 'mps', or 'cpu'. Auto-detected if None.
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        print(f"Loading FinBERT model on {device}...")

        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.to(self.device)
        self.model.eval()

    def score(self, text: str) -> float:
        """
        Score a single text for sentiment.

        Args:
            text: Text to analyze

        Returns:
            Sentiment score in [-1, 1]
        """
        if not text or text.strip() == "":
            return 0.0

        # Truncate very long texts
        text = text[:512]

        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            logits = self.model(**tokens).logits

        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        # FinBERT outputs: [negative, neutral, positive]
        # Return P(positive) - P(negative)
        return float(probs[2] - probs[0])

    def score_batch(self, texts: List[str], batch_size: int = 16) -> List[float]:
        """
        Score multiple texts efficiently in batches.

        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing

        Returns:
            List of sentiment scores
        """
        scores = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [t[:512] if t else "" for t in batch]

            tokens = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            with torch.no_grad():
                logits = self.model(**tokens).logits

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            batch_scores = probs[:, 2] - probs[:, 0]
            scores.extend(batch_scores.tolist())

        return scores


class GDELTNewsLoader:
    """
    Fetches historical news from GDELT Project.

    GDELT provides free access to global news data dating back to 2015+.
    Uses the gdeltdoc library to query the GDELT DOC 2.0 API.
    """

    def __init__(self, cache_dir: str = "data"):
        """
        Initialize GDELT loader.

        Args:
            cache_dir: Directory for caching news data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Try to import gdeltdoc
        try:
            from gdeltdoc import GdeltDoc, Filters
            self.gdelt = GdeltDoc()
            self.Filters = Filters
            self._available = True
        except ImportError:
            print("Warning: gdeltdoc not installed. Run: pip install gdeltdoc")
            self._available = False

    def is_available(self) -> bool:
        """Check if GDELT is available."""
        return self._available

    def fetch_news(
        self,
        keywords: List[str],
        start_date: str,
        end_date: str,
        max_records: int = 250
    ) -> pd.DataFrame:
        """
        Fetch news articles from GDELT for given keywords and date range.

        Args:
            keywords: List of search keywords
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_records: Maximum number of records to fetch

        Returns:
            DataFrame with columns: date, title, url
        """
        if not self._available:
            return pd.DataFrame(columns=["date", "title", "url"])

        try:
            # GDELT queries one date range at a time
            f = self.Filters(
                keyword=keywords,
                start_date=start_date,
                end_date=end_date,
                num_records=max_records,
                country="US"  # Focus on US news for ETFs
            )

            articles = self.gdelt.article_search(f)

            if articles.empty:
                return pd.DataFrame(columns=["date", "title", "url"])

            # Process results
            result = pd.DataFrame({
                "date": pd.to_datetime(articles["seendate"]).dt.date,
                "title": articles["title"],
                "url": articles["url"]
            })

            return result

        except Exception as e:
            print(f"GDELT query failed: {e}")
            return pd.DataFrame(columns=["date", "title", "url"])

    def fetch_news_for_etf(
        self,
        etf: str,
        start_date: str,
        end_date: str,
        max_records_per_month: int = 100
    ) -> pd.DataFrame:
        """
        Fetch news for an ETF by querying its associated keywords.

        Handles large date ranges by batching queries by month.

        Args:
            etf: ETF ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_records_per_month: Max records per monthly query

        Returns:
            DataFrame with news data
        """
        keywords = ETF_KEYWORDS.get(etf, [etf])

        if not keywords:
            return pd.DataFrame(columns=["date", "title", "url"])

        # Break into monthly chunks to avoid API limits
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        all_news = []
        current = start

        while current < end:
            # Get end of current month
            month_end = min(current + pd.DateOffset(months=1) - pd.DateOffset(days=1), end)

            news = self.fetch_news(
                keywords=keywords,
                start_date=current.strftime("%Y-%m-%d"),
                end_date=month_end.strftime("%Y-%m-%d"),
                max_records=max_records_per_month
            )

            if not news.empty:
                news["etf"] = etf
                all_news.append(news)

            current = month_end + pd.DateOffset(days=1)

        if not all_news:
            return pd.DataFrame(columns=["date", "title", "url", "etf"])

        return pd.concat(all_news, ignore_index=True)


class SentimentCache:
    """
    Caches sentiment scores to avoid recomputation during backtesting.
    """

    def __init__(self, cache_dir: str = "data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "sentiment_cache.csv"

    def load(self) -> Optional[pd.DataFrame]:
        """Load cached sentiment data."""
        if self.cache_file.exists():
            df = pd.read_csv(self.cache_file, index_col=0, parse_dates=True)
            print(f"Loaded cached sentiment data: {len(df)} days")
            return df
        return None

    def save(self, df: pd.DataFrame):
        """Save sentiment data to cache."""
        df.to_csv(self.cache_file)
        print(f"Saved sentiment cache to {self.cache_file}")

    def is_cached(self) -> bool:
        """Check if cache exists."""
        return self.cache_file.exists()


def compute_historical_sentiment(
    start_date: str,
    end_date: str,
    tickers: List[str] = None,
    cache_dir: str = "data",
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Compute historical sentiment for ETFs using GDELT + FinBERT.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        tickers: List of ETF tickers (uses all if None)
        cache_dir: Cache directory
        force_refresh: If True, recompute even if cached

    Returns:
        DataFrame with columns: {ticker}_sentiment for each ticker
        Index is DatetimeIndex with trading days
    """
    if tickers is None:
        tickers = list(ETF_KEYWORDS.keys())

    cache = SentimentCache(cache_dir)

    # Try to load from cache
    if not force_refresh and cache.is_cached():
        cached = cache.load()
        if cached is not None:
            # Check if we have all required tickers
            required_cols = [f"{t}_sentiment" for t in tickers]
            if all(col in cached.columns for col in required_cols):
                # Filter to date range
                cached = cached.loc[start_date:end_date]
                if len(cached) > 0:
                    return cached[required_cols]

    print(f"Computing historical sentiment from {start_date} to {end_date}...")
    print(f"ETFs: {tickers}")
    print("This may take a while for large date ranges...")

    # Initialize components
    gdelt = GDELTNewsLoader(cache_dir)
    finbert = FinBertSentiment()

    if not gdelt.is_available():
        print("GDELT not available. Returning zeros.")
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        df = pd.DataFrame(index=dates)
        for ticker in tickers:
            df[f"{ticker}_sentiment"] = 0.0
        return df

    all_sentiment = {}

    for ticker in tickers:
        print(f"\nProcessing {ticker}...")

        # Fetch news
        news = gdelt.fetch_news_for_etf(
            etf=ticker,
            start_date=start_date,
            end_date=end_date
        )

        if news.empty:
            print(f"  No news found for {ticker}")
            all_sentiment[f"{ticker}_sentiment"] = pd.Series(dtype=float)
            continue

        print(f"  Found {len(news)} articles, scoring with FinBERT...")

        # Score all titles
        titles = news["title"].fillna("").tolist()
        scores = finbert.score_batch(titles)
        news["sentiment"] = scores

        # Aggregate by date
        daily_sentiment = news.groupby("date")["sentiment"].mean()
        daily_sentiment.index = pd.to_datetime(daily_sentiment.index)

        all_sentiment[f"{ticker}_sentiment"] = daily_sentiment
        print(f"  Got sentiment for {len(daily_sentiment)} days")

    # Combine into DataFrame
    df = pd.DataFrame(all_sentiment)

    # Create full business day index and forward-fill gaps
    full_index = pd.date_range(start=start_date, end=end_date, freq="B")
    df = df.reindex(full_index)
    df = df.fillna(method="ffill").fillna(0)

    # Save to cache
    cache.save(df)

    return df


def compute_sentiment_features(
    sentiment_df: pd.DataFrame,
    ma_window: int = 5
) -> pd.DataFrame:
    """
    Compute additional sentiment features from raw sentiment.

    Args:
        sentiment_df: DataFrame with {ticker}_sentiment columns
        ma_window: Window for moving average

    Returns:
        DataFrame with additional features:
        - {ticker}_sentiment_ma{window}: Moving average
        - {ticker}_sentiment_momentum: Change in sentiment
    """
    features = sentiment_df.copy()

    for col in sentiment_df.columns:
        if col.endswith("_sentiment"):
            base = col.replace("_sentiment", "")

            # Moving average
            features[f"{base}_sentiment_ma{ma_window}"] = (
                sentiment_df[col].rolling(window=ma_window, min_periods=1).mean()
            )

            # Momentum (change over ma_window days)
            features[f"{base}_sentiment_momentum"] = (
                sentiment_df[col] - sentiment_df[col].shift(ma_window)
            ).fillna(0)

    return features


if __name__ == "__main__":
    # Test the sentiment pipeline
    print("Testing sentiment pipeline...")

    # Test FinBERT
    finbert = FinBertSentiment()

    test_texts = [
        "Stock market rallies to new highs on strong earnings",
        "Market crashes amid recession fears",
        "Federal Reserve holds interest rates steady",
    ]

    print("\nFinBERT test:")
    for text in test_texts:
        score = finbert.score(text)
        print(f"  Score: {score:+.3f} | {text[:50]}...")

    # Test GDELT (if available)
    print("\nTesting GDELT...")
    gdelt = GDELTNewsLoader()

    if gdelt.is_available():
        # Test with recent date range
        end = datetime.today()
        start = end - timedelta(days=7)

        news = gdelt.fetch_news(
            keywords=["stock market", "S&P 500"],
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            max_records=10
        )

        print(f"  Found {len(news)} articles")
        if not news.empty:
            print(news.head())
    else:
        print("  GDELT not available (install gdeltdoc)")

    print("\nDone.")
