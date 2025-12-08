"""
Sentiment analysis for ETF portfolio optimization.

Uses GDELT BigQuery for historical news data (2017+) and FinBERT for sentiment scoring.
Supports caching to avoid repeated API calls during backtesting.

Pipeline:
1. Query GDELT events/GKG from BigQuery for relevant articles
2. Fetch headlines from article URLs
3. Score headlines with FinBERT
4. Aggregate to daily sentiment features per ETF
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=FutureWarning)


# Mapping from ETF tickers to GDELT search terms
# Optimized for better coverage and relevance
ETF_SEARCH_TERMS = {
    # US Equity ETFs - Large Cap
    "SPY": [
        # S&P 500 variants (handles all spacing/punctuation)
        "S&P 500", "SP500", "S&P500", "S&P-500", "SPX",
        "Standard & Poor's", "Standard and Poor",
        # Market terms
        "Wall Street", "stock market rally", "stock market surge",
        "equity market", "blue chip", "large cap stocks"
    ],
    "QQQ": [
        # NASDAQ variants
        "NASDAQ", "Nasdaq", "NASDAQ-100", "NASDAQ 100", "NDX",
        # Tech sector
        "tech stocks", "technology shares", "technology sector",
        "Silicon Valley", "big tech", "mega cap tech",
        # Major holdings (FAANG+)
        "Apple", "Microsoft", "Amazon", "Tesla", "Google", "Meta",
        "Nvidia", "Facebook"
    ],
    "VTI": [
        "US equities", "American stocks", "total market",
        "broad market", "stock index", "equity index"
    ],

    # US Small Cap
    "IWM": [
        # Russell 2000 variants
        "Russell 2000", "Russell2000", "RUT", "RTY",
        # Small cap terms
        "small cap", "small-cap", "small cap stocks", "smallcap",
        "mid cap", "mid-cap", "midcap",
        # Style
        "growth stocks", "value stocks", "small cap index",
        # Related
        "regional banks", "smaller companies", "small business"
    ],

    # Bond ETFs
    "TLT": [
        "Treasury bonds", "government bonds", "long-term bonds",
        "10-year yield", "30-year yield", "bond yields",
        "Federal Reserve", "interest rates", "rate hike",
        "inflation"
    ],
    "BND": [
        "bond market", "fixed income", "corporate bonds",
        "investment grade", "bond yields", "credit market",
        "debt market"
    ],

    # Commodities
    "GLD": [
        # Gold price terms
        "gold price", "gold prices", "gold rally", "gold surge",
        "gold falls", "gold drops", "gold market",
        # Gold products
        "gold bullion", "gold bars", "physical gold",
        # Investment terms
        "gold ETF", "gold investment", "gold assets",
        # Related terms
        "precious metals", "safe haven", "haven asset",
        "gold miners", "mining stocks", "Newmont", "Barrick",
        # Variants
        "XAU", "gold futures", "spot gold"
    ],

    # International - Developed Markets
    "VEA": [
        "European stocks", "European markets", "FTSE",
        "DAX", "CAC 40", "Japan stocks", "Nikkei",
        "developed markets", "international equities",
        "eurozone", "UK stocks", "Germany economy"
    ],

    # International - Emerging Markets
    "VWO": [
        "emerging markets", "emerging economies",
        "China stocks", "Chinese economy", "Shanghai",
        "India stocks", "Brazil stocks", "BRICS",
        "developing markets", "frontier markets",
        "EM equities"
    ],

    # Sector - Energy
    "XLE": [
        "oil price", "oil prices", "crude oil",
        "energy sector", "energy stocks", "petroleum",
        "Exxon", "Chevron", "OPEC", "natural gas",
        "energy crisis", "oil production", "shale"
    ],
}

# Finance-focused domains for filtering
# Based on 2025 research of top financial news sources
FINANCE_DOMAINS = [
    # Tier 1: Premier Financial News (Institutional Grade)
    "bloomberg.com",           # Bloomberg - Professional standard
    "wsj.com",                 # Wall Street Journal - 39 Pulitzer Prizes
    "reuters.com",             # Reuters - Real-time global news
    "ft.com",                  # Financial Times - International authority
    "financialtimes.com",      # FT alternate domain

    # Tier 2: Major Business News Networks
    "cnbc.com",                # CNBC - Real-time market coverage
    "barrons.com",             # Barron's - Investment analysis
    "economist.com",           # The Economist - Economic policy depth
    "forbes.com",              # Forbes - Business & markets
    "fortune.com",             # Fortune - Corporate news

    # Tier 3: Retail Investor Platforms (High Volume)
    "finance.yahoo.com",       # Yahoo Finance - Most visited finance site
    "marketwatch.com",         # MarketWatch - Dow Jones property
    "seekingalpha.com",        # Seeking Alpha - Investment research
    "fool.com",                # Motley Fool - Retail investor focus
    "thestreet.com",           # TheStreet - Jim Cramer's site
    "investopedia.com",        # Investopedia - Education + news

    # Tier 4: Trading & Analysis Platforms
    "benzinga.com",            # Benzinga - Real-time trading news
    "investing.com",           # Investing.com - Global markets
    "morningstar.com",         # Morningstar - Fund analysis
    "zerohedge.com",           # Zero Hedge - Alternative perspective

    # Tier 5: Exchange & Market Data Sites
    "nasdaq.com",              # NASDAQ official
    "nyse.com",                # NYSE official

    # Tier 6: General News Business Sections
    "businessinsider.com",     # Business Insider
    "cnn.com/business",        # CNN Business
    "bbc.com/business",        # BBC Business
    "theguardian.com/business", # Guardian Business
    "foxbusiness.com",         # Fox Business

    # Tier 7: International Finance
    "nikkei.com",              # Nikkei Asia
    "scmp.com",                # South China Morning Post
]


class FinBertSentiment:
    """
    FinBERT-based sentiment scorer for financial text.

    Returns sentiment scores in range [-1, 1] where:
    - Positive values indicate bullish/positive sentiment
    - Negative values indicate bearish/negative sentiment
    - Zero indicates neutral sentiment
    """

    def __init__(self, device: str = None, model_path: str = None):
        """
        Initialize FinBERT model.

        Args:
            device: 'cuda', 'mps', or 'cpu'. Auto-detected if None.
            model_path: Path to fine-tuned model. If None, uses pre-trained FinBERT.
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.torch = torch

        # Load model (fine-tuned or pre-trained)
        if model_path is not None and os.path.exists(model_path):
            print(f"Loading fine-tuned FinBERT model from {model_path} on {device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.is_finetuned = True
        else:
            print(f"Loading pre-trained FinBERT model on {device}...")
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.is_finetuned = False

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

        text = text[:512]

        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with self.torch.no_grad():
            logits = self.model(**tokens).logits

        probs = self.torch.softmax(logits, dim=1).cpu().numpy()[0]
        # FinBERT outputs: [negative, neutral, positive]
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

            with self.torch.no_grad():
                logits = self.model(**tokens).logits

            probs = self.torch.softmax(logits, dim=1).cpu().numpy()
            batch_scores = probs[:, 2] - probs[:, 0]
            scores.extend(batch_scores.tolist())

        return scores


class GDELTBigQueryLoader:
    """
    Load news data from GDELT using Google BigQuery.

    GDELT BigQuery tables:
    - gdelt-bq.gdeltv2.events: Event records with actors, URLs
    - gdelt-bq.gdeltv2.gkg: Global Knowledge Graph with themes, tone

    Requires Google Cloud authentication:
    - Set GOOGLE_APPLICATION_CREDENTIALS environment variable, or
    - Run `gcloud auth application-default login`
    """

    def __init__(self, project_id: str = None):
        """
        Initialize BigQuery client.

        Args:
            project_id: Google Cloud project ID (optional, uses default if None)
        """
        self._available = False
        self.client = None

        try:
            from google.cloud import bigquery
            self.client = bigquery.Client(project=project_id)
            self._available = True
            print("BigQuery client initialized successfully")
        except Exception as e:
            print(f"BigQuery initialization failed: {e}")
            print("To use GDELT BigQuery, you need:")
            print("  1. pip install google-cloud-bigquery")
            print("  2. Set up Google Cloud authentication:")
            print("     - Run: gcloud auth application-default login")
            print("     - Or set GOOGLE_APPLICATION_CREDENTIALS env var")

    def is_available(self) -> bool:
        return self._available

    def query_events(
        self,
        start_date: str,
        end_date: str,
        search_terms: List[str] = None,
        domains: List[str] = None,
        max_results: int = 10000
    ) -> pd.DataFrame:
        """
        Query GDELT events table for news URLs.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            search_terms: List of terms to search for in actor names/URLs
            domains: List of domains to filter to (e.g., reuters.com)
            max_results: Maximum number of results

        Returns:
            DataFrame with columns: date, url, goldstein_scale, num_articles
        """
        if not self._available:
            return pd.DataFrame(columns=["date", "url", "goldstein_scale", "num_articles"])

        # Convert dates to GDELT format (YYYYMMDD integer)
        start_int = int(pd.to_datetime(start_date).strftime("%Y%m%d"))
        end_int = int(pd.to_datetime(end_date).strftime("%Y%m%d"))

        # Build WHERE clause for search terms
        where_clauses = [f"SQLDATE BETWEEN {start_int} AND {end_int}"]

        if search_terms:
            term_conditions = []
            for term in search_terms:
                # Normalize term to catch variants (e.g., "S&P 500" -> "s%p%500")
                # Replace special chars and spaces with % wildcards for flexible matching
                normalized = term.lower()
                # Replace common special chars with wildcards
                for char in ['&', '-', '_', '.', ',', "'", '"']:
                    normalized = normalized.replace(char, '%')
                # Replace spaces with wildcards
                normalized = normalized.replace(' ', '%')
                # Remove consecutive % wildcards
                while '%%' in normalized:
                    normalized = normalized.replace('%%', '%')

                term_conditions.append(f"LOWER(SOURCEURL) LIKE '%{normalized}%'")
            where_clauses.append(f"({' OR '.join(term_conditions)})")

        if domains:
            domain_conditions = [f"SOURCEURL LIKE '%{d}%'" for d in domains]
            where_clauses.append(f"({' OR '.join(domain_conditions)})")

        query = f"""
        SELECT
            DATE(PARSE_TIMESTAMP('%Y%m%d', CAST(SQLDATE AS STRING))) AS date,
            SOURCEURL AS url,
            GoldsteinScale AS goldstein_scale,
            NumArticles AS num_articles
        FROM `gdelt-bq.gdeltv2.events`
        WHERE {' AND '.join(where_clauses)}
            AND SOURCEURL IS NOT NULL
            AND SOURCEURL != ''
        ORDER BY SQLDATE DESC, NumArticles DESC
        LIMIT {max_results}
        """

        try:
            df = self.client.query(query).to_dataframe()
            print(f"  Retrieved {len(df)} events from BigQuery")
            return df
        except Exception as e:
            print(f"BigQuery query failed: {e}")
            return pd.DataFrame(columns=["date", "url", "goldstein_scale", "num_articles"])

    def query_gkg_tone(
        self,
        start_date: str,
        end_date: str,
        themes: List[str] = None,
        max_results: int = 10000
    ) -> pd.DataFrame:
        """
        Query GDELT GKG (Global Knowledge Graph) for built-in tone scores.

        GKG provides pre-computed tone scores, which can be faster than
        fetching headlines and running FinBERT.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            themes: GDELT themes to filter by (e.g., ECON_STOCKMARKET)
            max_results: Maximum results

        Returns:
            DataFrame with columns: date, url, tone, themes
        """
        if not self._available:
            return pd.DataFrame(columns=["date", "url", "tone", "themes"])

        start_int = int(pd.to_datetime(start_date).strftime("%Y%m%d"))
        end_int = int(pd.to_datetime(end_date).strftime("%Y%m%d"))

        where_clauses = [f"CAST(SUBSTR(CAST(DATE AS STRING), 1, 8) AS INT64) BETWEEN {start_int} AND {end_int}"]

        if themes:
            theme_conditions = [f"Themes LIKE '%{t}%'" for t in themes]
            where_clauses.append(f"({' OR '.join(theme_conditions)})")

        query = f"""
        SELECT
            PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS date,
            DocumentIdentifier AS url,
            CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS tone,
            Themes AS themes
        FROM `gdelt-bq.gdeltv2.gkg`
        WHERE {' AND '.join(where_clauses)}
            AND V2Tone IS NOT NULL
        ORDER BY date DESC
        LIMIT {max_results}
        """

        try:
            df = self.client.query(query).to_dataframe()
            print(f"  Retrieved {len(df)} GKG records from BigQuery")
            return df
        except Exception as e:
            print(f"BigQuery GKG query failed: {e}")
            return pd.DataFrame(columns=["date", "url", "tone", "themes"])


class HeadlineScraper:
    """
    Scrape headlines from news article URLs.

    Uses parallel requests with rate limiting to fetch headlines efficiently.
    """

    def __init__(self, max_workers: int = 10, timeout: int = 5):
        """
        Initialize scraper.

        Args:
            max_workers: Maximum concurrent requests
            timeout: Request timeout in seconds
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; research bot)'
        })

    def fetch_headline(self, url: str) -> Optional[str]:
        """
        Fetch headline from a single URL.

        Args:
            url: Article URL

        Returns:
            Headline text or None if failed
        """
        try:
            from bs4 import BeautifulSoup

            response = self.session.get(url, timeout=self.timeout)
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.content, 'html.parser')

            # Try different headline selectors
            headline = None

            # Try <title> tag
            if soup.title:
                headline = soup.title.string

            # Try common headline tags
            if not headline:
                for selector in ['h1', 'article h1', '.headline', '.article-title']:
                    elem = soup.select_one(selector)
                    if elem:
                        headline = elem.get_text()
                        break

            # Try meta tags
            if not headline:
                meta = soup.find('meta', property='og:title')
                if meta:
                    headline = meta.get('content')

            if headline:
                return headline.strip()[:500]  # Limit length

            return None

        except Exception:
            return None

    def fetch_headlines_batch(
        self,
        urls: List[str],
        progress_callback=None
    ) -> Dict[str, str]:
        """
        Fetch headlines from multiple URLs in parallel.

        Args:
            urls: List of URLs
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping URL to headline
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {executor.submit(self.fetch_headline, url): url for url in urls}

            completed = 0
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    headline = future.result()
                    if headline:
                        results[url] = headline
                except Exception:
                    pass

                completed += 1
                if progress_callback and completed % 100 == 0:
                    progress_callback(completed, len(urls))

        return results


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


def compute_sentiment_bigquery(
    start_date: str,
    end_date: str,
    tickers: List[str] = None,
    cache_dir: str = "data",
    force_refresh: bool = False,
    use_gkg_tone: bool = True,
    max_articles_per_day: int = 100,
    finbert_model_path: str = None,
) -> pd.DataFrame:
    """
    Compute sentiment using GDELT BigQuery + FinBERT.

    Pipeline:
    1. Query GDELT events/GKG from BigQuery
    2. Optionally fetch headlines and score with FinBERT
    3. Aggregate to daily sentiment per ETF

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        tickers: List of ETF tickers (uses all if None)
        cache_dir: Cache directory
        force_refresh: If True, recompute even if cached
        use_gkg_tone: If True, use GDELT's built-in tone scores (faster)
                      If False, fetch headlines and run FinBERT (more accurate)
        max_articles_per_day: Max articles to process per day
        finbert_model_path: Path to fine-tuned FinBERT model (optional)

    Returns:
        DataFrame with columns: {ticker}_sentiment for each ticker
    """
    if tickers is None:
        tickers = list(ETF_SEARCH_TERMS.keys())

    cache = SentimentCache(cache_dir)

    # Try to load from cache
    if not force_refresh and cache.is_cached():
        cached = cache.load()
        if cached is not None:
            required_cols = [f"{t}_sentiment" for t in tickers]
            if all(col in cached.columns for col in required_cols):
                cached = cached.loc[start_date:end_date]
                if len(cached) > 0:
                    return cached[required_cols]

    print(f"Computing sentiment from {start_date} to {end_date}...")
    print(f"ETFs: {tickers}")

    # Initialize BigQuery loader
    bq = GDELTBigQueryLoader()

    if not bq.is_available():
        print("BigQuery not available. Returning zeros.")
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        df = pd.DataFrame(index=dates)
        for ticker in tickers:
            df[f"{ticker}_sentiment"] = 0.0
        return df

    all_sentiment = {}

    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        search_terms = ETF_SEARCH_TERMS.get(ticker, [ticker])

        if use_gkg_tone:
            # Use GDELT's built-in tone scores (faster)
            # Map ETFs to GDELT themes
            theme_map = {
                "SPY": ["ECON_STOCKMARKET", "ECON_WORLDCURRENCIES"],
                "QQQ": ["ECON_STOCKMARKET", "TAX_FNCACT_TECH"],
                "TLT": ["ECON_INTEREST_RATES", "CRISISLEX_CRISISLANG"],
                "GLD": ["ECON_WORLDCURRENCIES", "ECON_INFLATION"],
            }
            themes = theme_map.get(ticker, ["ECON_STOCKMARKET"])

            data = bq.query_gkg_tone(
                start_date=start_date,
                end_date=end_date,
                themes=themes,
                max_results=50000
            )

            if data.empty:
                all_sentiment[f"{ticker}_sentiment"] = pd.Series(dtype=float)
                continue

            # Aggregate tone by date
            # GDELT tone: positive = good, negative = bad (already correct direction)
            # Normalize to [-1, 1] range (GDELT tone is typically -10 to +10)
            daily = data.groupby("date")["tone"].mean() / 10.0
            daily = daily.clip(-1, 1)

        else:
            # Fetch headlines and score with FinBERT (more accurate but slower)
            data = bq.query_events(
                start_date=start_date,
                end_date=end_date,
                search_terms=search_terms,
                domains=FINANCE_DOMAINS,
                max_results=50000
            )

            if data.empty:
                all_sentiment[f"{ticker}_sentiment"] = pd.Series(dtype=float)
                continue

            # Sample top articles per day
            data = data.groupby("date").head(max_articles_per_day)

            print(f"  Fetching headlines for {len(data)} articles...")
            scraper = HeadlineScraper(max_workers=20)
            headlines = scraper.fetch_headlines_batch(data["url"].tolist())

            if not headlines:
                all_sentiment[f"{ticker}_sentiment"] = pd.Series(dtype=float)
                continue

            # Map headlines back to dates
            data["headline"] = data["url"].map(headlines)
            data = data.dropna(subset=["headline"])

            if data.empty:
                all_sentiment[f"{ticker}_sentiment"] = pd.Series(dtype=float)
                continue

            print(f"  Scoring {len(data)} headlines with FinBERT...")
            finbert = FinBertSentiment(model_path=finbert_model_path)
            data["sentiment"] = finbert.score_batch(data["headline"].tolist())

            # Aggregate by date
            daily = data.groupby("date")["sentiment"].mean()

        daily.index = pd.to_datetime(daily.index)
        all_sentiment[f"{ticker}_sentiment"] = daily
        print(f"  Got sentiment for {len(daily)} days")

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


# Backward compatibility alias
def compute_historical_sentiment(
    start_date: str,
    end_date: str,
    tickers: List[str] = None,
    cache_dir: str = "data",
    force_refresh: bool = False
) -> pd.DataFrame:
    """Alias for compute_sentiment_bigquery with default settings."""
    return compute_sentiment_bigquery(
        start_date=start_date,
        end_date=end_date,
        tickers=tickers,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        use_gkg_tone=True,  # Use GKG tone for speed
    )


if __name__ == "__main__":
    print("Testing sentiment pipeline...")

    # Test FinBERT
    print("\n1. Testing FinBERT...")
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

    # Test BigQuery
    print("\n2. Testing GDELT BigQuery...")
    bq = GDELTBigQueryLoader()

    if bq.is_available():
        # Test with recent date range
        end = datetime.today()
        start = end - timedelta(days=7)

        print(f"  Querying events from {start.date()} to {end.date()}...")
        events = bq.query_events(
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            search_terms=["stock market"],
            domains=["reuters.com", "bloomberg.com"],
            max_results=10
        )

        if not events.empty:
            print(f"  Found {len(events)} events")
            print(events[["date", "url"]].head())
        else:
            print("  No events found")

        print("\n  Querying GKG tone...")
        gkg = bq.query_gkg_tone(
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            themes=["ECON_STOCKMARKET"],
            max_results=10
        )

        if not gkg.empty:
            print(f"  Found {len(gkg)} GKG records")
            print(f"  Average tone: {gkg['tone'].mean():.2f}")
        else:
            print("  No GKG records found")
    else:
        print("  BigQuery not available")

    print("\nDone.")
