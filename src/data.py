"""
Data pipeline for ETF historical prices.

Handles downloading, caching, and preprocessing of ETF data with strict
no-look-ahead bias guarantees.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf


class ETFDataLoader:
    """
    Download and manage ETF price data with caching.

    Ensures reproducibility through caching and strict date handling.
    """

    def __init__(self, cache_dir: str = "data"):
        """
        Initialize data loader.

        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def download_etfs(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
        ohlcv: bool = True
    ) -> pd.DataFrame:
        """
        Download ETF OHLCV data (Open, High, Low, Close, Volume).

        Args:
            tickers: List of ETF ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_refresh: If True, re-download even if cached
            ohlcv: If True, return OHLCV data; if False, return Close only (legacy)

        Returns:
            DataFrame with DatetimeIndex and columns like {ticker}_Close, {ticker}_Volume, etc.
        """
        data_type = "ohlcv" if ohlcv else "prices"
        cache_file = self.cache_dir / f"{data_type}_{'_'.join(tickers)}_{start_date}_{end_date}.csv"

        # Load from cache if available
        if cache_file.exists() and not force_refresh:
            print(f"Loading cached data from {cache_file}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df

        print(f"Downloading {data_type.upper()} data for {tickers} from {start_date} to {end_date}...")

        # Download data using yfinance
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True  # Use adjusted prices (handles splits and dividends)
        )

        if not ohlcv:
            # Legacy behavior: return Close prices only
            if len(tickers) == 1:
                prices = data['Close'].to_frame()
                prices.columns = tickers
            else:
                prices = data['Close']
            prices = prices.dropna()
            prices.to_csv(cache_file)
            print(f"Cached data to {cache_file}")
            return prices

        # New behavior: return OHLCV with flattened column names
        if len(tickers) == 1:
            # Single ticker: data has columns like Open, High, Low, Close, Volume
            ohlcv_data = pd.DataFrame()
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in data.columns:
                    ohlcv_data[f"{tickers[0]}_{col}"] = data[col]
        else:
            # Multiple tickers: data has MultiIndex columns (metric, ticker)
            # Reshape to {ticker}_{metric} format
            ohlcv_data = pd.DataFrame()
            for metric in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if metric in data.columns.levels[0]:
                    for ticker in tickers:
                        if ticker in data[metric].columns:
                            ohlcv_data[f"{ticker}_{metric}"] = data[metric][ticker]

        # Remove any NaN rows (market holidays, missing data)
        ohlcv_data = ohlcv_data.dropna()

        # Save to cache
        ohlcv_data.to_csv(cache_file)
        print(f"Cached {len(ohlcv_data)} days of OHLCV data to {cache_file}")

        return ohlcv_data

    def download_market_indicators(
        self,
        start_date: str,
        end_date: str,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Download market-wide indicators: VIX, Treasury yields, etc.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_refresh: If True, re-download even if cached

        Returns:
            DataFrame with DatetimeIndex and columns: VIX, TNX_10Y, IRX_3M, DGS2_2Y
        """
        cache_file = self.cache_dir / f"indicators_{start_date}_{end_date}.csv"

        # Load from cache if available
        if cache_file.exists() and not force_refresh:
            print(f"Loading cached indicators from {cache_file}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df

        print(f"Downloading market indicators from {start_date} to {end_date}...")

        # Download VIX and yields
        indicators = {}
        indicator_tickers = {
            'VIX': '^VIX',       # CBOE Volatility Index
            'TNX_10Y': '^TNX',   # 10-Year Treasury Yield
            'IRX_3M': '^IRX',    # 3-Month Treasury Yield
        }

        for name, ticker in indicator_tickers.items():
            try:
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                if not data.empty:
                    # Use Close price for the indicator value
                    indicators[name] = data['Close']
                    print(f"  Downloaded {name} ({ticker}): {len(data)} days")
                else:
                    print(f"  WARNING: No data for {name} ({ticker})")
            except Exception as e:
                print(f"  ERROR downloading {name} ({ticker}): {e}")

        if not indicators:
            raise ValueError("Failed to download any market indicators")

        # Combine into single DataFrame with proper index alignment
        # First, find common index across all indicators
        common_index = None
        for name, series in indicators.items():
            if common_index is None:
                common_index = series.index
            else:
                common_index = common_index.union(series.index)

        # Create DataFrame with common index
        indicators_df = pd.DataFrame(index=common_index.sort_values())

        # Add each indicator, reindexing to common dates
        for name, series in indicators.items():
            indicators_df[name] = series

        # Forward-fill to handle weekends/holidays (never backfill to avoid look-ahead)
        indicators_df = indicators_df.ffill()

        # Remove any remaining NaN rows at the beginning
        indicators_df = indicators_df.dropna()

        # Save to cache
        indicators_df.to_csv(cache_file)
        print(f"Cached {len(indicators_df)} days of indicators to {cache_file}")

        return indicators_df

    def split_train_val_test(
        self,
        prices: pd.DataFrame,
        train_end: str = "2021-12-31",
        val_end: str = "2022-12-31"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets by date.

        Args:
            prices: Full price DataFrame
            train_end: Last date of training period (inclusive)
            val_end: Last date of validation period (inclusive)

        Returns:
            Tuple of (train_prices, val_prices, test_prices)
        """
        train = prices[:train_end]
        val = prices[train_end:val_end].iloc[1:]  # Exclude overlap
        test = prices[val_end:].iloc[1:]  # Exclude overlap

        print(f"\nData split:")
        print(f"  Train: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} days)")
        print(f"  Val:   {val.index[0].date()} to {val.index[-1].date()} ({len(val)} days)")
        print(f"  Test:  {test.index[0].date()} to {test.index[-1].date()} ({len(test)} days)")

        return train, val, test

    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily percentage returns.

        Args:
            prices: DataFrame of prices

        Returns:
            DataFrame of daily returns (first row will be NaN)
        """
        returns = prices.pct_change()
        return returns

    def get_data_summary(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for price data.

        Args:
            prices: DataFrame of prices

        Returns:
            DataFrame with summary statistics
        """
        returns = self.calculate_returns(prices)

        summary = pd.DataFrame({
            'Start Date': prices.index[0],
            'End Date': prices.index[-1],
            'Days': len(prices),
            'Mean Daily Return (%)': returns.mean() * 100,
            'Daily Volatility (%)': returns.std() * 100,
            'Annualized Return (%)': returns.mean() * 252 * 100,
            'Annualized Volatility (%)': returns.std() * np.sqrt(252) * 100,
            'Total Return (%)': (prices.iloc[-1] / prices.iloc[0] - 1) * 100,
            'Min Price': prices.min(),
            'Max Price': prices.max(),
        })

        return summary.T


def load_default_etfs(
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    cache_dir: str = "data",
    expanded: bool = True,
    ohlcv: bool = True,
    include_indicators: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Convenience function to load default ETF set with OHLCV and market indicators.

    Core ETFs (always included):
    - SPY: S&P 500 (US large-cap stocks)
    - QQQ: NASDAQ-100 (US tech stocks)
    - VTI: Vanguard Total Stock Market (broad US stocks)
    - TLT: iShares 20+ Year Treasury Bond (long-term US bonds)
    - BND: Vanguard Total Bond Market (broad US bonds)
    - GLD: SPDR Gold Trust (gold)

    Expanded ETFs (when expanded=True):
    - VEA: Vanguard FTSE Developed Markets (international developed)
    - VWO: Vanguard FTSE Emerging Markets (emerging markets)
    - IWM: iShares Russell 2000 (US small-cap)
    - XLE: Energy Select Sector SPDR (energy sector)

    Market Indicators (when include_indicators=True):
    - VIX: CBOE Volatility Index
    - TNX_10Y: 10-Year Treasury Yield
    - IRX_3M: 3-Month Treasury Yield

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
        cache_dir: Cache directory
        expanded: If True, include additional diversifying ETFs (10 total)
        ohlcv: If True, return OHLCV data; if False, return Close only (legacy)
        include_indicators: If True, also download and return market indicators

    Returns:
        Tuple of (ohlcv_data, indicators_data) where indicators_data is None if not requested
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    loader = ETFDataLoader(cache_dir=cache_dir)

    # Core ETFs
    tickers = ['SPY', 'QQQ', 'VTI', 'TLT', 'BND', 'GLD']

    # Add diversifying ETFs for expanded universe
    if expanded:
        tickers.extend(['VEA', 'VWO', 'IWM', 'XLE'])

    # Download ETF data (OHLCV or Close only)
    etf_data = loader.download_etfs(tickers, start_date, end_date, ohlcv=ohlcv)

    # Download market indicators if requested
    indicators = None
    if include_indicators:
        try:
            indicators = loader.download_market_indicators(start_date, end_date)
        except Exception as e:
            print(f"WARNING: Failed to download market indicators: {e}")
            print("Continuing without indicators...")

    return etf_data, indicators


if __name__ == "__main__":
    # Example usage
    loader = ETFDataLoader()

    # Download default ETFs with OHLCV and indicators
    print("Testing new OHLCV + Indicators functionality...")
    ohlcv_data, indicators = load_default_etfs(
        start_date="2020-01-01",
        end_date="2023-12-31"
    )

    # Show data structure
    print("\n" + "="*80)
    print("OHLCV Data")
    print("="*80)
    print(f"Shape: {ohlcv_data.shape}")
    print(f"Columns: {list(ohlcv_data.columns[:5])}... (showing first 5)")
    print(f"\nSample data:")
    print(ohlcv_data.head())

    if indicators is not None:
        print("\n" + "="*80)
        print("Market Indicators")
        print("="*80)
        print(f"Shape: {indicators.shape}")
        print(f"Columns: {list(indicators.columns)}")
        print(f"\nSample data:")
        print(indicators.head())

    # Extract Close prices for legacy compatibility
    close_cols = [col for col in ohlcv_data.columns if col.endswith('_Close')]
    prices = ohlcv_data[close_cols].copy()
    prices.columns = [col.replace('_Close', '') for col in close_cols]

    print("\n" + "="*80)
    print("Extracted Close Prices (legacy format)")
    print("="*80)
    print(prices.head())
