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
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Download ETF adjusted close prices.

        Args:
            tickers: List of ETF ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_refresh: If True, re-download even if cached

        Returns:
            DataFrame with DatetimeIndex and columns for each ticker
        """
        cache_file = self.cache_dir / f"prices_{'_'.join(tickers)}_{start_date}_{end_date}.csv"

        # Load from cache if available
        if cache_file.exists() and not force_refresh:
            print(f"Loading cached data from {cache_file}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df

        print(f"Downloading data for {tickers} from {start_date} to {end_date}...")

        # Download data using yfinance
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True  # Use adjusted prices (handles splits and dividends)
        )

        # Handle single ticker case (yfinance returns different structure)
        if len(tickers) == 1:
            prices = data['Close'].to_frame()
            prices.columns = tickers
        else:
            prices = data['Close']

        # Remove any NaN rows (market holidays, missing data)
        prices = prices.dropna()

        # Save to cache
        prices.to_csv(cache_file)
        print(f"Cached data to {cache_file}")

        return prices

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
    cache_dir: str = "data"
) -> pd.DataFrame:
    """
    Convenience function to load default ETF set (VTI, BND, QQQ).

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
        cache_dir: Cache directory

    Returns:
        DataFrame with price data
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    loader = ETFDataLoader(cache_dir=cache_dir)
    tickers = ['VTI', 'BND', 'QQQ']

    prices = loader.download_etfs(tickers, start_date, end_date)

    return prices


if __name__ == "__main__":
    # Example usage
    loader = ETFDataLoader()

    # Download default ETFs
    prices = load_default_etfs()

    # Show summary
    print("\n" + "="*80)
    print("ETF Data Summary")
    print("="*80)
    print(loader.get_data_summary(prices))

    # Split data
    train, val, test = loader.split_train_val_test(prices)

    print("\n" + "="*80)
    print("Training Set Summary")
    print("="*80)
    print(loader.get_data_summary(train))
