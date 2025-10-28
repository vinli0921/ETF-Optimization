"""
Feature engineering for ETF portfolio optimization.

All features are computed with strict no-look-ahead bias:
- At time t, only use data from periods strictly before t
- Rolling windows use past data only
"""

import pandas as pd
import numpy as np
from typing import Optional


class FeatureEngineer:
    """
    Compute features from price/return data for portfolio optimization.

    All features maintain temporal consistency for backtesting.
    """

    def __init__(self, lookback_window: int = 30):
        """
        Initialize feature engineer.

        Args:
            lookback_window: Number of days to use for rolling statistics
        """
        self.lookback_window = lookback_window

    def compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute daily percentage returns.

        Args:
            prices: DataFrame of prices with DatetimeIndex

        Returns:
            DataFrame of returns (first row will be NaN)
        """
        return prices.pct_change()

    def compute_rolling_volatility(
        self,
        returns: pd.DataFrame,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute rolling standard deviation of returns (volatility).

        Args:
            returns: DataFrame of returns
            window: Rolling window size (defaults to self.lookback_window)

        Returns:
            DataFrame of rolling volatility (annualized)
        """
        window = window or self.lookback_window

        # Rolling standard deviation
        rolling_vol = returns.rolling(window=window, min_periods=window).std()

        # Annualize: multiply by sqrt(252)
        rolling_vol = rolling_vol * np.sqrt(252)

        return rolling_vol

    def compute_rolling_momentum(
        self,
        returns: pd.DataFrame,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute rolling average return (momentum).

        Args:
            returns: DataFrame of returns
            window: Rolling window size (defaults to self.lookback_window)

        Returns:
            DataFrame of rolling average returns (annualized)
        """
        window = window or self.lookback_window

        # Rolling mean
        rolling_momentum = returns.rolling(window=window, min_periods=window).mean()

        # Annualize: multiply by 252
        rolling_momentum = rolling_momentum * 252

        return rolling_momentum

    def compute_rolling_sharpe(
        self,
        returns: pd.DataFrame,
        window: Optional[int] = None,
        risk_free_rate: float = 0.02
    ) -> pd.DataFrame:
        """
        Compute rolling Sharpe ratio.

        Args:
            returns: DataFrame of returns
            window: Rolling window size (defaults to self.lookback_window)
            risk_free_rate: Annual risk-free rate (default 2%)

        Returns:
            DataFrame of rolling Sharpe ratios
        """
        window = window or self.lookback_window

        # Compute rolling mean and std
        rolling_mean = returns.rolling(window=window, min_periods=window).mean()
        rolling_std = returns.rolling(window=window, min_periods=window).std()

        # Annualize
        annualized_return = rolling_mean * 252
        annualized_vol = rolling_std * np.sqrt(252)

        # Sharpe ratio
        sharpe = (annualized_return - risk_free_rate) / annualized_vol

        return sharpe

    def compute_rolling_correlation(
        self,
        returns: pd.DataFrame,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute rolling pairwise correlation matrix.

        Args:
            returns: DataFrame of returns
            window: Rolling window size (defaults to self.lookback_window)

        Returns:
            DataFrame with MultiIndex (date, ticker) showing correlations
        """
        window = window or self.lookback_window
        tickers = returns.columns

        # Store correlation matrices over time
        corr_matrices = []
        dates = []

        for i in range(window, len(returns)):
            window_data = returns.iloc[i-window:i]
            corr_matrix = window_data.corr()

            # Flatten to series with date
            date = returns.index[i]
            dates.append(date)
            corr_matrices.append(corr_matrix)

        # Convert to DataFrame with proper indexing
        # For simplicity, return average correlation for each asset
        avg_corr = []
        for corr_matrix in corr_matrices:
            # Average correlation with other assets (excluding self)
            avg_per_asset = {}
            for ticker in tickers:
                other_corrs = corr_matrix[ticker].drop(ticker)
                avg_per_asset[ticker] = other_corrs.mean()
            avg_corr.append(avg_per_asset)

        result = pd.DataFrame(avg_corr, index=dates)
        result.columns = [f"{col}_avg_corr" for col in result.columns]

        return result

    def compute_all_features(
        self,
        prices: pd.DataFrame,
        include_correlations: bool = True
    ) -> pd.DataFrame:
        """
        Compute all features for the given price data.

        Args:
            prices: DataFrame of prices
            include_correlations: Whether to include correlation features

        Returns:
            DataFrame with all features (will have NaNs for initial lookback period)
        """
        # Compute returns
        returns = self.compute_returns(prices)

        # Compute rolling features
        volatility = self.compute_rolling_volatility(returns)
        momentum = self.compute_rolling_momentum(returns)
        sharpe = self.compute_rolling_sharpe(returns)

        # Rename columns for clarity
        volatility.columns = [f"{col}_volatility" for col in volatility.columns]
        momentum.columns = [f"{col}_momentum" for col in momentum.columns]
        sharpe.columns = [f"{col}_sharpe" for col in sharpe.columns]

        # Combine features
        features = pd.concat([
            returns,
            volatility,
            momentum,
            sharpe
        ], axis=1)

        # Add correlation features if requested
        if include_correlations and len(prices.columns) > 1:
            correlations = self.compute_rolling_correlation(returns)
            features = pd.concat([features, correlations], axis=1)

        return features

    def get_features_at_date(
        self,
        features: pd.DataFrame,
        date: pd.Timestamp
    ) -> pd.Series:
        """
        Get features for a specific date.

        Args:
            features: DataFrame of all features
            date: Date to retrieve features for

        Returns:
            Series of features for that date (may contain NaN if in lookback period)
        """
        if date not in features.index:
            raise ValueError(f"Date {date} not in feature data")

        return features.loc[date]


def create_feature_summary(features: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics for features.

    Args:
        features: DataFrame of features

    Returns:
        DataFrame with summary stats
    """
    summary = pd.DataFrame({
        'Count': features.count(),
        'Mean': features.mean(),
        'Std': features.std(),
        'Min': features.min(),
        'Max': features.max(),
        'NaN Count': features.isna().sum()
    })

    return summary


if __name__ == "__main__":
    # Example usage
    from data import load_default_etfs, ETFDataLoader

    # Load data
    prices = load_default_etfs()

    # Create feature engineer
    feature_eng = FeatureEngineer(lookback_window=30)

    # Compute all features
    print("\nComputing features...")
    features = feature_eng.compute_all_features(prices)

    print("\n" + "="*80)
    print("Feature Summary")
    print("="*80)
    print(create_feature_summary(features))

    print("\n" + "="*80)
    print("Sample Features (last 5 days)")
    print("="*80)
    print(features.tail())

    print(f"\nTotal features: {len(features.columns)}")
    print(f"Feature columns: {list(features.columns)}")
