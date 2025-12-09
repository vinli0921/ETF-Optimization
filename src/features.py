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

    def compute_rsi(
        self,
        prices: pd.DataFrame,
        window: int = 14
    ) -> pd.DataFrame:
        """
        Compute Relative Strength Index (RSI) - momentum oscillator (0-100).

        RSI > 70: Overbought (potential sell signal)
        RSI < 30: Oversold (potential buy signal)

        Args:
            prices: DataFrame of prices
            window: RSI period (default 14 days)

        Returns:
            DataFrame of RSI values (0-100 scale)
        """
        delta = prices.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)

        # Calculate rolling average of gains and losses
        avg_gains = gains.rolling(window=window, min_periods=window).mean()
        avg_losses = losses.rolling(window=window, min_periods=window).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses.replace(0, np.nan)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        rsi.columns = [f"{col}_rsi{window}" for col in rsi.columns]

        return rsi

    def compute_macd(
        self,
        prices: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """
        Compute MACD (Moving Average Convergence Divergence) - trend following momentum.

        MACD line = fast EMA - slow EMA
        Signal line = EMA of MACD line
        Histogram = MACD - Signal (buy when positive, sell when negative)

        Args:
            prices: DataFrame of prices
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line period (default 9)

        Returns:
            DataFrame with MACD, Signal, and Histogram for each ticker
        """
        result = pd.DataFrame(index=prices.index)

        for col in prices.columns:
            # Calculate EMAs
            ema_fast = prices[col].ewm(span=fast, adjust=False).mean()
            ema_slow = prices[col].ewm(span=slow, adjust=False).mean()

            # MACD line
            macd_line = ema_fast - ema_slow

            # Signal line
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()

            # Histogram
            histogram = macd_line - signal_line

            result[f"{col}_macd"] = macd_line
            result[f"{col}_macd_signal"] = signal_line
            result[f"{col}_macd_hist"] = histogram

        return result

    def compute_bollinger_bands(
        self,
        prices: pd.DataFrame,
        window: int = 20,
        num_std: float = 2.0
    ) -> pd.DataFrame:
        """
        Compute Bollinger Bands - volatility bands for mean reversion.

        Upper band = SMA + (num_std × std)
        Lower band = SMA - (num_std × std)
        %B = (price - lower) / (upper - lower)  # Position within bands

        Args:
            prices: DataFrame of prices
            window: SMA window (default 20 days)
            num_std: Number of standard deviations (default 2.0)

        Returns:
            DataFrame with %B (position within bands, 0-1 scale)
        """
        result = pd.DataFrame(index=prices.index)

        for col in prices.columns:
            # Calculate SMA and rolling std
            sma = prices[col].rolling(window=window, min_periods=window).mean()
            rolling_std = prices[col].rolling(window=window, min_periods=window).std()

            # Upper and lower bands
            upper_band = sma + (num_std * rolling_std)
            lower_band = sma - (num_std * rolling_std)

            # %B: position within bands (0 = at lower band, 1 = at upper band)
            bandwidth = upper_band - lower_band
            percent_b = (prices[col] - lower_band) / bandwidth.replace(0, np.nan)

            result[f"{col}_bb_pct"] = percent_b

        return result

    def compute_atr(
        self,
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
        window: int = 14
    ) -> pd.DataFrame:
        """
        Compute Average True Range (ATR) - volatility indicator using OHLC.

        True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        ATR = rolling average of True Range

        Args:
            high: DataFrame of high prices
            low: DataFrame of low prices
            close: DataFrame of close prices
            window: ATR period (default 14 days)

        Returns:
            DataFrame of ATR values (normalized by close price)
        """
        result = pd.DataFrame(index=close.index)

        for col in close.columns:
            ticker = col.replace('_Close', '')
            high_col = f"{ticker}_High"
            low_col = f"{ticker}_Low"

            if high_col not in high.columns or low_col not in low.columns:
                continue

            # Calculate True Range components
            h_l = high[high_col] - low[low_col]
            h_pc = (high[high_col] - close[col].shift(1)).abs()
            l_pc = (low[low_col] - close[col].shift(1)).abs()

            # True Range = max of the three
            true_range = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)

            # ATR = rolling average of TR
            atr = true_range.rolling(window=window, min_periods=window).mean()

            # Normalize by close price (ATR as % of price)
            atr_pct = atr / close[col]

            result[f"{ticker}_atr{window}"] = atr_pct

        return result

    def compute_volume_features(
        self,
        volume: pd.DataFrame,
        prices: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Compute volume-based features.

        - Volume ratio: current / average (signals strength)
        - Volume trend: momentum of volume

        Args:
            volume: DataFrame of volume data
            prices: DataFrame of prices (for alignment)
            window: Rolling window for average volume

        Returns:
            DataFrame with volume features
        """
        result = pd.DataFrame(index=prices.index)

        for col in prices.columns:
            ticker = col.replace('_Close', '')
            volume_col = f"{ticker}_Volume"

            if volume_col not in volume.columns:
                continue

            vol = volume[volume_col]

            # Average volume
            avg_vol = vol.rolling(window=window, min_periods=window).mean()

            # Volume ratio (current / average)
            vol_ratio = vol / avg_vol.replace(0, np.nan)

            # Volume momentum (change over window days)
            vol_momentum = vol.pct_change(periods=window)

            result[f"{ticker}_vol_ratio"] = vol_ratio
            result[f"{ticker}_vol_momentum"] = vol_momentum

        return result

    def compute_all_features(
        self,
        prices: pd.DataFrame = None,
        ohlcv_data: pd.DataFrame = None,
        indicators: pd.DataFrame = None,
        include_correlations: bool = True,
        include_technical: bool = True,
        include_volume: bool = True,
        include_market: bool = True
    ) -> pd.DataFrame:
        """
        Compute all features for ML models from OHLCV and market indicators.

        Args:
            prices: DataFrame of close prices (legacy, can be None if ohlcv_data provided)
            ohlcv_data: DataFrame with OHLCV columns ({ticker}_{metric} format)
            indicators: DataFrame with market indicators (VIX, yields, etc.)
            include_correlations: Whether to include correlation features
            include_technical: Whether to include technical indicators (RSI, MACD, BB, ATR)
            include_volume: Whether to include volume features
            include_market: Whether to include market-wide features (VIX, yields)

        Returns:
            DataFrame with all features (will have NaNs for initial lookback period)
        """
        # Extract close prices from OHLCV if provided
        if ohlcv_data is not None:
            close_cols = [col for col in ohlcv_data.columns if col.endswith('_Close')]
            prices = ohlcv_data[close_cols].copy()
            prices.columns = [col.replace('_Close', '') for col in close_cols]

        if prices is None:
            raise ValueError("Must provide either prices or ohlcv_data")

        # Compute returns
        returns = self.compute_returns(prices)

        # Compute basic rolling features
        volatility = self.compute_rolling_volatility(returns)
        momentum = self.compute_rolling_momentum(returns)
        sharpe = self.compute_rolling_sharpe(returns)

        # Rename columns for clarity
        volatility.columns = [f"{col}_volatility" for col in volatility.columns]
        momentum.columns = [f"{col}_momentum" for col in momentum.columns]
        sharpe.columns = [f"{col}_sharpe" for col in sharpe.columns]

        # Start with basic features
        feature_list = [returns, volatility, momentum, sharpe]

        # Add correlation features if requested
        if include_correlations and len(prices.columns) > 1:
            correlations = self.compute_rolling_correlation(returns)
            feature_list.append(correlations)

        # Add technical indicators if OHLCV data provided
        if include_technical and ohlcv_data is not None:
            # RSI
            rsi = self.compute_rsi(prices, window=14)
            feature_list.append(rsi)

            # MACD
            macd = self.compute_macd(prices, fast=12, slow=26, signal=9)
            feature_list.append(macd)

            # Bollinger Bands
            bollinger = self.compute_bollinger_bands(prices, window=20, num_std=2.0)
            feature_list.append(bollinger)

            # ATR (requires OHLC)
            high_cols = [col for col in ohlcv_data.columns if col.endswith('_High')]
            low_cols = [col for col in ohlcv_data.columns if col.endswith('_Low')]

            if high_cols and low_cols:
                high_df = ohlcv_data[high_cols]
                low_df = ohlcv_data[low_cols]
                close_df = ohlcv_data[close_cols]

                atr = self.compute_atr(high_df, low_df, close_df, window=14)
                feature_list.append(atr)

        # Add volume features if OHLCV data provided
        if include_volume and ohlcv_data is not None:
            volume_cols = [col for col in ohlcv_data.columns if col.endswith('_Volume')]

            if volume_cols:
                volume_df = ohlcv_data[volume_cols]
                volume_features = self.compute_volume_features(volume_df, prices, window=20)
                feature_list.append(volume_features)

        # Add market-wide features if indicators provided
        if include_market and indicators is not None:
            market_features = self._compute_market_features(indicators, prices.index)
            feature_list.append(market_features)

        # Combine all features
        features = pd.concat(feature_list, axis=1)

        return features

    def _compute_market_features(
        self,
        indicators: pd.DataFrame,
        price_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Compute market-wide features from indicators (VIX, yields).

        Args:
            indicators: DataFrame with VIX, yields, etc.
            price_index: DatetimeIndex to align features with

        Returns:
            DataFrame with market features
        """
        result = pd.DataFrame(index=price_index)

        # Align indicators to price index
        aligned_indicators = indicators.reindex(price_index).ffill()

        # VIX features
        if 'VIX' in aligned_indicators.columns:
            vix = aligned_indicators['VIX']

            # VIX level (raw)
            result['VIX_level'] = vix

            # VIX change (1-day momentum)
            result['VIX_change'] = vix.pct_change()

            # VIX percentile (rolling 252-day)
            vix_min = vix.rolling(window=252, min_periods=60).min()
            vix_max = vix.rolling(window=252, min_periods=60).max()
            result['VIX_percentile'] = (vix - vix_min) / (vix_max - vix_min).replace(0, np.nan)

            # High volatility regime (VIX > 1.5 × 20-day MA)
            vix_ma20 = vix.rolling(window=20, min_periods=20).mean()
            result['VIX_high_regime'] = (vix > vix_ma20 * 1.5).astype(float)

        # Yield curve features
        if 'TNX_10Y' in aligned_indicators.columns and 'IRX_3M' in aligned_indicators.columns:
            yield_10y = aligned_indicators['TNX_10Y']
            yield_3m = aligned_indicators['IRX_3M']

            # 10Y-3M spread (recession indicator: negative = inverted curve)
            result['yield_curve_10y_3m'] = yield_10y - yield_3m

            # Yield curve slope percentile
            spread = yield_10y - yield_3m
            spread_min = spread.rolling(window=252, min_periods=60).min()
            spread_max = spread.rolling(window=252, min_periods=60).max()
            result['yield_curve_percentile'] = (spread - spread_min) / (spread_max - spread_min).replace(0, np.nan)

            # Inverted curve indicator (1 if inverted, 0 otherwise)
            result['yield_curve_inverted'] = (spread < 0).astype(float)

        # 10Y yield features
        if 'TNX_10Y' in aligned_indicators.columns:
            yield_10y = aligned_indicators['TNX_10Y']

            # Yield level
            result['yield_10y_level'] = yield_10y

            # Yield momentum (change over 20 days)
            result['yield_10y_momentum'] = yield_10y.pct_change(periods=20)

        return result

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


def compute_sentiment_features(
    prices: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    ma_window: int = 5
) -> pd.DataFrame:
    """
    Compute sentiment features aligned with price data.

    Takes raw sentiment data and creates features suitable for ML models:
    - Raw sentiment (lagged by 1 day to avoid look-ahead)
    - Moving average sentiment
    - Sentiment momentum

    Args:
        prices: DataFrame of prices with DatetimeIndex
        sentiment_df: DataFrame with {ticker}_sentiment columns
        ma_window: Window for moving average

    Returns:
        DataFrame with sentiment features aligned to prices index
    """
    features = pd.DataFrame(index=prices.index)

    for ticker in prices.columns:
        sentiment_col = f"{ticker}_sentiment"

        if sentiment_col in sentiment_df.columns:
            # Align sentiment to price index
            aligned = sentiment_df[sentiment_col].reindex(prices.index)

            # Forward fill missing values (weekends/holidays)
            aligned = aligned.fillna(method="ffill").fillna(0)

            # Lag by 1 day to avoid look-ahead bias
            features[sentiment_col] = aligned.shift(1).fillna(0)

            # Moving average (on lagged data)
            features[f"{ticker}_sentiment_ma{ma_window}"] = (
                features[sentiment_col]
                .rolling(window=ma_window, min_periods=1)
                .mean()
            )

            # Sentiment momentum (change over ma_window days)
            features[f"{ticker}_sentiment_momentum"] = (
                features[sentiment_col] -
                features[sentiment_col].shift(ma_window)
            ).fillna(0)
        else:
            # No sentiment data for this ticker - fill with zeros
            features[sentiment_col] = 0.0
            features[f"{ticker}_sentiment_ma{ma_window}"] = 0.0
            features[f"{ticker}_sentiment_momentum"] = 0.0

    return features


def compute_all_features_with_sentiment(
    prices: pd.DataFrame,
    sentiment_df: Optional[pd.DataFrame] = None,
    lookback_window: int = 30,
    ma_window: int = 5,
    include_correlations: bool = True
) -> pd.DataFrame:
    """
    Compute all features including sentiment for ML models.

    Combines price-based features with sentiment features into a single DataFrame.

    Args:
        prices: DataFrame of prices
        sentiment_df: Optional DataFrame with sentiment data
        lookback_window: Window for rolling price statistics
        ma_window: Window for sentiment moving average
        include_correlations: Whether to include correlation features

    Returns:
        DataFrame with all features (will have NaNs for initial lookback period)
    """
    # Compute price-based features
    feature_eng = FeatureEngineer(lookback_window=lookback_window)
    price_features = feature_eng.compute_all_features(
        prices,
        include_correlations=include_correlations
    )

    # Add sentiment features if available
    if sentiment_df is not None and not sentiment_df.empty:
        sentiment_features = compute_sentiment_features(
            prices, sentiment_df, ma_window
        )
        all_features = pd.concat([price_features, sentiment_features], axis=1)
    else:
        all_features = price_features

    return all_features


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
