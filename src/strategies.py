"""
Portfolio allocation strategies.

Implementst baseline strategies including equal weight and mean-variance optimization.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import warnings
from typing import Optional, Dict
from pypfopt import expected_returns, risk_models, objective_functions
from pypfopt.efficient_frontier import EfficientFrontier
from sklearn.linear_model import Ridge
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from features import FeatureEngineer
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError):
    # OSError covers missing libomp on macOS; ImportError covers missing package
    LIGHTGBM_AVAILABLE = False
    print("Warning: lightgbm unavailable (missing package or libomp). "
          "GradientBoostingSharpeStrategy will fall back to RandomForest.")
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost unavailable. Install with: conda install -c conda-forge xgboost")
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from lstm_model import LSTMReturnPredictor 


class BaseStrategy(ABC):
    """
    Abstract base class for portfolio allocation strategies.
    """

    def __init__(self, name: str):
        """
        Initialize strategy.

        Args:
            name: Name of the strategy
        """
        self.name = name

    @abstractmethod
    def allocate(
        self,
        prices: pd.DataFrame,
        current_date: Optional[pd.Timestamp] = None,
        ohlcv_data: Optional[pd.DataFrame] = None,
        indicators: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute portfolio weights for given price data.

        Args:
            prices: Historical price data (up to but not including current_date)
            current_date: Date for which to compute allocation
            ohlcv_data: Optional OHLCV data (Open, High, Low, Close, Volume)
            indicators: Optional market indicators (VIX, yields, etc.)
            **kwargs: Strategy-specific parameters

        Returns:
            Dictionary mapping ticker to portfolio weight (should sum to 1.0)
        """
        pass

    def validate_weights(self, weights: Dict[str, float], tol: float = 1e-4) -> bool:
        """
        Validate that weights sum to approximately 1.0 and are non-negative.

        Args:
            weights: Dictionary of ticker -> weight
            tol: Tolerance for sum check

        Returns:
            True if valid, raises ValueError otherwise
        """
        total = sum(weights.values())
        if abs(total - 1.0) > tol:
            raise ValueError(f"Weights sum to {total}, expected 1.0")

        for ticker, weight in weights.items():
            if weight < -tol:  # Allow tiny negative due to numerical errors
                raise ValueError(f"Negative weight for {ticker}: {weight}")
            if weight < 0:
                weights[ticker] = 0.0  # Fix tiny negative

        # Renormalize to exactly 1.0
        total = sum(weights.values())
        for ticker in weights:
            weights[ticker] /= total

        return True


class EqualWeightStrategy(BaseStrategy):
    """
    Equal weight (1/N) allocation strategy.

    Allocates equal weights to all assets regardless of market conditions.
    """

    def __init__(self):
        super().__init__("Equal Weight")

    def allocate(
        self,
        prices: pd.DataFrame,
        current_date: Optional[pd.Timestamp] = None,
        ohlcv_data: Optional[pd.DataFrame] = None,
        indicators: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Allocate equal weights to all tickers.

        Args:
            prices: Historical price data
            current_date: Unused for this strategy
            ohlcv_data: Unused for this strategy
            indicators: Unused for this strategy
            **kwargs: Unused

        Returns:
            Dictionary with equal weights
        """
        tickers = prices.columns
        n_assets = len(tickers)
        weight = 1.0 / n_assets

        weights = {ticker: weight for ticker in tickers}

        self.validate_weights(weights)
        return weights


class MeanVarianceStrategy(BaseStrategy):
    """
    Mean-variance optimization strategy.

    Uses historical returns and covariances to find the maximum Sharpe ratio
    (or other mean-variance objectives) via PyPortfolioOpt.
    """

    def __init__(
        self,
        lookback_days: int = 252,
        risk_free_rate: float = 0.0,
        method: str = "max_sharpe",
        use_shrinkage: bool = True,
        l2_gamma: float = 0.01,
        max_weight: float = 0.7,
        min_history_days: int = 30,
    ):
        """
        Initialize mean-variance strategy.

        Args:
            lookback_days: Number of calendar days of history to use for estimation.
            risk_free_rate: Annual risk-free rate for Sharpe calculation.
            method: 'max_sharpe', 'min_volatility', or 'efficient_risk'.
            use_shrinkage: Whether to use Ledoit-Wolf shrinkage covariance.
            l2_gamma: Strength of L2 weight regularization (0 disables it).
            max_weight: Upper bound per-asset (e.g., 0.7 = max 70% in one asset).
                       NOTE: Must be at least 1/n_assets to be feasible!
            min_history_days: Minimum number of rows required before optimizing.
        """
        super().__init__("Mean-Variance Optimization")
        self.lookback_days = lookback_days
        self.risk_free_rate = risk_free_rate
        self.method = method
        self.use_shrinkage = use_shrinkage
        self.l2_gamma = l2_gamma
        self.max_weight = max_weight
        self.min_history_days = min_history_days

    def allocate(
        self,
        prices: pd.DataFrame,
        current_date: Optional[pd.Timestamp] = None,
        ohlcv_data: Optional[pd.DataFrame] = None,
        indicators: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, float]:

        # Use only data up to current_date (inclusive)
        if current_date is not None:
            prices = prices.loc[:current_date]

        # Use last lookback_days of history (by row count)
        if len(prices) > self.lookback_days:
            prices = prices.iloc[-self.lookback_days:]

        # Require minimum history
        if len(prices) < self.min_history_days:
            print(f"Warning: Insufficient data ({len(prices)} days), using equal weights")
            return EqualWeightStrategy().allocate(prices)

        try:
            # Expected returns (annualized)
            mu = expected_returns.mean_historical_return(prices, frequency=252)

            # Covariance matrix (annualized)
            if self.use_shrinkage:
                S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
            else:
                S = risk_models.sample_cov(prices, frequency=252)

            # Ensure max_weight is feasible (at least 1/n_assets)
            n_assets = len(prices.columns)
            effective_max_weight = max(self.max_weight, 1.0 / n_assets + 0.01)

            # Efficient frontier, long-only, limited concentration
            ef = EfficientFrontier(mu, S, weight_bounds=(0, effective_max_weight))

            # Optional L2 regularization to smooth weights
            if self.l2_gamma is not None and self.l2_gamma > 0:
                ef.add_objective(objective_functions.L2_reg, gamma=self.l2_gamma)

            # Choose optimization objective
            if self.method == "max_sharpe":
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', message='.*max_sharpe.*')
                        ef.max_sharpe(risk_free_rate=self.risk_free_rate)
                except Exception:
                    # Fallback to min_volatility if max_sharpe fails
                    # (e.g., when all returns < risk-free rate)
                    ef = EfficientFrontier(mu, S, weight_bounds=(0, effective_max_weight))
                    if self.l2_gamma is not None and self.l2_gamma > 0:
                        ef.add_objective(objective_functions.L2_reg, gamma=self.l2_gamma)
                    ef.min_volatility()
            elif self.method == "min_volatility":
                ef.min_volatility()
            elif self.method == "efficient_risk":
                target_vol = kwargs.get("target_volatility", 0.15)
                ef.efficient_risk(target_vol)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Clean and export weights
            weights = ef.clean_weights()

            # Ensure all tickers present
            for ticker in prices.columns:
                weights.setdefault(ticker, 0.0)

            self.validate_weights(weights)
            return weights

        except Exception as e:
            print(f"Warning: Optimization failed ({str(e)}), using equal weights")
            return EqualWeightStrategy().allocate(prices)


class PredictiveSharpeStrategy(BaseStrategy):
    """
    Regression-based Sharpe ratio optimization strategy.

    Uses Ridge regression to predict expected returns from features (momentum,
    volatility, rolling Sharpe), then optimizes for maximum Sharpe ratio using
    those predictions with Ledoit-Wolf shrinkage covariance.

    This approach aims to improve out-of-sample performance by:
    1. Regularizing return predictions (Ridge regression)
    2. Shrinking covariance estimates (Ledoit-Wolf)
    3. Shrinking predicted returns toward the grand mean (James-Stein style)
    """

    def __init__(
        self,
        lookback_days: int = 252,
        feature_window: int = 30,
        risk_free_rate: float = 0.0,
        ridge_alpha: float = 1.0,
        shrinkage_intensity: float = 0.5,
        max_weight: float = 0.7,
        l2_gamma: float = 0.01,
        min_history_days: int = 60,
    ):
        """
        Initialize predictive Sharpe strategy.

        Args:
            lookback_days: Number of trading days of history to use for training.
            feature_window: Rolling window for feature computation (momentum, vol).
            risk_free_rate: Annual risk-free rate for Sharpe calculation.
            ridge_alpha: Regularization strength for Ridge regression (higher = more regularization).
            shrinkage_intensity: How much to shrink predicted returns toward grand mean (0-1).
                0 = use raw predictions, 1 = use global mean for all assets.
            max_weight: Maximum weight per asset (concentration limit).
                       NOTE: Must be at least 1/n_assets to be feasible!
            l2_gamma: L2 regularization on portfolio weights.
            min_history_days: Minimum rows required before using predictions.
        """
        super().__init__("Predictive Sharpe (Ridge Regression)")
        self.lookback_days = lookback_days
        self.feature_window = feature_window
        self.risk_free_rate = risk_free_rate
        self.ridge_alpha = ridge_alpha
        self.shrinkage_intensity = shrinkage_intensity
        self.max_weight = max_weight
        self.l2_gamma = l2_gamma
        self.min_history_days = min_history_days

        # Initialize FeatureEngineer for comprehensive feature computation
        self.feature_engineer = FeatureEngineer(
            lookback_window=feature_window
        )

        # Store trained models (one per asset)
        self._models: Dict[str, Ridge] = {}
        self.model = None  # Store the trained model for feature importance analysis

    def _compute_features(
        self,
        prices: pd.DataFrame,
        ohlcv_data: Optional[pd.DataFrame] = None,
        indicators: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compute features using FeatureEngineer.

        Returns 160+ features including:
        - Basic: returns, volatility, momentum, Sharpe (6 per ticker)
        - Technical: RSI, MACD, Bollinger Bands, ATR (16 per ticker)
        - Volume: volume features (4 per ticker)
        - Market: VIX, yields, spreads (global)
        - Correlations: rolling correlations between assets

        Args:
            prices: Historical price data
            ohlcv_data: Optional OHLCV data for technical indicators
            indicators: Optional market indicators

        Returns:
            DataFrame with comprehensive features
        """
        # Use FeatureEngineer to compute all 160+ features
        features = self.feature_engineer.compute_all_features(
            prices=prices,
            ohlcv_data=ohlcv_data,
            indicators=indicators,
            include_correlations=True,
            include_technical=True,
            include_volume=True,
            include_market=True
        )

        # Shift all features by 1 day to prevent look-ahead bias
        # (FeatureEngineer already does this internally, but we double-check)
        features = features.shift(1)

        return features

    def _tune_ridge_alpha(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> float:
        """
        Tune ridge_alpha via TimeSeriesSplit cross-validation.

        Args:
            X_train: Training features
            y_train: Training targets

        Returns:
            Best alpha value
        """
        from sklearn.metrics import mean_squared_error

        # EXPANDED GRID: Add lower values to prevent over-regularization
        alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
        tscv = TimeSeriesSplit(n_splits=5)
        best_alpha = self.ridge_alpha
        best_score = float('inf')
        cv_scores = {}  # Track all scores for debugging

        for alpha in alphas:
            scores = []
            for train_idx, val_idx in tscv.split(X_train):
                model = Ridge(alpha=alpha, fit_intercept=True)
                model.fit(X_train[train_idx], y_train[train_idx])
                y_pred = model.predict(X_train[val_idx])
                mse = mean_squared_error(y_train[val_idx], y_pred)
                scores.append(mse)

            avg_score = np.mean(scores)
            cv_scores[alpha] = avg_score
            if avg_score < best_score:
                best_score = avg_score
                best_alpha = alpha

        # Print top 3 alphas for debugging
        sorted_alphas = sorted(cv_scores.items(), key=lambda x: x[1])
        print(f"    Top 3 alphas (alpha, MSE): {sorted_alphas[:3]}")
        print(f"    Selected alpha: {best_alpha} (MSE: {best_score:.6f})")

        return best_alpha

    def _train_models(
        self,
        prices: pd.DataFrame,
        ohlcv_data: Optional[pd.DataFrame] = None,
        indicators: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Train Ridge regression models to predict returns with feature scaling.

        Args:
            prices: Historical price data
            ohlcv_data: Optional OHLCV data for technical indicators
            indicators: Optional market indicators

        Returns:
            Dictionary of predicted annualized returns per asset
        """
        from sklearn.preprocessing import StandardScaler

        returns = prices.pct_change()
        features = self._compute_features(prices, ohlcv_data, indicators)

        print(f"  Feature count: {len(features.columns)} features (vs old 60)")

        # NEW: Standardize features before training (Ridge is sensitive to scale)
        scaler = StandardScaler()

        predictions = {}

        for ticker in prices.columns:
            # Use ALL features for predicting each ticker
            X = features.copy()

            # Target: next-day return
            y = returns[ticker]

            # Align X and y, drop NaNs
            combined = pd.concat([X, y.rename('target')], axis=1).dropna()

            if len(combined) < self.min_history_days:
                # Not enough data, use historical mean
                mean_return = returns[ticker].mean() * 252
                predictions[ticker] = mean_return if not np.isnan(mean_return) else 0.0
                continue

            X_train = combined.drop('target', axis=1).values
            y_train = combined['target'].values

            # Standardize features
            X_train_scaled = scaler.fit_transform(X_train)

            # Tune alpha on scaled data (only for first ticker)
            if ticker == prices.columns[0]:
                tuned_alpha = self._tune_ridge_alpha(X_train_scaled, y_train)
            else:
                tuned_alpha = getattr(self, '_tuned_alpha', self.ridge_alpha)

            self._tuned_alpha = tuned_alpha

            # Train Ridge regression on scaled data
            model = Ridge(alpha=tuned_alpha, fit_intercept=True)
            model.fit(X_train_scaled, y_train)
            self._models[ticker] = model

            # Store model AND scaler
            self._scaler = scaler
            self.model = model

            # Predict on scaled features
            latest_features = features.iloc[-1:].values
            if np.any(np.isnan(latest_features)):
                mean_return = returns[ticker].mean() * 252
                predictions[ticker] = mean_return if not np.isnan(mean_return) else 0.0
            else:
                # Scale latest features and predict
                latest_features_scaled = scaler.transform(latest_features)
                pred_daily = model.predict(latest_features_scaled)[0]
                predictions[ticker] = pred_daily * 252

        return predictions

    def _apply_shrinkage(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """
        Apply James-Stein style shrinkage to predicted returns.

        Shrinks predictions toward the grand mean to reduce estimation error.

        Args:
            predictions: Raw predicted returns per asset

        Returns:
            Shrunk predictions
        """
        if not predictions:
            return predictions

        values = np.array(list(predictions.values()))
        grand_mean = np.mean(values)

        # Shrink toward grand mean
        shrunk = {}
        for ticker, pred in predictions.items():
            shrunk[ticker] = (
                (1 - self.shrinkage_intensity) * pred +
                self.shrinkage_intensity * grand_mean
            )

        return shrunk

    def allocate(
        self,
        prices: pd.DataFrame,
        current_date: Optional[pd.Timestamp] = None,
        ohlcv_data: Optional[pd.DataFrame] = None,
        indicators: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute portfolio weights using predicted returns.

        Args:
            prices: Historical price data (up to but not including current_date)
            current_date: Date for which to compute allocation
            ohlcv_data: Optional OHLCV data for technical indicators
            indicators: Optional market indicators
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping ticker to portfolio weight
        """
        # Use only data up to current_date
        if current_date is not None:
            prices = prices.loc[:current_date]

        # Use last lookback_days of history
        if len(prices) > self.lookback_days:
            prices = prices.iloc[-self.lookback_days:]

        # Require minimum history
        if len(prices) < self.min_history_days:
            print(f"Warning: Insufficient data ({len(prices)} days), using equal weights")
            return EqualWeightStrategy().allocate(prices)

        try:
            # Step 1: Predict expected returns using Ridge regression with enhanced features
            predicted_returns = self._train_models(prices, ohlcv_data, indicators)

            # Step 2: Apply shrinkage to predictions
            shrunk_returns = self._apply_shrinkage(predicted_returns)

            # Convert to pandas Series for PyPortfolioOpt
            mu = pd.Series(shrunk_returns)

            # Step 3: Estimate covariance using Ledoit-Wolf shrinkage
            returns_df = prices.pct_change().dropna()
            lw = LedoitWolf().fit(returns_df.values)

            # Convert to annualized covariance matrix
            cov_matrix = pd.DataFrame(
                lw.covariance_ * 252,
                index=prices.columns,
                columns=prices.columns
            )

            # Step 4: Optimize using PyPortfolioOpt
            ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(0, self.max_weight))

            # Add L2 regularization
            if self.l2_gamma > 0:
                ef.add_objective(objective_functions.L2_reg, gamma=self.l2_gamma)

            # Maximize Sharpe ratio
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*max_sharpe.*')
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)

            # Get cleaned weights
            weights = ef.clean_weights()

            # Ensure all tickers present
            for ticker in prices.columns:
                weights.setdefault(ticker, 0.0)

            self.validate_weights(weights)
            return weights

        except Exception as e:
            print(f"Warning: Predictive optimization failed ({str(e)}), using equal weights")
            return EqualWeightStrategy().allocate(prices)


class GradientBoostingSharpeStrategy(BaseStrategy):
    """
    Gradient boosting-based Sharpe ratio optimization strategy.

    Uses LightGBM (or RandomForest fallback) to predict expected returns from features,
    then optimizes for maximum Sharpe ratio using those predictions.

    This approach aims to capture non-linear relationships in the data that
    linear methods like Ridge regression may miss.
    """

    def __init__(
        self,
        lookback_days: int = 252,
        feature_window: int = 30,
        risk_free_rate: float = 0.0,
        shrinkage_intensity: float = 0.5,
        max_weight: float = 0.4,
        l2_gamma: float = 0.01,
        min_history_days: int = 60,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        use_sentiment: bool = False,
    ):
        """
        Initialize gradient boosting Sharpe strategy.

        Args:
            lookback_days: Number of trading days of history to use for training.
            feature_window: Rolling window for feature computation (momentum, vol).
            risk_free_rate: Annual risk-free rate for Sharpe calculation.
            shrinkage_intensity: How much to shrink predicted returns toward grand mean (0-1).
            max_weight: Maximum weight per asset (concentration limit).
            l2_gamma: L2 regularization on portfolio weights.
            min_history_days: Minimum rows required before using predictions.
            n_estimators: Number of boosting iterations.
            max_depth: Maximum tree depth.
            learning_rate: Boosting learning rate.
            use_sentiment: Whether to include sentiment features (if available).
        """
        super().__init__("Gradient Boosting Sharpe (LightGBM)")
        self.lookback_days = lookback_days
        self.feature_window = feature_window
        self.risk_free_rate = risk_free_rate
        self.shrinkage_intensity = shrinkage_intensity
        self.max_weight = max_weight
        self.l2_gamma = l2_gamma
        self.min_history_days = min_history_days
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.use_sentiment = use_sentiment

        # Store trained models (one per asset)
        self._models: Dict[str, object] = {}

    def _compute_features(
        self,
        prices: pd.DataFrame,
        sentiment_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compute features for prediction.

        Features per asset:
        - Lagged returns (1, 5, 21 days)
        - Rolling momentum (annualized)
        - Rolling volatility (annualized)
        - Rolling Sharpe ratio
        - Sentiment features (if available and enabled)

        Args:
            prices: Historical price data
            sentiment_df: Optional sentiment features DataFrame

        Returns:
            DataFrame with features
        """
        returns = prices.pct_change()

        features_list = []

        for ticker in prices.columns:
            ticker_returns = returns[ticker]

            # Lagged returns
            lag_1 = ticker_returns.shift(1)
            lag_5 = ticker_returns.rolling(5).mean().shift(1)
            lag_21 = ticker_returns.rolling(21).mean().shift(1)

            # Rolling momentum (annualized)
            momentum = ticker_returns.rolling(self.feature_window).mean().shift(1) * 252

            # Rolling volatility (annualized)
            volatility = ticker_returns.rolling(self.feature_window).std().shift(1) * np.sqrt(252)

            # Rolling Sharpe
            rolling_sharpe = momentum / (volatility + 1e-8)

            # Combine into feature DataFrame
            ticker_features = pd.DataFrame({
                f'{ticker}_lag1': lag_1,
                f'{ticker}_lag5': lag_5,
                f'{ticker}_lag21': lag_21,
                f'{ticker}_momentum': momentum,
                f'{ticker}_volatility': volatility,
                f'{ticker}_sharpe': rolling_sharpe,
            })

            features_list.append(ticker_features)

        features = pd.concat(features_list, axis=1)

        # Add sentiment features if available
        if self.use_sentiment and sentiment_df is not None:
            for ticker in prices.columns:
                sentiment_col = f"{ticker}_sentiment"
                if sentiment_col in sentiment_df.columns:
                    # Align sentiment with price index
                    aligned_sentiment = sentiment_df[sentiment_col].reindex(prices.index).fillna(0)
                    features[sentiment_col] = aligned_sentiment.shift(1)  # Lag to avoid look-ahead

                    # Add sentiment momentum
                    features[f"{ticker}_sentiment_momentum"] = (
                        aligned_sentiment - aligned_sentiment.shift(5)
                    ).shift(1).fillna(0)

        return features

    def _create_model(self):
        """Create a gradient boosting model (LightGBM or RandomForest fallback)."""
        if LIGHTGBM_AVAILABLE:
            return lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
            )
        else:
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1,
            )

    def _train_models(
        self,
        prices: pd.DataFrame,
        sentiment_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Train gradient boosting models to predict returns.

        Args:
            prices: Historical price data
            sentiment_df: Optional sentiment features

        Returns:
            Dictionary of predicted annualized returns per asset
        """
        returns = prices.pct_change()
        features = self._compute_features(prices, sentiment_df)

        predictions = {}

        for ticker in prices.columns:
            # Get all features (we use all features for each ticker)
            X = features.copy()

            # Target: next-day return
            y = returns[ticker]

            # Align X and y, drop NaNs
            combined = pd.concat([X, y.rename('target')], axis=1).dropna()

            if len(combined) < self.min_history_days:
                # Not enough data, use historical mean
                mean_return = returns[ticker].mean() * 252
                predictions[ticker] = mean_return if not np.isnan(mean_return) else 0.0
                continue

            X_train = combined.drop('target', axis=1)  # Keep as DataFrame for feature names
            y_train = combined['target'].values

            # Train gradient boosting model
            model = self._create_model()
            model.fit(X_train, y_train)
            self._models[ticker] = model

            # Predict using most recent features (keep as DataFrame)
            latest_features = features.iloc[-1:]

            if latest_features.isna().any().any():
                # Features have NaN, use historical mean
                mean_return = returns[ticker].mean() * 252
                predictions[ticker] = mean_return if not np.isnan(mean_return) else 0.0
            else:
                # Predict and annualize
                pred_daily = model.predict(latest_features)[0]
                predictions[ticker] = pred_daily * 252

        return predictions

    def _apply_shrinkage(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """
        Apply James-Stein style shrinkage to predicted returns.

        Shrinks predictions toward the grand mean to reduce estimation error.

        Args:
            predictions: Raw predicted returns per asset

        Returns:
            Shrunk predictions
        """
        if not predictions:
            return predictions

        values = np.array(list(predictions.values()))
        grand_mean = np.mean(values)

        # Shrink toward grand mean
        shrunk = {}
        for ticker, pred in predictions.items():
            shrunk[ticker] = (
                (1 - self.shrinkage_intensity) * pred +
                self.shrinkage_intensity * grand_mean
            )

        return shrunk

    def allocate(
        self,
        prices: pd.DataFrame,
        current_date: Optional[pd.Timestamp] = None,
        ohlcv_data: Optional[pd.DataFrame] = None,
        indicators: Optional[pd.DataFrame] = None,
        sentiment_df: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute portfolio weights using gradient boosting predicted returns.

        Args:
            prices: Historical price data (up to but not including current_date)
            current_date: Date for which to compute allocation
            ohlcv_data: Optional OHLCV data for technical indicators
            indicators: Optional market indicators
            sentiment_df: Optional sentiment features DataFrame
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping ticker to portfolio weight
        """
        # Use only data up to current_date
        if current_date is not None:
            prices = prices.loc[:current_date]

        # Use last lookback_days of history
        if len(prices) > self.lookback_days:
            prices = prices.iloc[-self.lookback_days:]

        # Require minimum history
        if len(prices) < self.min_history_days:
            print(f"Warning: Insufficient data ({len(prices)} days), using equal weights")
            return EqualWeightStrategy().allocate(prices)

        try:
            # Step 1: Predict expected returns using gradient boosting
            predicted_returns = self._train_models(prices, sentiment_df)

            # Step 2: Apply shrinkage to predictions
            shrunk_returns = self._apply_shrinkage(predicted_returns)

            # Convert to pandas Series for PyPortfolioOpt
            mu = pd.Series(shrunk_returns)

            # Step 3: Estimate covariance using Ledoit-Wolf shrinkage
            returns_df = prices.pct_change().dropna()
            lw = LedoitWolf().fit(returns_df.values)

            # Convert to annualized covariance matrix
            cov_matrix = pd.DataFrame(
                lw.covariance_ * 252,
                index=prices.columns,
                columns=prices.columns
            )

            # Ensure max_weight is feasible (at least 1/n_assets)
            n_assets = len(prices.columns)
            effective_max_weight = max(self.max_weight, 1.0 / n_assets + 0.01)

            # Step 4: Optimize using PyPortfolioOpt
            ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(0, effective_max_weight))

            # Add L2 regularization
            if self.l2_gamma > 0:
                ef.add_objective(objective_functions.L2_reg, gamma=self.l2_gamma)

            # Maximize Sharpe ratio
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*max_sharpe.*')
                    ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            except Exception:
                # Fallback to min volatility if max_sharpe fails
                ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(0, effective_max_weight))
                if self.l2_gamma > 0:
                    ef.add_objective(objective_functions.L2_reg, gamma=self.l2_gamma)
                ef.min_volatility()

            # Get cleaned weights
            weights = ef.clean_weights()

            # Ensure all tickers present
            for ticker in prices.columns:
                weights.setdefault(ticker, 0.0)

            self.validate_weights(weights)
            return weights

        except Exception as e:
            print(f"Warning: Gradient boosting optimization failed ({str(e)}), using equal weights")
            return EqualWeightStrategy().allocate(prices)


class XGBoostSharpeStrategy(BaseStrategy):
    """
    XGBoost-based Sharpe ratio optimization strategy.

    More conservative than LightGBM (level-wise vs leaf-wise tree growth),
    better for small datasets with many features. Uses comprehensive
    feature set from FeatureEngineer (160+ features).
    """

    def __init__(
        self,
        lookback_days: int = 756,
        feature_window: int = 60,
        risk_free_rate: float = 0.02,
        shrinkage_intensity: float = 0.25,
        max_weight: float = 0.4,
        l2_gamma: float = 0.01,
        min_history_days: int = 60,
        # XGBoost hyperparameters (conservative)
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.01,
        subsample: float = 0.8,
        colsample_bytree: float = 0.6,
        reg_alpha: float = 1.0,
        reg_lambda: float = 10.0,
    ):
        """
        Initialize XGBoost Sharpe strategy.

        Args:
            lookback_days: Number of trading days of history to use for training.
            feature_window: Rolling window for feature computation.
            risk_free_rate: Annual risk-free rate for Sharpe calculation.
            shrinkage_intensity: How much to shrink predicted returns toward grand mean (0-1).
            max_weight: Maximum weight per asset (concentration limit).
            l2_gamma: L2 regularization on portfolio weights.
            min_history_days: Minimum rows required before using predictions.
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth (shallow for regularization).
            learning_rate: Boosting learning rate.
            subsample: Row sampling ratio.
            colsample_bytree: Column sampling ratio (important for 160+ features).
            reg_alpha: L1 regularization on weights.
            reg_lambda: L2 regularization on weights.
        """
        super().__init__("XGBoost Sharpe")
        self.lookback_days = lookback_days
        self.feature_window = feature_window
        self.risk_free_rate = risk_free_rate
        self.shrinkage_intensity = shrinkage_intensity
        self.max_weight = max_weight
        self.l2_gamma = l2_gamma
        self.min_history_days = min_history_days
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

        # Initialize FeatureEngineer for comprehensive feature computation
        self.feature_engineer = FeatureEngineer(lookback_window=feature_window)

        # Store trained models (one per asset)
        self._models: Dict[str, object] = {}

    def _compute_features(
        self,
        prices: pd.DataFrame,
        ohlcv_data: Optional[pd.DataFrame] = None,
        indicators: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compute features using FeatureEngineer.

        Returns 160+ features including technical indicators, volume, market data, correlations.

        Args:
            prices: Historical price data
            ohlcv_data: Optional OHLCV data for technical indicators
            indicators: Optional market indicators

        Returns:
            DataFrame with comprehensive features
        """
        features = self.feature_engineer.compute_all_features(
            prices=prices,
            ohlcv_data=ohlcv_data,
            indicators=indicators,
            include_correlations=True,
            include_technical=True,
            include_volume=True,
            include_market=True
        )

        # Shift all features by 1 day to prevent look-ahead bias
        features = features.shift(1)

        return features

    def _create_model(self):
        """Create XGBoost model with conservative hyperparameters."""
        if XGBOOST_AVAILABLE:
            return xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                enable_categorical=False
            )
        else:
            # Fallback to RandomForest if XGBoost unavailable
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1
            )

    def _train_models(
        self,
        prices: pd.DataFrame,
        ohlcv_data: Optional[pd.DataFrame] = None,
        indicators: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Train XGBoost models to predict returns.

        Args:
            prices: Historical price data
            ohlcv_data: Optional OHLCV data
            indicators: Optional market indicators

        Returns:
            Dictionary of predicted annualized returns per asset
        """
        returns = prices.pct_change()
        features = self._compute_features(prices, ohlcv_data, indicators)

        print(f"  Feature count: {len(features.columns)} features")

        predictions = {}

        for ticker in prices.columns:
            # Use ALL features for predicting each ticker
            X = features.copy()
            y = returns[ticker]

            # Align X and y, drop NaNs
            combined = pd.concat([X, y.rename('target')], axis=1).dropna()

            if len(combined) < self.min_history_days:
                # Not enough data, use historical mean
                mean_return = returns[ticker].mean() * 252
                predictions[ticker] = mean_return if not np.isnan(mean_return) else 0.0
                continue

            X_train = combined.drop('target', axis=1)
            y_train = combined['target'].values

            # Train XGBoost model
            model = self._create_model()
            model.fit(X_train, y_train)
            self._models[ticker] = model

            # Predict using most recent features
            latest_features = features.iloc[-1:]

            if latest_features.isna().any().any():
                # Features have NaN, use historical mean
                mean_return = returns[ticker].mean() * 252
                predictions[ticker] = mean_return if not np.isnan(mean_return) else 0.0
            else:
                # Predict and annualize
                pred_daily = model.predict(latest_features)[0]
                predictions[ticker] = pred_daily * 252

        return predictions

    def _apply_shrinkage(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """
        Apply James-Stein style shrinkage to predicted returns.

        Args:
            predictions: Raw predicted returns per asset

        Returns:
            Shrunk predictions
        """
        if not predictions:
            return predictions

        values = np.array(list(predictions.values()))
        grand_mean = np.mean(values)

        shrunk = {}
        for ticker, pred in predictions.items():
            shrunk[ticker] = (
                (1 - self.shrinkage_intensity) * pred +
                self.shrinkage_intensity * grand_mean
            )

        return shrunk

    def allocate(
        self,
        prices: pd.DataFrame,
        current_date: Optional[pd.Timestamp] = None,
        ohlcv_data: Optional[pd.DataFrame] = None,
        indicators: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute portfolio weights using XGBoost predicted returns.

        Args:
            prices: Historical price data (up to but not including current_date)
            current_date: Date for which to compute allocation
            ohlcv_data: Optional OHLCV data for technical indicators
            indicators: Optional market indicators
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping ticker to portfolio weight
        """
        # Use only data up to current_date
        if current_date is not None:
            prices = prices.loc[:current_date]
            if ohlcv_data is not None:
                ohlcv_data = ohlcv_data.loc[:current_date]
            if indicators is not None:
                indicators = indicators.loc[:current_date]

        # Use last lookback_days of history
        if len(prices) > self.lookback_days:
            prices = prices.iloc[-self.lookback_days:]
            if ohlcv_data is not None:
                ohlcv_data = ohlcv_data.iloc[-self.lookback_days:]
            if indicators is not None:
                indicators = indicators.iloc[-self.lookback_days:]

        # Require minimum history
        if len(prices) < self.min_history_days:
            print(f"Warning: Insufficient data ({len(prices)} days), using equal weights")
            return EqualWeightStrategy().allocate(prices)

        try:
            # Step 1: Predict expected returns using XGBoost
            predicted_returns = self._train_models(prices, ohlcv_data, indicators)

            # Step 2: Apply shrinkage to predictions
            shrunk_returns = self._apply_shrinkage(predicted_returns)

            # Convert to pandas Series for PyPortfolioOpt
            mu = pd.Series(shrunk_returns)

            # Step 3: Estimate covariance using Ledoit-Wolf shrinkage
            returns_df = prices.pct_change().dropna()
            lw = LedoitWolf().fit(returns_df.values)

            # Convert to annualized covariance matrix
            cov_matrix = pd.DataFrame(
                lw.covariance_ * 252,
                index=prices.columns,
                columns=prices.columns
            )

            # Ensure max_weight is feasible
            n_assets = len(prices.columns)
            effective_max_weight = max(self.max_weight, 1.0 / n_assets + 0.01)

            # Step 4: Optimize using PyPortfolioOpt
            ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(0, effective_max_weight))

            # Add L2 regularization
            if self.l2_gamma > 0:
                ef.add_objective(objective_functions.L2_reg, gamma=self.l2_gamma)

            # Maximize Sharpe ratio
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*max_sharpe.*')
                    ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            except Exception:
                # Fallback to min volatility if max_sharpe fails
                ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(0, effective_max_weight))
                if self.l2_gamma > 0:
                    ef.add_objective(objective_functions.L2_reg, gamma=self.l2_gamma)
                ef.min_volatility()

            # Get cleaned weights
            weights = ef.clean_weights()

            # Ensure all tickers present
            for ticker in prices.columns:
                weights.setdefault(ticker, 0.0)

            self.validate_weights(weights)
            return weights

        except Exception as e:
            print(f"Warning: XGBoost optimization failed ({str(e)}), using equal weights")
            return EqualWeightStrategy().allocate(prices)


class EnsembleSharpeStrategy(BaseStrategy):
    """
    Ensemble strategy that combines predictions from multiple models.

    Averages predicted returns from Ridge, LightGBM, and RandomForest
    to produce more robust return estimates.
    """

    def __init__(
        self,
        lookback_days: int = 252,
        feature_window: int = 30,
        risk_free_rate: float = 0.0,
        shrinkage_intensity: float = 0.5,
        max_weight: float = 0.4,
        l2_gamma: float = 0.01,
        min_history_days: int = 60,
        use_sentiment: bool = False,
    ):
        """
        Initialize ensemble strategy.

        Args:
            lookback_days: Number of trading days of history.
            feature_window: Rolling window for feature computation.
            risk_free_rate: Annual risk-free rate.
            shrinkage_intensity: Shrinkage toward grand mean.
            max_weight: Maximum weight per asset.
            l2_gamma: L2 regularization on portfolio weights.
            min_history_days: Minimum rows required.
            use_sentiment: Whether to include sentiment features.
        """
        super().__init__("Ensemble Sharpe (Ridge + GBM + RF)")
        self.lookback_days = lookback_days
        self.feature_window = feature_window
        self.risk_free_rate = risk_free_rate
        self.shrinkage_intensity = shrinkage_intensity
        self.max_weight = max_weight
        self.l2_gamma = l2_gamma
        self.min_history_days = min_history_days
        self.use_sentiment = use_sentiment

    def _compute_features(
        self,
        prices: pd.DataFrame,
        sentiment_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Compute features (same as GradientBoostingSharpeStrategy)."""
        returns = prices.pct_change()
        features_list = []

        for ticker in prices.columns:
            ticker_returns = returns[ticker]

            lag_1 = ticker_returns.shift(1)
            lag_5 = ticker_returns.rolling(5).mean().shift(1)
            lag_21 = ticker_returns.rolling(21).mean().shift(1)
            momentum = ticker_returns.rolling(self.feature_window).mean().shift(1) * 252
            volatility = ticker_returns.rolling(self.feature_window).std().shift(1) * np.sqrt(252)
            rolling_sharpe = momentum / (volatility + 1e-8)

            ticker_features = pd.DataFrame({
                f'{ticker}_lag1': lag_1,
                f'{ticker}_lag5': lag_5,
                f'{ticker}_lag21': lag_21,
                f'{ticker}_momentum': momentum,
                f'{ticker}_volatility': volatility,
                f'{ticker}_sharpe': rolling_sharpe,
            })
            features_list.append(ticker_features)

        features = pd.concat(features_list, axis=1)

        if self.use_sentiment and sentiment_df is not None:
            for ticker in prices.columns:
                sentiment_col = f"{ticker}_sentiment"
                if sentiment_col in sentiment_df.columns:
                    aligned_sentiment = sentiment_df[sentiment_col].reindex(prices.index).fillna(0)
                    features[sentiment_col] = aligned_sentiment.shift(1)
                    features[f"{ticker}_sentiment_momentum"] = (
                        aligned_sentiment - aligned_sentiment.shift(5)
                    ).shift(1).fillna(0)

        return features

    def _train_ensemble(
        self,
        prices: pd.DataFrame,
        sentiment_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Train ensemble of models and average predictions."""
        returns = prices.pct_change()
        features = self._compute_features(prices, sentiment_df)

        predictions = {}

        for ticker in prices.columns:
            X = features.copy()
            y = returns[ticker]

            combined = pd.concat([X, y.rename('target')], axis=1).dropna()

            if len(combined) < self.min_history_days:
                mean_return = returns[ticker].mean() * 252
                predictions[ticker] = mean_return if not np.isnan(mean_return) else 0.0
                continue

            X_train = combined.drop('target', axis=1).values
            y_train = combined['target'].values
            latest_features = features.iloc[-1:].values

            if np.any(np.isnan(latest_features)):
                mean_return = returns[ticker].mean() * 252
                predictions[ticker] = mean_return if not np.isnan(mean_return) else 0.0
                continue

            # Train multiple models
            model_predictions = []

            # 1. Ridge regression
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)
            model_predictions.append(ridge.predict(latest_features)[0] * 252)

            # 2. Random Forest
            rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            model_predictions.append(rf.predict(latest_features)[0] * 252)

            # 3. LightGBM (if available)
            if LIGHTGBM_AVAILABLE:
                gbm = lgb.LGBMRegressor(n_estimators=50, max_depth=5, learning_rate=0.05, verbose=-1)
                gbm.fit(X_train, y_train)
                model_predictions.append(gbm.predict(latest_features)[0] * 252)

            # Average predictions
            predictions[ticker] = np.mean(model_predictions)

        return predictions

    def _apply_shrinkage(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """Apply shrinkage to predictions."""
        if not predictions:
            return predictions

        values = np.array(list(predictions.values()))
        grand_mean = np.mean(values)

        shrunk = {}
        for ticker, pred in predictions.items():
            shrunk[ticker] = (
                (1 - self.shrinkage_intensity) * pred +
                self.shrinkage_intensity * grand_mean
            )
        return shrunk

    def allocate(
        self,
        prices: pd.DataFrame,
        current_date: Optional[pd.Timestamp] = None,
        ohlcv_data: Optional[pd.DataFrame] = None,
        indicators: Optional[pd.DataFrame] = None,
        sentiment_df: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Compute portfolio weights using ensemble predicted returns."""
        if current_date is not None:
            prices = prices.loc[:current_date]

        if len(prices) > self.lookback_days:
            prices = prices.iloc[-self.lookback_days:]

        if len(prices) < self.min_history_days:
            return EqualWeightStrategy().allocate(prices)

        try:
            predicted_returns = self._train_ensemble(prices, sentiment_df)
            shrunk_returns = self._apply_shrinkage(predicted_returns)
            mu = pd.Series(shrunk_returns)

            returns_df = prices.pct_change().dropna()
            lw = LedoitWolf().fit(returns_df.values)
            cov_matrix = pd.DataFrame(
                lw.covariance_ * 252,
                index=prices.columns,
                columns=prices.columns
            )

            n_assets = len(prices.columns)
            effective_max_weight = max(self.max_weight, 1.0 / n_assets + 0.01)

            ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(0, effective_max_weight))
            if self.l2_gamma > 0:
                ef.add_objective(objective_functions.L2_reg, gamma=self.l2_gamma)

            try:
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            except Exception:
                ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(0, effective_max_weight))
                if self.l2_gamma > 0:
                    ef.add_objective(objective_functions.L2_reg, gamma=self.l2_gamma)
                ef.min_volatility()

            weights = ef.clean_weights()
            for ticker in prices.columns:
                weights.setdefault(ticker, 0.0)

            self.validate_weights(weights)
            return weights

        except Exception as e:
            print(f"Warning: Ensemble optimization failed ({str(e)}), using equal weights")
            return EqualWeightStrategy().allocate(prices)


class BuyAndHoldStrategy(BaseStrategy):
    """
    Buy and hold strategy with initial allocation.

    Useful as a benchmark - allocate once and never rebalance.
    """

    def __init__(self, initial_weights: Optional[Dict[str, float]] = None):
        """
        Initialize buy and hold strategy.

        Args:
            initial_weights: Initial allocation (if None, uses equal weight)
        """
        super().__init__("Buy and Hold")
        self.initial_weights = initial_weights
        self._weights_set = False

    def allocate(
        self,
        prices: pd.DataFrame,
        current_date: Optional[pd.Timestamp] = None,
        ohlcv_data: Optional[pd.DataFrame] = None,
        indicators: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Return fixed initial weights (no rebalancing).

        Args:
            prices: Historical price data
            current_date: Unused
            ohlcv_data: Unused
            indicators: Unused
            **kwargs: Unused

        Returns:
            Dictionary of fixed weights
        """
        if not self._weights_set:
            if self.initial_weights is None:
                # Use equal weight
                self.initial_weights = EqualWeightStrategy().allocate(prices)
            self._weights_set = True

        return self.initial_weights.copy()


class StaticStrategy(BaseStrategy):
    """
    Static allocation strategy (e.g., 60/40 stocks/bonds).

    Rebalances periodically to maintain fixed target weights.
    """

    def __init__(self, target_weights: Dict[str, float]):
        """
        Initialize static strategy.

        Args:
            target_weights: Target allocation (must sum to 1.0)
        """
        super().__init__("Static Allocation")
        self.target_weights = target_weights
        self.validate_weights(target_weights)

    def allocate(
        self,
        prices: pd.DataFrame,
        current_date: Optional[pd.Timestamp] = None,
        ohlcv_data: Optional[pd.DataFrame] = None,
        indicators: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Return fixed target weights.

        Args:
            prices: Historical price data
            current_date: Unused
            ohlcv_data: Unused
            indicators: Unused
            **kwargs: Unused

        Returns:
            Dictionary of target weights
        """
        # Ensure all tickers in prices have weights
        weights = {}
        for ticker in prices.columns:
            if ticker in self.target_weights:
                weights[ticker] = self.target_weights[ticker]
            else:
                weights[ticker] = 0.0

        self.validate_weights(weights)
        return weights


def create_60_40_strategy(stock_tickers: list, bond_tickers: list) -> StaticStrategy:
    """
    Create a 60/40 stocks/bonds strategy.

    Args:
        stock_tickers: List of stock ETF tickers
        bond_tickers: List of bond ETF tickers

    Returns:
        StaticStrategy with 60/40 allocation
    """
    weights = {}

    # Allocate 60% equally among stocks
    stock_weight = 0.6 / len(stock_tickers)
    for ticker in stock_tickers:
        weights[ticker] = stock_weight

    # Allocate 40% equally among bonds
    bond_weight = 0.4 / len(bond_tickers)
    for ticker in bond_tickers:
        weights[ticker] = bond_weight

    return StaticStrategy(weights)

class RiskAdjustedLSTMStrategy(BaseStrategy):
    def __init__(self,lookback=30,hidden_dim=32,epochs=20,lr=1e-3):
        super().__init__("Risk Adjusted LSTM")
        self.lookback=lookback
        self.hidden_dim=hidden_dim
        self.epochs=epochs
        self.lr=lr
        self.models: Dict[str,torch.nn.Module]={} #this is to ensure one lstm for each ticker
        
    def _prepare_data(self,series):
        data=series.pct_change().dropna().values.astype(np.float32)
        data = (data - data.mean()) / (data.std() + 1e-6)
        X,Y=[],[]
        for i in range(len(data)-self.lookback-1):
            X.append(data[i:i+self.lookback])
            Y.append(data[i+self.lookback]) #this is for next day returns
        if len(X)==0:
            return torch.empty(0), torch.empty(0)

        X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
        Y = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(-1)

        return X,Y
    def _train_model(self,series):
        self.series=pd.Series
        X,Y=self._prepare_data(series)
        if len(X)<10:
            model=LSTMReturnPredictor(input_dim=1,hidden_dim=self.hidden_dim) #if there isn't enough data
            return model
        model=LSTMReturnPredictor(input_dim=1,hidden_dim=self.hidden_dim)
        optimizer=torch.optim.Adam(model.parameters(),lr=self.lr)
        loss_fn=torch.nn.MSELoss()
        ds=TensorDataset(X,Y)
        dl=DataLoader(ds,batch_size=32,shuffle=True)
        for i in range(self.epochs):
            for x_batch,y_batch in dl:
                optimizer.zero_grad()
                pred=model(x_batch)
                loss=loss_fn(pred,y_batch)
                loss.backward()
                optimizer.step()
        return model
    def _predict_next_return(self,model,series:pd.Series):
        data=series.pct_change().dropna().values.astype(np.float32)
        if len(data)<self.lookback:
            return 0.0
        
        window=torch.tensor(data[-self.lookback:]).unsqueeze(0).unsqueeze(-1)
        return float(model(window).item())
    def _calculate_recent_volatility(self,series:pd.Series,window=20):
        returns=series.pct_change().dropna()
        if len(returns)<window:
            return returns.std() if len(returns) > 0 else 1e-6
        return returns.tail(window).std()
    
    def allocate(self, prices, current_date=None, ohlcv_data=None, indicators=None, **kwargs):
        if current_date is not None:
            price=prices.loc[:current_date]
        else:
            price = prices
        tickers=list(prices.columns)
        if not self.models:
            for ticker in tickers:
                self.models[ticker]=self._train_model(prices[ticker])
        
        pred_returns={}
        vol={}
        for ticker in tickers:
            series=price[ticker]
            model=self.models[ticker]
            prediction=self._predict_next_return(model,series)
            volatility=self._calculate_recent_volatility(series)
            pred_returns[ticker]=prediction
            vol[ticker]=max(volatility,1e-6)
            
        scores={}
        #print("\nPredicted next-day returns:")
        #for t, r in pred_returns.items():
            #print(t, r)

        for ticker in tickers:
            stock_return = pred_returns[ticker]
            scores[ticker]=stock_return/vol[ticker]
            
        score_values=np.array(list(scores.values()))
        
        if score_values.sum()<=0:
            weights={t:1/len(tickers)for t in tickers}
        else:
            exp_scores = np.exp(score_values - np.max(score_values))
            softmax_scores = exp_scores / exp_scores.sum()
            weights = {t: softmax_scores[i] for i, t in enumerate(tickers)}

        self.validate_weights(weights)
        return weights
        
if __name__ == "__main__":
    # Example usage
    from data import load_default_etfs, ETFDataLoader

    # Load data
    prices = load_default_etfs()
    loader = ETFDataLoader()
    train, val, test = loader.split_train_val_test(prices)

    print("\n" + "="*80)
    print("Testing Strategies on Training Data")
    print("="*80)

    # Test equal weight
    ew_strategy = EqualWeightStrategy()
    ew_weights = ew_strategy.allocate(train)
    print(f"\n{ew_strategy.name}:")
    for ticker, weight in ew_weights.items():
        print(f"  {ticker}: {weight:.2%}")

    # Test mean-variance
    mv_strategy = MeanVarianceStrategy(lookback_days=252)
    mv_weights = mv_strategy.allocate(train)
    print(f"\n{mv_strategy.name}:")
    for ticker, weight in mv_weights.items():
        print(f"  {ticker}: {weight:.2%}")

    # Test 60/40 (VTI, QQQ as stocks, BND as bonds)
    static_60_40 = create_60_40_strategy(['VTI', 'QQQ'], ['BND'])
    static_weights = static_60_40.allocate(train)
    print(f"\n{static_60_40.name} (60/40):")
    for ticker, weight in static_weights.items():
        print(f"  {ticker}: {weight:.2%}")
