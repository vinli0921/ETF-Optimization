"""
Portfolio allocation strategies.

Implementst baseline strategies including equal weight and mean-variance optimization.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, Dict
from pypfopt import expected_returns, risk_models, objective_functions
from pypfopt.efficient_frontier import EfficientFrontier
from sklearn.linear_model import Ridge
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import TimeSeriesSplit
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
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute portfolio weights for given price data.

        Args:
            prices: Historical price data (up to but not including current_date)
            current_date: Date for which to compute allocation
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
        **kwargs
    ) -> Dict[str, float]:
        """
        Allocate equal weights to all tickers.

        Args:
            prices: Historical price data
            current_date: Unused for this strategy
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

        # Store trained models (one per asset)
        self._models: Dict[str, Ridge] = {}

    def _compute_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for prediction.

        Features per asset:
        - Lagged returns (1, 5, 21 days)
        - Rolling momentum (annualized)
        - Rolling volatility (annualized)
        - Rolling Sharpe ratio

        Args:
            prices: Historical price data

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
            rolling_sharpe = momentum / volatility

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
        return features

    def _train_models(self, prices: pd.DataFrame) -> Dict[str, float]:
        """
        Train Ridge regression models to predict returns.

        Args:
            prices: Historical price data

        Returns:
            Dictionary of predicted annualized returns per asset
        """
        returns = prices.pct_change()
        features = self._compute_features(prices)

        predictions = {}

        for ticker in prices.columns:
            # Get features for this asset
            ticker_feature_cols = [col for col in features.columns if col.startswith(ticker)]
            X = features[ticker_feature_cols]

            # Target: next-day return
            y = returns[ticker]

            # Align X and y, drop NaNs
            combined = pd.concat([X, y.rename('target')], axis=1).dropna()

            if len(combined) < self.min_history_days:
                # Not enough data, use historical mean
                mean_return = returns[ticker].mean() * 252
                predictions[ticker] = mean_return if not np.isnan(mean_return) else 0.0
                continue

            X_train = combined[ticker_feature_cols].values
            y_train = combined['target'].values

            # Train Ridge regression
            model = Ridge(alpha=self.ridge_alpha)
            model.fit(X_train, y_train)
            self._models[ticker] = model

            # Predict using most recent features
            latest_features = X.iloc[-1:].values

            if np.any(np.isnan(latest_features)):
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
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute portfolio weights using predicted returns.

        Args:
            prices: Historical price data (up to but not including current_date)
            current_date: Date for which to compute allocation
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
            # Step 1: Predict expected returns using Ridge regression
            predicted_returns = self._train_models(prices)

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
        **kwargs
    ) -> Dict[str, float]:
        """
        Return fixed initial weights (no rebalancing).

        Args:
            prices: Historical price data
            current_date: Unused
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
        **kwargs
    ) -> Dict[str, float]:
        """
        Return fixed target weights.

        Args:
            prices: Historical price data
            current_date: Unused
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
    
    def allocate(self,prices,current_date=None,**kwargs):
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
