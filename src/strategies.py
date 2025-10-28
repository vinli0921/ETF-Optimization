"""
Portfolio allocation strategies.

Implements baseline strategies including equal weight and mean-variance optimization.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Optional, Dict
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier


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

    Uses historical returns and covariances to find the maximum Sharpe ratio portfolio.
    """

    def __init__(
        self,
        lookback_days: int = 252,
        risk_free_rate: float = 0.02,
        method: str = "max_sharpe"
    ):
        """
        Initialize mean-variance strategy.

        Args:
            lookback_days: Number of days of history to use for estimation
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            method: Optimization method ('max_sharpe', 'min_volatility', 'efficient_risk')
        """
        super().__init__("Mean-Variance Optimization")
        self.lookback_days = lookback_days
        self.risk_free_rate = risk_free_rate
        self.method = method

    def allocate(
        self,
        prices: pd.DataFrame,
        current_date: Optional[pd.Timestamp] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute optimal portfolio weights using mean-variance optimization.

        Args:
            prices: Historical price data
            current_date: Date for which to compute allocation (uses data up to this date)
            **kwargs: Additional parameters

        Returns:
            Dictionary of optimal weights
        """
        # Use only data up to current_date for strict no-look-ahead
        if current_date is not None:
            prices = prices[:current_date]

        # Use last lookback_days of history
        if len(prices) > self.lookback_days:
            prices = prices.iloc[-self.lookback_days:]

        # Need at least 30 days for reasonable estimation
        if len(prices) < 30:
            # Fall back to equal weight if insufficient data
            print(f"Warning: Insufficient data ({len(prices)} days), using equal weights")
            return EqualWeightStrategy().allocate(prices)

        try:
            # Compute expected returns using historical mean
            mu = expected_returns.mean_historical_return(prices, frequency=252)

            # Compute sample covariance matrix
            S = risk_models.sample_cov(prices, frequency=252)

            # Set up efficient frontier
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))  # Long-only

            # Optimize based on method
            if self.method == "max_sharpe":
                weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            elif self.method == "min_volatility":
                weights = ef.min_volatility()
            elif self.method == "efficient_risk":
                target_volatility = kwargs.get("target_volatility", 0.15)
                weights = ef.efficient_risk(target_volatility)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Clean weights (remove tiny values)
            weights = ef.clean_weights()

            # Ensure all tickers are present (PyPortfolioOpt may drop some)
            for ticker in prices.columns:
                if ticker not in weights:
                    weights[ticker] = 0.0

            self.validate_weights(weights)

            return weights

        except Exception as e:
            # If optimization fails, fall back to equal weight
            print(f"Warning: Optimization failed ({str(e)}), using equal weights")
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
