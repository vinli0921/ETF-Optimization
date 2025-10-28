"""
Backtesting engine for portfolio strategies.

Simulates portfolio performance over time with realistic rebalancing,
transaction costs, and strict no-look-ahead bias.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import warnings

from strategies import BaseStrategy


class PortfolioBacktest:
    """
    Backtest a portfolio strategy over historical data.

    Handles:
    - Periodic rebalancing
    - Transaction costs
    - Position tracking
    - Strict no-look-ahead (only uses past data)
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,  # 0.1% per transaction
        rebalance_frequency: str = 'M'  # 'D'=daily, 'W'=weekly, 'M'=monthly
    ):
        """
        Initialize backtesting engine.

        Args:
            initial_capital: Starting portfolio value
            transaction_cost: Transaction cost as decimal (e.g., 0.001 = 0.1%)
            rebalance_frequency: Pandas frequency string for rebalancing
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.rebalance_frequency = rebalance_frequency

        # Results storage
        self.portfolio_values = None
        self.allocations = None
        self.positions = None
        self.turnover_history = None

    def run(
        self,
        strategy: BaseStrategy,
        prices: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **strategy_kwargs
    ) -> pd.Series:
        """
        Run backtest for a given strategy.

        Args:
            strategy: Strategy to backtest
            prices: Historical price data with DatetimeIndex
            start_date: Start date for backtest (None = use first date)
            end_date: End date for backtest (None = use last date)
            **strategy_kwargs: Additional arguments to pass to strategy.allocate()

        Returns:
            Series of portfolio values over time
        """
        # Filter date range
        if start_date:
            prices = prices[start_date:]
        if end_date:
            prices = prices[:end_date]

        if len(prices) == 0:
            raise ValueError("No price data in specified date range")

        # Identify rebalancing dates
        rebalance_dates = self._get_rebalance_dates(prices.index)

        print(f"Running backtest for {strategy.name}")
        print(f"  Period: {prices.index[0].date()} to {prices.index[-1].date()}")
        print(f"  Rebalancing: {self.rebalance_frequency} ({len(rebalance_dates)} times)")

        # Initialize tracking variables
        portfolio_values = []
        allocations = []
        positions = {}  # Current shares held for each ticker
        turnover_history = []
        cash = self.initial_capital

        # Track previous weights for turnover calculation
        prev_weights = None

        # Iterate through all dates
        for i, date in enumerate(prices.index):
            # Check if this is a rebalancing date
            if date in rebalance_dates:
                # Get historical prices up to (but not including) current date
                # This ensures no look-ahead bias
                historical_prices = prices[:date].iloc[:-1]  # Exclude current date

                if len(historical_prices) < 10:
                    # Not enough history, use equal weights
                    tickers = prices.columns
                    weights = {ticker: 1.0 / len(tickers) for ticker in tickers}
                else:
                    # Compute optimal weights using strategy
                    weights = strategy.allocate(
                        historical_prices,
                        current_date=date,
                        **strategy_kwargs
                    )

                # Calculate turnover if not first rebalancing
                if prev_weights is not None:
                    turnover = self._calculate_turnover(prev_weights, weights)
                    turnover_history.append(turnover)
                else:
                    turnover = 0.0

                # Rebalance portfolio
                positions, cash = self._rebalance(
                    positions,
                    cash,
                    weights,
                    prices.loc[date],
                    turnover
                )

                # Store allocation
                allocations.append({
                    'date': date,
                    **weights
                })

                prev_weights = weights

            # Calculate portfolio value
            position_value = sum(
                shares * prices.loc[date, ticker]
                for ticker, shares in positions.items()
            )
            total_value = position_value + cash

            portfolio_values.append({
                'date': date,
                'value': total_value,
                'position_value': position_value,
                'cash': cash
            })

        # Convert results to DataFrames/Series
        self.portfolio_values = pd.DataFrame(portfolio_values).set_index('date')
        self.allocations = pd.DataFrame(allocations).set_index('date')
        self.turnover_history = pd.Series(turnover_history, name='turnover')

        # Return just the portfolio value series
        return self.portfolio_values['value']

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
        """
        Get list of rebalancing dates based on frequency.

        Args:
            dates: All available dates

        Returns:
            List of dates on which to rebalance
        """
        if self.rebalance_frequency == 'D':
            # Rebalance daily
            return list(dates)

        # Use pandas resampling to find period boundaries
        resampled = pd.Series(1, index=dates).resample(self.rebalance_frequency).first()
        rebalance_dates = resampled.index.tolist()

        # Ensure dates are within available range
        rebalance_dates = [d for d in rebalance_dates if d in dates]

        return rebalance_dates

    def _calculate_turnover(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float]
    ) -> float:
        """
        Calculate turnover between two weight allocations.

        Args:
            old_weights: Previous allocation
            new_weights: New allocation

        Returns:
            Turnover as a decimal (sum of absolute changes / 2)
        """
        all_tickers = set(old_weights.keys()) | set(new_weights.keys())

        total_change = 0.0
        for ticker in all_tickers:
            old_w = old_weights.get(ticker, 0.0)
            new_w = new_weights.get(ticker, 0.0)
            total_change += abs(new_w - old_w)

        # Divide by 2 because buying X% of one asset means selling X% of another
        turnover = total_change / 2.0

        return turnover

    def _rebalance(
        self,
        positions: Dict[str, float],
        cash: float,
        target_weights: Dict[str, float],
        current_prices: pd.Series,
        turnover: float
    ) -> Tuple[Dict[str, float], float]:
        """
        Rebalance portfolio to target weights.

        Args:
            positions: Current positions (shares per ticker)
            cash: Current cash balance
            target_weights: Target allocation
            current_prices: Current prices for each ticker
            turnover: Turnover for this rebalance

        Returns:
            Tuple of (new_positions, new_cash)
        """
        # Calculate current portfolio value
        position_value = sum(
            shares * current_prices[ticker]
            for ticker, shares in positions.items()
            if ticker in current_prices
        )
        total_value = position_value + cash

        # Apply transaction costs based on turnover
        transaction_costs = total_value * turnover * self.transaction_cost
        total_value -= transaction_costs

        # Calculate target positions
        new_positions = {}
        total_allocated = 0.0

        for ticker, weight in target_weights.items():
            if weight > 0 and ticker in current_prices:
                target_value = total_value * weight
                shares = target_value / current_prices[ticker]
                new_positions[ticker] = shares
                total_allocated += shares * current_prices[ticker]

        # Remaining cash after allocation
        new_cash = total_value - total_allocated

        return new_positions, new_cash

    def get_summary(self) -> Dict:
        """
        Get summary of backtest results.

        Returns:
            Dictionary with key metrics and statistics
        """
        if self.portfolio_values is None:
            raise ValueError("Must run backtest before getting summary")

        summary = {
            'initial_value': self.portfolio_values['value'].iloc[0],
            'final_value': self.portfolio_values['value'].iloc[-1],
            'peak_value': self.portfolio_values['value'].max(),
            'total_rebalances': len(self.allocations),
            'avg_turnover': self.turnover_history.mean() if len(self.turnover_history) > 0 else 0.0,
            'avg_cash_holding': self.portfolio_values['cash'].mean(),
        }

        return summary


def compare_strategies(
    strategies: Dict[str, BaseStrategy],
    prices: pd.DataFrame,
    initial_capital: float = 100000.0,
    transaction_cost: float = 0.001,
    rebalance_frequency: str = 'M',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Tuple[Dict[str, pd.Series], Dict[str, pd.DataFrame]]:
    """
    Run backtest for multiple strategies and compare results.

    Args:
        strategies: Dictionary of strategy_name -> strategy
        prices: Historical price data
        initial_capital: Starting capital
        transaction_cost: Transaction cost per trade
        rebalance_frequency: Rebalancing frequency
        start_date: Start date for backtest
        end_date: End date for backtest

    Returns:
        Tuple of (portfolio_values_dict, allocations_dict)
    """
    results = {}
    allocations = {}

    for name, strategy in strategies.items():
        backtest = PortfolioBacktest(
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            rebalance_frequency=rebalance_frequency
        )

        portfolio_values = backtest.run(
            strategy,
            prices,
            start_date=start_date,
            end_date=end_date
        )

        results[name] = portfolio_values
        allocations[name] = backtest.allocations

        print(f"  Final value: ${portfolio_values.iloc[-1]:,.2f}")
        print()

    return results, allocations


if __name__ == "__main__":
    # Example usage
    from data import load_default_etfs, ETFDataLoader
    from strategies import EqualWeightStrategy, MeanVarianceStrategy

    # Load data
    prices = load_default_etfs()
    loader = ETFDataLoader()
    train, val, test = loader.split_train_val_test(prices)

    print("\n" + "="*80)
    print("Backtesting on Training Data")
    print("="*80)

    # Define strategies
    strategies = {
        'Equal Weight': EqualWeightStrategy(),
        'Mean-Variance': MeanVarianceStrategy(lookback_days=252),
    }

    # Run comparison
    results, allocations = compare_strategies(
        strategies,
        train,
        rebalance_frequency='M'
    )

    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)
    for name, values in results.items():
        total_return = (values.iloc[-1] / values.iloc[0] - 1) * 100
        print(f"{name:20s}: {total_return:>6.2f}% total return")
