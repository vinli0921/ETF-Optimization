"""
Portfolio performance metrics.

Computes various risk-adjusted return metrics for portfolio evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def calculate_returns(portfolio_values: pd.Series) -> pd.Series:
    """
    Calculate returns from portfolio values.

    Args:
        portfolio_values: Series of portfolio values over time

    Returns:
        Series of returns
    """
    return portfolio_values.pct_change()


def calculate_total_return(portfolio_values: pd.Series) -> float:
    """
    Calculate total return over the period.

    Args:
        portfolio_values: Series of portfolio values

    Returns:
        Total return as a decimal (e.g., 0.5 = 50% return)
    """
    return (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1


def calculate_annualized_return(portfolio_values: pd.Series) -> float:
    """
    Calculate annualized return (CAGR).

    Args:
        portfolio_values: Series of portfolio values with DatetimeIndex

    Returns:
        Annualized return as a decimal
    """
    total_return = calculate_total_return(portfolio_values)
    n_years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25

    if n_years <= 0:
        return 0.0

    annualized = (1 + total_return) ** (1 / n_years) - 1
    return annualized


def calculate_annualized_volatility(portfolio_values: pd.Series) -> float:
    """
    Calculate annualized volatility (standard deviation of returns).

    Args:
        portfolio_values: Series of portfolio values

    Returns:
        Annualized volatility as a decimal
    """
    returns = calculate_returns(portfolio_values)
    daily_vol = returns.std()

    # Annualize assuming 252 trading days
    annualized_vol = daily_vol * np.sqrt(252)

    return annualized_vol


def calculate_sharpe_ratio(
    portfolio_values: pd.Series,
    risk_free_rate: float = 0.02
) -> float:
    """
    Calculate Sharpe ratio (risk-adjusted return).

    Args:
        portfolio_values: Series of portfolio values
        risk_free_rate: Annual risk-free rate (default 2%)

    Returns:
        Sharpe ratio
    """
    ann_return = calculate_annualized_return(portfolio_values)
    ann_vol = calculate_annualized_volatility(portfolio_values)

    if ann_vol == 0:
        return 0.0

    sharpe = (ann_return - risk_free_rate) / ann_vol
    return sharpe


def calculate_sortino_ratio(
    portfolio_values: pd.Series,
    risk_free_rate: float = 0.02
) -> float:
    """
    Calculate Sortino ratio (return vs downside deviation).

    Like Sharpe ratio but only penalizes downside volatility.

    Args:
        portfolio_values: Series of portfolio values
        risk_free_rate: Annual risk-free rate (default 2%)

    Returns:
        Sortino ratio
    """
    returns = calculate_returns(portfolio_values)
    ann_return = calculate_annualized_return(portfolio_values)

    # Downside deviation (only negative returns)
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return np.inf

    downside_std = downside_returns.std()
    downside_vol = downside_std * np.sqrt(252)

    if downside_vol == 0:
        return np.inf

    sortino = (ann_return - risk_free_rate) / downside_vol
    return sortino


def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    """
    Calculate maximum drawdown (largest peak-to-trough decline).

    Args:
        portfolio_values: Series of portfolio values

    Returns:
        Maximum drawdown as a positive decimal (e.g., 0.3 = 30% drawdown)
    """
    # Calculate cumulative max (running peak)
    cumulative_max = portfolio_values.cummax()

    # Calculate drawdown at each point
    drawdown = (portfolio_values - cumulative_max) / cumulative_max

    # Return the maximum (most negative) drawdown as positive value
    max_dd = abs(drawdown.min())

    return max_dd


def calculate_calmar_ratio(portfolio_values: pd.Series) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        portfolio_values: Series of portfolio values

    Returns:
        Calmar ratio
    """
    ann_return = calculate_annualized_return(portfolio_values)
    max_dd = calculate_max_drawdown(portfolio_values)

    if max_dd == 0:
        return np.inf

    calmar = ann_return / max_dd
    return calmar


def calculate_win_rate(portfolio_values: pd.Series) -> float:
    """
    Calculate percentage of periods with positive returns.

    Args:
        portfolio_values: Series of portfolio values

    Returns:
        Win rate as a decimal (e.g., 0.6 = 60% of periods were positive)
    """
    returns = calculate_returns(portfolio_values).dropna()

    if len(returns) == 0:
        return 0.0

    win_rate = (returns > 0).sum() / len(returns)
    return win_rate


def calculate_downside_periods(portfolio_values: pd.Series) -> int:
    """
    Count number of periods with negative returns.

    Args:
        portfolio_values: Series of portfolio values

    Returns:
        Number of negative return periods
    """
    returns = calculate_returns(portfolio_values).dropna()
    return (returns < 0).sum()


def calculate_turnover(allocations: pd.DataFrame) -> float:
    """
    Calculate average portfolio turnover (how much trading occurs).

    Args:
        allocations: DataFrame with dates as index and tickers as columns,
                    values are portfolio weights at each rebalance

    Returns:
        Average turnover per rebalance as a decimal
    """
    if len(allocations) <= 1:
        return 0.0

    # Calculate change in weights at each rebalance
    weight_changes = allocations.diff().abs()

    # Turnover is sum of absolute weight changes divided by 2
    # (buying X% of one asset means selling X% of others)
    turnover_per_period = weight_changes.sum(axis=1) / 2

    # Return average turnover
    avg_turnover = turnover_per_period.mean()

    return avg_turnover


def calculate_all_metrics(
    portfolio_values: pd.Series,
    allocations: Optional[pd.DataFrame] = None,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Calculate all performance metrics for a portfolio.

    Args:
        portfolio_values: Series of portfolio values over time
        allocations: Optional DataFrame of portfolio weights over time
        risk_free_rate: Annual risk-free rate (default 2%)

    Returns:
        Dictionary of metric name -> value
    """
    metrics = {
        'Total Return': calculate_total_return(portfolio_values),
        'Annualized Return': calculate_annualized_return(portfolio_values),
        'Annualized Volatility': calculate_annualized_volatility(portfolio_values),
        'Sharpe Ratio': calculate_sharpe_ratio(portfolio_values, risk_free_rate),
        'Sortino Ratio': calculate_sortino_ratio(portfolio_values, risk_free_rate),
        'Max Drawdown': calculate_max_drawdown(portfolio_values),
        'Calmar Ratio': calculate_calmar_ratio(portfolio_values),
        'Win Rate': calculate_win_rate(portfolio_values),
        'Downside Periods': calculate_downside_periods(portfolio_values),
    }

    # Add turnover if allocations provided
    if allocations is not None:
        metrics['Avg Turnover'] = calculate_turnover(allocations)

    return metrics


def compare_strategies(
    results: Dict[str, pd.Series],
    allocations: Optional[Dict[str, pd.DataFrame]] = None,
    risk_free_rate: float = 0.02
) -> pd.DataFrame:
    """
    Compare multiple strategies using all metrics.

    Args:
        results: Dictionary of strategy_name -> portfolio_values
        allocations: Optional dictionary of strategy_name -> allocations
        risk_free_rate: Annual risk-free rate

    Returns:
        DataFrame comparing all strategies across all metrics
    """
    comparison = {}

    for strategy_name, portfolio_values in results.items():
        alloc = allocations.get(strategy_name) if allocations else None
        metrics = calculate_all_metrics(portfolio_values, alloc, risk_free_rate)
        comparison[strategy_name] = metrics

    df = pd.DataFrame(comparison).T

    return df


def format_metrics_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format metrics table for display with appropriate units.

    Args:
        metrics_df: DataFrame of metrics (from compare_strategies)

    Returns:
        Formatted DataFrame
    """
    formatted = metrics_df.copy()

    # Convert to percentages
    pct_cols = ['Total Return', 'Annualized Return', 'Annualized Volatility',
                'Max Drawdown', 'Win Rate', 'Avg Turnover']

    for col in pct_cols:
        if col in formatted.columns:
            formatted[col] = formatted[col] * 100

    return formatted


if __name__ == "__main__":
    # Example usage with synthetic data
    dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')

    # Synthetic portfolio values (growing with some volatility)
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, len(dates))  # Daily returns
    portfolio_values = pd.Series(100 * (1 + returns).cumprod(), index=dates)

    print("\n" + "="*80)
    print("Example Portfolio Metrics")
    print("="*80)

    metrics = calculate_all_metrics(portfolio_values)

    for metric_name, value in metrics.items():
        if 'Return' in metric_name or 'Volatility' in metric_name or 'Drawdown' in metric_name or 'Rate' in metric_name or 'Turnover' in metric_name:
            print(f"{metric_name:25s}: {value:>8.2%}")
        elif 'Periods' in metric_name:
            print(f"{metric_name:25s}: {value:>8.0f}")
        else:
            print(f"{metric_name:25s}: {value:>8.2f}")
