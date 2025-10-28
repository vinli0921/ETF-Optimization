"""
Visualization utilities for portfolio analysis.

Provides plotting functions for equity curves, allocations, drawdowns,
and other portfolio metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List
import warnings

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_equity_curves(
    results: Dict[str, pd.Series],
    title: str = "Portfolio Performance Comparison",
    figsize: tuple = (14, 7),
    log_scale: bool = False
):
    """
    Plot equity curves for multiple strategies.

    Args:
        results: Dictionary of strategy_name -> portfolio_values
        title: Plot title
        figsize: Figure size
        log_scale: Use logarithmic scale for y-axis
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, values in results.items():
        # Normalize to percentage return from initial investment
        normalized = (values / values.iloc[0] - 1) * 100
        ax.plot(normalized.index, normalized.values, label=name, linewidth=2)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Return (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale('log')

    plt.tight_layout()
    return fig


def plot_drawdown(
    portfolio_values: pd.Series,
    title: str = "Drawdown Over Time",
    figsize: tuple = (14, 5)
):
    """
    Plot drawdown (decline from peak) over time.

    Args:
        portfolio_values: Series of portfolio values
        title: Plot title
        figsize: Figure size
    """
    # Calculate running maximum
    running_max = portfolio_values.cummax()

    # Calculate drawdown
    drawdown = (portfolio_values - running_max) / running_max * 100

    fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(drawdown.index, drawdown.values, 0,
                     alpha=0.3, color='red', label='Drawdown')
    ax.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1.5)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    plt.tight_layout()
    return fig


def plot_multiple_drawdowns(
    results: Dict[str, pd.Series],
    title: str = "Drawdown Comparison",
    figsize: tuple = (14, 6)
):
    """
    Plot drawdowns for multiple strategies.

    Args:
        results: Dictionary of strategy_name -> portfolio_values
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, values in results.items():
        # Calculate drawdown
        running_max = values.cummax()
        drawdown = (values - running_max) / running_max * 100

        ax.plot(drawdown.index, drawdown.values, label=name, linewidth=2)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    plt.tight_layout()
    return fig


def plot_allocation_over_time(
    allocations: pd.DataFrame,
    title: str = "Portfolio Allocation Over Time",
    figsize: tuple = (14, 7)
):
    """
    Plot stacked area chart of portfolio allocations.

    Args:
        allocations: DataFrame with dates as index, tickers as columns
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create stacked area plot
    allocations_pct = allocations * 100  # Convert to percentages
    allocations_pct.plot(kind='area', stacked=True, ax=ax, alpha=0.7)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Allocation (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=10)
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_allocation_heatmap(
    allocations: pd.DataFrame,
    title: str = "Allocation Heatmap",
    figsize: tuple = (14, 6)
):
    """
    Plot heatmap of allocations over time.

    Args:
        allocations: DataFrame with dates as index, tickers as columns
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Transpose so tickers are rows
    allocations_pct = allocations.T * 100

    # Create heatmap
    sns.heatmap(
        allocations_pct,
        cmap='YlOrRd',
        cbar_kws={'label': 'Allocation (%)'},
        ax=ax,
        vmin=0,
        vmax=100
    )

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Ticker', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Reduce number of x-tick labels for readability
    n_ticks = min(10, len(allocations))
    tick_indices = np.linspace(0, len(allocations)-1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([allocations.index[i].strftime('%Y-%m-%d') for i in tick_indices],
                       rotation=45, ha='right')

    plt.tight_layout()
    return fig


def plot_rolling_sharpe(
    results: Dict[str, pd.Series],
    window: int = 252,
    title: str = "Rolling Sharpe Ratio",
    figsize: tuple = (14, 6),
    risk_free_rate: float = 0.02
):
    """
    Plot rolling Sharpe ratio for multiple strategies.

    Args:
        results: Dictionary of strategy_name -> portfolio_values
        window: Rolling window in days
        title: Plot title
        figsize: Figure size
        risk_free_rate: Annual risk-free rate
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, values in results.items():
        returns = values.pct_change()

        # Calculate rolling Sharpe
        rolling_mean = returns.rolling(window=window).mean() * 252
        rolling_std = returns.rolling(window=window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_std

        ax.plot(rolling_sharpe.index, rolling_sharpe.values, label=name, linewidth=2)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    returns: pd.DataFrame,
    title: str = "Correlation Matrix",
    figsize: tuple = (8, 6)
):
    """
    Plot correlation matrix of asset returns.

    Args:
        returns: DataFrame of returns with assets as columns
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate correlation matrix
    corr = returns.corr()

    # Create heatmap
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Correlation'},
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    metrics_to_plot: Optional[List[str]] = None,
    figsize: tuple = (14, 10)
):
    """
    Plot bar charts comparing metrics across strategies.

    Args:
        metrics_df: DataFrame with strategies as rows, metrics as columns
        metrics_to_plot: List of metrics to plot (None = all)
        figsize: Figure size
    """
    if metrics_to_plot is None:
        metrics_to_plot = metrics_df.columns.tolist()

    n_metrics = len(metrics_to_plot)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]

    for i, metric in enumerate(metrics_to_plot):
        if i >= len(axes):
            break

        ax = axes[i]
        values = metrics_df[metric]

        # Create bar plot
        values.plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)

        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.set_xlabel('')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for j, v in enumerate(values):
            if not np.isnan(v) and not np.isinf(v):
                ax.text(j, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig


def plot_returns_distribution(
    results: Dict[str, pd.Series],
    title: str = "Returns Distribution",
    figsize: tuple = (14, 6)
):
    """
    Plot distribution of returns for multiple strategies.

    Args:
        results: Dictionary of strategy_name -> portfolio_values
        title: Plot title
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Collect all returns
    all_returns = {}
    for name, values in results.items():
        returns = values.pct_change().dropna() * 100  # Convert to percentage
        all_returns[name] = returns

    # Plot histograms
    ax = axes[0]
    for name, returns in all_returns.items():
        ax.hist(returns, bins=50, alpha=0.5, label=name)

    ax.set_xlabel('Daily Return (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Return Distribution', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot box plots
    ax = axes[1]
    returns_list = [returns.values for returns in all_returns.values()]
    labels = list(all_returns.keys())

    bp = ax.boxplot(returns_list, labels=labels, patch_artist=True)

    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    ax.set_ylabel('Daily Return (%)', fontsize=12)
    ax.set_title('Return Box Plot', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def create_performance_dashboard(
    results: Dict[str, pd.Series],
    allocations: Dict[str, pd.DataFrame],
    metrics_df: pd.DataFrame,
    figsize: tuple = (16, 12)
):
    """
    Create comprehensive dashboard with multiple charts.

    Args:
        results: Dictionary of strategy_name -> portfolio_values
        allocations: Dictionary of strategy_name -> allocations
        metrics_df: DataFrame of metrics
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Equity curves
    ax1 = fig.add_subplot(gs[0, :])
    for name, values in results.items():
        normalized = (values / values.iloc[0] - 1) * 100
        ax1.plot(normalized.index, normalized.values, label=name, linewidth=2)
    ax1.set_ylabel('Return (%)', fontsize=11)
    ax1.set_title('Equity Curves', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Drawdowns
    ax2 = fig.add_subplot(gs[1, :])
    for name, values in results.items():
        running_max = values.cummax()
        drawdown = (values - running_max) / running_max * 100
        ax2.plot(drawdown.index, drawdown.values, label=name, linewidth=2)
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # Sharpe ratio comparison
    ax3 = fig.add_subplot(gs[2, 0])
    if 'Sharpe Ratio' in metrics_df.columns:
        metrics_df['Sharpe Ratio'].plot(kind='bar', ax=ax3, color='steelblue', alpha=0.7)
        ax3.set_title('Sharpe Ratio', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Sharpe Ratio', fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.tick_params(axis='x', rotation=45)

    # Max drawdown comparison
    ax4 = fig.add_subplot(gs[2, 1])
    if 'Max Drawdown' in metrics_df.columns:
        (metrics_df['Max Drawdown'] * 100).plot(kind='bar', ax=ax4, color='crimson', alpha=0.7)
        ax4.set_title('Max Drawdown', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Max Drawdown (%)', fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.tick_params(axis='x', rotation=45)

    plt.suptitle('Portfolio Performance Dashboard', fontsize=16, fontweight='bold', y=0.995)

    return fig


if __name__ == "__main__":
    # Example usage
    print("Visualization module loaded successfully")
    print("Available functions:")
    print("  - plot_equity_curves()")
    print("  - plot_drawdown()")
    print("  - plot_allocation_over_time()")
    print("  - plot_correlation_matrix()")
    print("  - create_performance_dashboard()")
