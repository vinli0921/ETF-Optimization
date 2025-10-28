# ETF Portfolio Optimization

A modular portfolio optimization system using baseline strategies and reinforcement learning for dynamic ETF allocation. This project aims to maximize risk-adjusted returns (Sharpe Ratio) through data-driven portfolio rebalancing.

## Overview

This system provides:
- **Data Pipeline**: Automated ETF data downloading and caching via yfinance
- **Feature Engineering**: Rolling statistics, momentum, volatility, and correlation features
- **Baseline Strategies**: Equal weight, mean-variance optimization, static allocations
- **Backtesting Engine**: Realistic simulation with transaction costs and no-look-ahead bias
- **Performance Metrics**: Sharpe ratio, Sortino ratio, max drawdown, and more
- **Visualization**: Comprehensive charts for equity curves, allocations, and risk metrics
- **RL-Ready**: Foundation for reinforcement learning extensions

## Project Structure

```
ETF-Optimization/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── data.py                  # Data pipeline (download, cache, split)
│   ├── features.py              # Feature engineering (volatility, momentum, etc.)
│   ├── strategies.py            # Portfolio strategies (equal weight, mean-variance)
│   ├── backtest.py              # Backtesting engine with transaction costs
│   ├── metrics.py               # Performance metrics (Sharpe, drawdown, etc.)
│   └── visualization.py         # Plotting utilities
├── notebooks/
│   └── baseline_demo.ipynb      # Comprehensive demo notebook
├── data/                        # Cached data (auto-generated, gitignored)
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ETF-Optimization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation by running a module:
```bash
python src/data.py
```

## Quick Start

### Using the Jupyter Notebook (Recommended)

The easiest way to get started is with the demo notebook:

```bash
jupyter notebook notebooks/baseline_demo.ipynb
```

This notebook demonstrates:
- Loading ETF data
- Feature engineering
- Running baseline strategies
- Backtesting and performance evaluation
- Comprehensive visualizations

### Using Python Scripts

#### 1. Load Data

```python
from src.data import load_default_etfs, ETFDataLoader

# Load VTI, BND, QQQ (2015-present)
prices = load_default_etfs(start_date='2015-01-01')

# Split into train/val/test
loader = ETFDataLoader()
train, val, test = loader.split_train_val_test(prices)
```

#### 2. Define Strategies

```python
from src.strategies import EqualWeightStrategy, MeanVarianceStrategy

strategies = {
    'Equal Weight': EqualWeightStrategy(),
    'Mean-Variance': MeanVarianceStrategy(lookback_days=252)
}
```

#### 3. Run Backtest

```python
from src.backtest import compare_strategies

results, allocations = compare_strategies(
    strategies,
    train,
    initial_capital=100000,
    transaction_cost=0.001,  # 0.1%
    rebalance_frequency='M'  # Monthly
)
```

#### 4. Evaluate Performance

```python
from src.metrics import compare_strategies as compare_metrics
from src.visualization import plot_equity_curves, create_performance_dashboard

# Calculate metrics
metrics = compare_metrics(results, allocations)
print(metrics)

# Visualize
plot_equity_curves(results)
create_performance_dashboard(results, allocations, metrics)
```

## Features

### Data Pipeline (`src/data.py`)
- **Automated downloading**: Fetches ETF data from Yahoo Finance
- **Caching**: Saves data locally for reproducibility and speed
- **Train/val/test splitting**: Automatic temporal data splitting
- **Summary statistics**: Quick overview of price data

### Feature Engineering (`src/features.py`)
- Daily percentage returns
- Rolling volatility (annualized standard deviation)
- Rolling momentum (average recent returns)
- Rolling Sharpe ratio
- Rolling correlations between assets
- **No look-ahead bias**: All features use only past data

### Strategies (`src/strategies.py`)

#### Equal Weight
Simple 1/N allocation across all assets. Rebalanced periodically.

#### Mean-Variance Optimization
Uses PyPortfolioOpt to find maximum Sharpe ratio portfolio based on historical returns and covariances.

#### Static Allocation (60/40)
Traditional portfolio with fixed weights (e.g., 60% stocks, 40% bonds).

#### Custom Strategies
Easy to extend by subclassing `BaseStrategy`.

### Backtesting (`src/backtest.py`)
- **Realistic simulation**: Includes transaction costs, slippage
- **Flexible rebalancing**: Daily, weekly, monthly, or custom frequencies
- **No look-ahead**: Strict temporal ordering
- **Position tracking**: Monitors shares and cash over time
- **Turnover calculation**: Measures trading activity

### Metrics (`src/metrics.py`)
- **Total return**: Overall gain/loss
- **Annualized return**: CAGR
- **Annualized volatility**: Risk measure
- **Sharpe ratio**: Risk-adjusted return
- **Sortino ratio**: Downside risk-adjusted return
- **Max drawdown**: Worst peak-to-trough decline
- **Calmar ratio**: Return vs max drawdown
- **Win rate**: Percentage of positive periods

### Visualization (`src/visualization.py`)
- Equity curves comparison
- Drawdown plots
- Allocation over time (stacked area, heatmap)
- Correlation matrices
- Rolling Sharpe ratio
- Returns distribution
- Comprehensive performance dashboard

## Default ETFs

The system uses three ETFs by default:
- **VTI**: Vanguard Total Stock Market (broad US stocks)
- **BND**: Vanguard Total Bond Market (US bonds)
- **QQQ**: Invesco QQQ Trust (NASDAQ-100 tech stocks)

Easily extend to more ETFs by modifying the ticker list.

## Data Splits

Default splits:
- **Train**: 2015-2021 (7 years) - Strategy development
- **Val**: 2022 (1 year) - Hyperparameter tuning
- **Test**: 2023-2025 (2+ years) - Final evaluation

## Configuration

### Transaction Costs
Default: 0.1% per trade (0.001)
Adjust in `PortfolioBacktest(transaction_cost=0.001)`

### Rebalancing Frequency
- `'D'`: Daily
- `'W'`: Weekly
- `'M'`: Monthly (default)
- `'Q'`: Quarterly
- `'Y'`: Yearly

### Lookback Window
Default: 252 trading days (1 year)
Used for rolling statistics and mean-variance optimization.

## Examples

### Example 1: Compare Rebalancing Frequencies

```python
from src.strategies import MeanVarianceStrategy
from src.backtest import PortfolioBacktest

strategy = MeanVarianceStrategy()

for freq in ['M', 'Q']:
    backtest = PortfolioBacktest(rebalance_frequency=freq)
    results = backtest.run(strategy, train)
    print(f"{freq}: Final value = ${results.iloc[-1]:,.2f}")
```

### Example 2: Custom Static Portfolio

```python
from src.strategies import StaticStrategy

# 50% VTI, 30% QQQ, 20% BND
custom_weights = {'VTI': 0.5, 'QQQ': 0.3, 'BND': 0.2}
strategy = StaticStrategy(custom_weights)
```

### Example 3: Walk-Forward Validation

```python
# Test on multiple rolling windows
for year in range(2018, 2024):
    train_subset = train[f'{year-3}-01-01':f'{year-1}-12-31']
    test_subset = train[f'{year}-01-01':f'{year}-12-31']

    results, _ = compare_strategies(strategies, test_subset)
    print(f"{year}: {results}")
```

## Extending the System

### Adding New Strategies

```python
from src.strategies import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("My Custom Strategy")

    def allocate(self, prices, current_date=None, **kwargs):
        # Your allocation logic here
        tickers = prices.columns
        weights = {ticker: 1.0/len(tickers) for ticker in tickers}

        self.validate_weights(weights)
        return weights
```

### Adding New Metrics

```python
from src.metrics import calculate_returns

def calculate_my_metric(portfolio_values):
    returns = calculate_returns(portfolio_values)
    # Your metric calculation
    return metric_value
```

### Adding New Features

```python
from src.features import FeatureEngineer

class MyFeatureEngineer(FeatureEngineer):
    def compute_rsi(self, prices, window=14):
        # Compute RSI indicator
        pass
```

## Troubleshooting

### Data Download Issues
If yfinance fails to download:
```python
loader = ETFDataLoader()
prices = loader.download_etfs(['VTI', 'BND', 'QQQ'], '2015-01-01', '2025-01-01', force_refresh=True)
```

### Import Errors
Ensure you're running from project root and src is in path:
```python
import sys
sys.path.append('src')
```

### Optimization Failures
If mean-variance optimization fails (singular matrix, etc.), it falls back to equal weight. Check for:
- Insufficient data (need at least 30 days)
- Highly correlated assets
- Numerical issues

## Performance Tips

1. **Use caching**: Data is cached automatically, don't force refresh unless needed
2. **Start simple**: Begin with fewer ETFs and shorter periods
3. **Monthly rebalancing**: Good balance between adaptivity and transaction costs
4. **Lookback window**: 252 days (1 year) is a good default

## Future Enhancements

- [ ] Reinforcement learning agents (PPO, A2C, DDPG)
- [ ] Custom RL environment with Sharpe-aware rewards
- [ ] Risk parity and minimum variance strategies
- [ ] Regime detection (bull/bear/volatile markets)
- [ ] Macroeconomic features (yield curve, VIX)
- [ ] Multi-objective optimization
- [ ] Live trading interface
- [ ] Advanced risk models (CVaR, tail risk)

## References

- PyPortfolioOpt: https://pyportfolioopt.readthedocs.io/
- yfinance: https://github.com/ranaroussi/yfinance
- Modern Portfolio Theory: Markowitz (1952)

## License

MIT License

## Authors

ETF Portfolio Optimization Team

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Citation

If you use this code in your research, please cite:

```
@software{etf_optimization,
  title={ETF Portfolio Optimization},
  year={2025},
  url={https://github.com/yourusername/ETF-Optimization}
}
```