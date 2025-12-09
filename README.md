# ETF Portfolio Optimization


![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)


A sophisticated portfolio optimization framework combining classical financial theory (Mean-Variance) with modern machine learning techniques (Ridge, LightGBM, XGBoost, LSTM) and sentiment analysis.


## Overview


This project dynamically allocates capital across 10 diversified ETFs to maximize risk-adjusted returns. It implements 7 portfolio allocation strategies—from simple equal-weight baselines to advanced ML and deep learning models—with rigorous backtesting that guarantees no look-ahead bias.


**Key differentiators:**
- **ML Integration**: Use machine learning to predict returns and optimize allocations
- **Sentiment Analysis**: Incorporate news sentiment from FinBERT + GDELT (optional)
- **Strict No-Look-Ahead**: All features use only past data; realistic transaction costs
- **Comprehensive Comparison**: Baseline, optimization, and ML strategies on the same footing

**→ [See comprehensive demo notebooks](#notebooks)**


## Key Features


- ✨ **7 Portfolio Strategies**: Equal Weight → Mean-Variance → ML (Ridge/LightGBM/XGBoost) → LSTM
- 📊 **10 ETFs**: US/International stocks, bonds, commodities (SPY, QQQ, VTI, IWM, VEA, VWO, XLE, TLT, BND, GLD)
- 🔬 **160+ Features**: Technical indicators, volume, correlations, market regime, sentiment
- 📈 **Rigorous Backtesting**: No-look-ahead bias, realistic transaction costs, temporal train/val/test splits
- 🎯 **Performance Metrics**: Sharpe, Sortino, Max Drawdown, Calmar, Win Rate
- 📰 **Sentiment Analysis**: FinBERT on GDELT news data (optional advanced feature)
- 🎨 **Rich Visualization**: Equity curves, drawdowns, allocations, correlation matrices
- 🔧 **Extensible**: Easy to add custom strategies, features, or ETFs


## Quick Start


```python
# Install dependencies
pip install -r requirements.txt


# Load data
from data import load_default_etfs
ohlcv_data, indicators = load_default_etfs(expanded=True)


# Extract close prices
close_cols = [col for col in ohlcv_data.columns if col.endswith('_Close')]
prices = ohlcv_data[close_cols].copy()
prices.columns = [col.replace('_Close', '') for col in close_cols]


# Run a strategy
from strategies import MeanVarianceStrategy
from backtest import PortfolioBacktest


strategy = MeanVarianceStrategy(lookback_days=252)
backtest = PortfolioBacktest(initial_capital=100000, transaction_cost=0.001)
portfolio_values = backtest.run(strategy, prices)


# Visualize
from visualization import plot_equity_curves
import matplotlib.pyplot as plt
plot_equity_curves({'Mean-Variance': portfolio_values})
plt.show()
```



## Repository Structure


```
ETF-Optimization/
├── src/                    # Core library
│   ├── data.py            # ETF data loading, caching, splits
│   ├── features.py        # Technical indicators, feature engineering (160+ features)
│   ├── strategies.py      # 9 portfolio allocation strategies
│   ├── backtest.py        # Backtesting engine with transaction costs
│   ├── metrics.py         # Performance metrics (Sharpe, drawdown, etc.)
│   ├── visualization.py   # Plotting utilities
│   ├── sentiment.py       # News sentiment analysis (GDELT + FinBERT)
│   ├── finetune_finbert.py # FinBERT fine-tuning pipeline
│   └── lstm_model.py      # LSTM neural network for predictions
├── notebooks/             # Interactive demos
│   └── baseline_demo.ipynb  # Complete workflow (data → strategies → results)
├── data/                  # Auto-generated cache (OHLCV, indicators, sentiment)
├── requirements.txt       # Dependencies
├── FINETUNE_GUIDE.md     # FinBERT fine-tuning guide
└── README.md             # This file
```


## ETF Coverage


**10 ETFs Across Asset Classes:**


| Ticker | Name | Asset Class | Purpose |
|--------|------|-------------|---------|
| **SPY** | S&P 500 | US Large-Cap Stocks | US equity core |
| **QQQ** | NASDAQ-100 | US Tech Stocks | Growth exposure |
| **VTI** | Vanguard Total Market | US Broad Stocks | Diversified US equity |
| **IWM** | Russell 2000 | US Small-Cap | Size factor exposure |
| **VEA** | FTSE Developed Markets | International Developed | Non-US diversification |
| **VWO** | FTSE Emerging Markets | Emerging Markets | EM exposure |
| **XLE** | Energy Sector SPDR | US Energy Stocks | Sector/commodity proxy |
| **TLT** | 20+ Year Treasury | Long-Term Bonds | Duration, safe haven |
| **BND** | Total Bond Market | Broad Bonds | Fixed income core |
| **GLD** | Gold Trust | Commodities | Inflation hedge, crisis alpha |


**Coverage Period:** 2015-2025 (10 years)


## Strategy Comparison


**7 Strategies from Baseline to Advanced ML:**


| Strategy | Type | Key Features | When to Use |
|----------|------|--------------|-------------|
| **Equal Weight** | Baseline | 1/N allocation | Benchmark, diversification baseline |
| **Mean-Variance** | Optimization | Max Sharpe, Ledoit-Wolf covariance | Strong historical data, stable markets |
| **60/40 Portfolio** | Static | 60% stocks, 40% bonds/alternatives | Passive benchmark, retirement accounts |
| **Predictive Sharpe** | ML (Linear) | Ridge regression + momentum features | Linear relationships, interpretability |
| **LightGBM ML** | ML (Tree) | Gradient boosting, 6 basic features/ticker | Non-linear patterns, moderate data |
| **XGBoost ML** | ML (Tree) | 160+ features, conservative hyperparameters | High-dimensional features, small samples |
| **LSTM** | Deep Learning | Sequential pattern learning | Temporal dependencies, sufficient data |


**Typical Performance (Test Set 2023-2025):**
- **Mean-Variance**: Sharpe **2.13**, 29% annual return, 13% volatility, 8% max drawdown
- **XGBoost ML**: Sharpe **~2.0**, ~27% annual return, ~13% volatility
- **Equal Weight**: Sharpe **1.50**, 17% annual return, 10% volatility (solid baseline)



## Installation


```bash
# Clone repository
git clone https://github.com/vinli0921/ETF-Optimization.git
cd ETF-Optimization


# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


# Install dependencies
pip install -r requirements.txt


# Optional: For sentiment analysis (requires Google Cloud credentials)
pip install google-cloud-bigquery google-cloud-bigquery-storage
```


**System Requirements:**
- Python 3.8+
- 4GB+ RAM (for ML strategies with 160+ features)


## Usage Examples


### Example 1: Compare Multiple Strategies


```python
from data import load_default_etfs
from strategies import (
   EqualWeightStrategy,
   MeanVarianceStrategy,
   XGBoostSharpeStrategy
)
from backtest import compare_strategies
from metrics import compare_strategies as compare_metrics


# Load data
ohlcv_data, indicators = load_default_etfs(expanded=True)
close_cols = [col for col in ohlcv_data.columns if col.endswith('_Close')]
prices = ohlcv_data[close_cols].copy()
prices.columns = [col.replace('_Close', '') for col in close_cols]


# Define strategies
strategies = {
   'Equal Weight': EqualWeightStrategy(),
   'Mean-Variance': MeanVarianceStrategy(lookback_days=252),
   'XGBoost ML': XGBoostSharpeStrategy(
       lookback_days=756,
       feature_window=60,
       n_estimators=100,
       max_depth=3
   )
}


# Run backtest comparison
results, allocations = compare_strategies(
   strategies,
   prices,
   initial_capital=100000,
   transaction_cost=0.001,  # 0.1% per trade
   rebalance_frequency='M'   # Monthly
)


# Calculate and display metrics
metrics = compare_metrics(results, allocations)
print(metrics)


# Visualize
from visualization import plot_equity_curves
plot_equity_curves(results, title='Strategy Comparison')
```


### Example 2: Custom Feature Engineering


```python
from features import FeatureEngineer


# Initialize feature engineer
fe = FeatureEngineer(lookback_window=60)


# Compute comprehensive features
features = fe.compute_all_features(
   prices=prices,
   ohlcv_data=ohlcv_data,
   indicators=indicators,
   include_technical=True,   # RSI, MACD, Bollinger Bands, ATR
   include_volume=True,      # Volume ratio, momentum
   include_market=True,      # VIX, yield curve
   include_correlations=True # Cross-asset correlations
)


print(f"Generated {len(features.columns)} features")
# Output: Generated 160 features


# View feature categories
print(features.columns.tolist()[:10])
# ['SPY_return', 'SPY_volatility', 'SPY_momentum', 'SPY_sharpe', ...]
```


### Example 3: Train/Validation/Test Split


```python
from data import ETFDataLoader


loader = ETFDataLoader()


# Split data with temporal ordering
train, val, test = loader.split_train_val_test(prices)


print(f"Train: {train.index[0]} to {train.index[-1]} ({len(train)} days)")
print(f"Val:   {val.index[0]} to {val.index[-1]} ({len(val)} days)")
print(f"Test:  {test.index[0]} to {test.index[-1]} ({len(test)} days)")


# Output:
# Train: 2015-01-02 to 2021-12-31 (1763 days)
# Val:   2022-01-03 to 2022-12-30 (251 days)
# Test:  2023-01-04 to 2025-12-05 (734 days)


# Backtest on test set only
results_test, _ = compare_strategies(strategies, test, rebalance_frequency='M')
```


### Example 4: Sentiment-Enhanced Strategy (Advanced)


```python
from sentiment import compute_all_etf_sentiment
from strategies import GradientBoostingSharpeStrategy


# Fetch sentiment data from GDELT (requires Google Cloud credentials)
sentiment_df = compute_all_etf_sentiment(
   start_date='2020-01-01',
   end_date='2024-12-31',
   tickers=['SPY', 'QQQ', 'VTI', 'TLT', 'BND', 'GLD']
)


# Use sentiment-enhanced strategy
strategy = GradientBoostingSharpeStrategy(
   lookback_days=756,
   use_sentiment=True  # Enables sentiment features
)


# Strategy will automatically incorporate sentiment if available
backtest = PortfolioBacktest()
portfolio_values = backtest.run(strategy, prices)
```


## Data Pipeline


### Automatic Data Management


The framework handles data loading, caching, and preprocessing automatically:


- **Downloads OHLCV** (Open, High, Low, Close, Volume) from Yahoo Finance via `yfinance`
- **Fetches market indicators**: VIX (volatility), 10Y Treasury yield, 3M Treasury yield
- **Caches locally** for reproducibility (stored in `data/` directory)
- **Smart refresh**: Use `force_refresh=True` to update cached data


### Data Splits


Temporal train/validation/test splits ensure realistic evaluation:


- **Train**: 2015-2021 (7 years) — Strategy development and initial fitting
- **Validation**: 2022 (1 year) — Hyperparameter tuning and model selection
- **Test**: 2023-2025 (2+ years) — Final evaluation (held-out, never seen during development)


### No-Look-Ahead Guarantee


**Critical for realistic backtesting:**


- All features computed with **past-only windows** (e.g., 30-day rolling momentum uses days t-30 to t-1)
- Backtester feeds strategies **only historical data** up to (but NOT including) current rebalance date
- **Strict temporal ordering** maintained throughout pipeline
- Predictions made at time t use only data from t-1 and earlier


This prevents data leakage and ensures results reflect real-world performance.


## Feature Engineering


### 160+ Features Across 5 Categories


#### 1. Price-Based Features (6 per ticker = 60 total)
- Daily percentage returns
- Rolling volatility (annualized, 30-60 day window)
- Rolling momentum (annualized recent returns)
- Rolling Sharpe ratio (return/volatility)
- Average correlation with other assets
- Lagged returns (1, 5, 21 days)


#### 2. Technical Indicators (4 per ticker = 40 total)
- **RSI** (Relative Strength Index, 14-day)
- **MACD** (Moving Average Convergence Divergence)
- **Bollinger Bands** (20-day, 2-sigma upper/lower/width)
- **ATR** (Average True Range, 14-day volatility measure)


#### 3. Volume Features (2 per ticker = 20 total)
- Volume ratio (current vs. 30-day average)
- Volume momentum (recent vs. historical)


#### 4. Market Regime Features (6 total)
- VIX level, change, percentile (volatility regime)
- High volatility indicator (VIX > threshold)
- Yield curve spread (10Y - 3M)
- Yield curve inversion indicator


#### 5. Sentiment Features (3 per ticker = 30 total, optional)
- Raw sentiment score from FinBERT
- Sentiment moving average (smoothed)
- Sentiment momentum (recent change)


**Total**: 60 + 40 + 20 + 6 + 30 = **156 features** (plus correlations → 160+)


**All features are lagged/shifted by 1 day to prevent look-ahead bias.**


## Backtesting


### Realistic Simulation Features


- **Transaction costs**: Default 0.1% per trade (customizable)
- **Flexible rebalancing**: Daily (`'D'`), Weekly (`'W'`), Monthly (`'M'`), Quarterly (`'Q'`)
- **Position tracking**: Shares held, cash balance, portfolio value
- **Turnover calculation**: Measures trading activity (important for cost-sensitive strategies)
- **Progress bars**: Uses `tqdm` for long-running backtests


### Performance Metrics


Comprehensive risk-adjusted metrics computed automatically:


| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| **Total Return** | (Final - Initial) / Initial | Absolute performance over period |
| **Annualized Return (CAGR)** | Compound annual growth rate | Fair comparison across time periods |
| **Annualized Volatility** | Std dev of returns × √252 | Risk measure (higher = more volatile) |
| **Sharpe Ratio** | (Return - Risk-free) / Volatility | Risk-adjusted return (all volatility) |
| **Sortino Ratio** | (Return - Risk-free) / Downside Vol | Risk-adjusted return (downside only) |
| **Max Drawdown** | Largest peak-to-trough decline | Worst-case loss from peak |
| **Calmar Ratio** | Annual Return / Max Drawdown | Return vs. catastrophic risk |
| **Win Rate** | % of positive return periods | Consistency measure |


### Example Backtest Output


```python
Running backtest for Mean-Variance Optimization
 Period: 2023-01-04 to 2025-12-05
 Rebalancing: M (25 times)
 Final value: $209,855.16


Performance Metrics:
                     Sharpe  Return  Volatility  Max Drawdown
Mean-Variance          2.13   28.9%      12.6%          8.3%
```


## Advanced: Sentiment Analysis


### Optional Feature (Not working)


Integrate news sentiment into allocation decisions using FinBERT and GDELT.


#### Data Pipeline


1. **Data Source**: GDELT (Global Database of Events, Language, and Tone)
  - 100M+ news articles daily from global sources
  - Query via Google BigQuery


2. **Sentiment Model**: FinBERT
  - BERT fine-tuned on financial text
  - Outputs: positive, neutral, negative scores
  - Pre-trained on 10K+ financial news headlines


3. **Pipeline Steps**:
  ```
  Query GDELT → Fetch article URLs → Extract headlines →
  Score with FinBERT → Aggregate to daily sentiment per ETF
  ```


#### Fine-Tuning FinBERT


For domain-specific accuracy, fine-tune FinBERT on your own labeled data:


- See **`FINETUNE_GUIDE.md`** for detailed instructions
- Default labels based on **forward 5-day returns**:
 - Positive: ETF price up >1%
 - Neutral: ETF price between -1% and +1%
 - Negative: ETF price down <-1%
- Training set: 18K+ labeled headlines (2020-2024)


#### Enable Sentiment in Strategies


```python
from strategies import GradientBoostingSharpeStrategy


# Enable sentiment features
strategy = GradientBoostingSharpeStrategy(use_sentiment=True)


# Strategy automatically incorporates sentiment features if available
# Falls back gracefully if sentiment data not present
```


**Requirements:**
- Google Cloud project with BigQuery API enabled
- `google-cloud-bigquery` package installed
- Credentials configured (`GOOGLE_APPLICATION_CREDENTIALS` env var)


## Extending the Framework


### Add a Custom Strategy


Inherit from `BaseStrategy` and implement the `allocate()` method:


```python
from strategies import BaseStrategy
import pandas as pd


class MomentumStrategy(BaseStrategy):
   """Allocate to assets with positive recent momentum."""


   def __init__(self, lookback_days=20):
       super().__init__("Momentum Strategy")
       self.lookback_days = lookback_days


   def allocate(self, prices, current_date=None, **kwargs):
       """
       Return allocation weights as dict: {ticker: weight}


       Args:
           prices: Historical price data (DataFrame)
           current_date: Current rebalancing date (Timestamp)
           **kwargs: Additional data (ohlcv_data, indicators, etc.)


       Returns:
           dict: {ticker: weight} where weights sum to 1.0
       """
       # Calculate momentum (% change over lookback period)
       returns = prices.pct_change(self.lookback_days).iloc[-1]


       # Only allocate to positive momentum assets
       positive_tickers = returns[returns > 0].index


       if len(positive_tickers) == 0:
           # Equal weight if all negative
           return {t: 1/len(prices.columns) for t in prices.columns}


       # Equal weight among positive momentum assets
       weight = 1 / len(positive_tickers)
       return {
           t: weight if t in positive_tickers else 0.0
           for t in prices.columns
       }


# Use your custom strategy
strategy = MomentumStrategy(lookback_days=30)
backtest = PortfolioBacktest()
portfolio_values = backtest.run(strategy, prices)
```


### Add Custom Features


Extend the feature set for ML strategies:


```python
from features import FeatureEngineer
import pandas as pd


# Compute default features
fe = FeatureEngineer(lookback_window=60)
features = fe.compute_all_features(prices, ohlcv_data, indicators)


# Add your custom indicator
def compute_custom_indicator(prices, window=20):
   """Example: Simple momentum oscillator"""
   returns = prices.pct_change(window)
   # Your custom logic here
   return returns.rank(axis=1, pct=True)  # Relative rank across assets


custom_feature = compute_custom_indicator(prices, window=20)
for ticker in prices.columns:
   features[f'{ticker}_custom_indicator'] = custom_feature[ticker]


# Use enhanced features in ML strategy
from strategies import XGBoostSharpeStrategy
strategy = XGBoostSharpeStrategy()
# Strategy will use all features in `features` DataFrame
```


## Notebooks


### Main Experiments


**`baseline_demo.ipynb`** — Run all main experiments with complete end-to-end workflow:


1. Load 10 ETFs with OHLCV data (2015-2025)
2. Compute 160+ features (technical, volume, market, correlations)
3. Run 7 strategies (Equal Weight, Mean-Variance, Ridge, LightGBM, XGBoost, LSTM, 60/40)
4. Backtest on train/validation/test splits
5. Visualize equity curves, drawdowns, allocations
6. Compare performance metrics (Sharpe, Sortino, Max DD)


**Start here:**


```bash
jupyter notebook notebooks/baseline_demo.ipynb
```


Run all cells to see the complete pipeline in action (~10 minutes).


### Comprehensive ML Experiments


**`ml_experiments_baseline.ipynb`** — In-depth machine learning experiments and analysis:


- Detailed feature engineering and selection
- Hyperparameter tuning for ML models
- Extended performance analysis
- Advanced visualizations and model comparisons


```bash
jupyter notebook notebooks/ml_experiments_baseline.ipynb
```


## Performance Benchmarks


### Test Set Results (2023-2025, 10 ETFs)


Results from rigorous backtesting on held-out test data:


| Strategy | Sharpe | Annual Return | Volatility | Max Drawdown |
|----------|-------:|-------------:|-----------:|------------:|
| **Mean-Variance** | **2.13** | 28.9% | 12.6% | 8.3% |
| **XGBoost ML** | ~2.0 | ~27% | ~13% | ~8-9% |
| **LightGBM ML** | ~1.9 | ~24% | ~12% | ~9-10% |
| **Equal Weight** | 1.50 | 16.9% | 10.0% | 10.0% |
| **Predictive Sharpe** | 1.47 | 17.3% | 10.4% | 8.1% |
| **LSTM** | 1.24 | 15.6% | 11.0% | 12.9% |
| **60/40 Portfolio** | 1.10 | 14.1% | 11.1% | 12.7% |


### Key Takeaways


- **Mean-Variance wins** on test set with 2.13 Sharpe (classical optimization still strong!)
- **XGBoost competitive** with 160+ features (~2.0 Sharpe)
- **Equal Weight solid baseline** at 1.50 Sharpe (hard to beat consistently)
- **All strategies beat 60/40 benchmark** (traditional passive allocation)
- **Lower drawdowns** than you might expect (6-13% max) due to diversification


**Important:** These are backtested results. Past performance does not guarantee future results. Real trading involves slippage, market impact, and other frictions not captured in backtests.


## Dependencies


### Core Libraries


- **`yfinance`**: Download ETF price data from Yahoo Finance
- **`pandas`**, **`numpy`**, **`scipy`**: Data manipulation and numerical computing
- **`scikit-learn`**: Machine learning algorithms, covariance estimation
- **`PyPortfolioOpt`**: Mean-variance optimization, efficient frontier


### Machine Learning


- **`lightgbm`**: Gradient boosting (LightGBM implementation)
- **`xgboost`**: Gradient boosting (XGBoost implementation)
- **`torch`**: PyTorch for LSTM neural networks


### Visualization


- **`matplotlib`**, **`seaborn`**: Plotting and data visualization


### Optional (Sentiment Analysis)


- **`transformers`**: Hugging Face library for FinBERT
- **`google-cloud-bigquery`**: Query GDELT database
- **`google-cloud-bigquery-storage`**: Fast data transfer from BigQuery
- **`requests`**, **`beautifulsoup4`**: Fetch and parse news articles


### Development


- **`jupyter`**, **`notebook`**: Interactive analysis
- **`tqdm`**: Progress bars for long-running operations


See **`requirements.txt`** for complete list with version pins.

