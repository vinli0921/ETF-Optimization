"""
Quick verification script to test the baseline system setup.

Run this to verify all modules are working correctly before using the system.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ETF Portfolio Optimization - Setup Verification")
print("="*80)

# Test imports
print("\n1. Testing imports...")
try:
    from src.data import ETFDataLoader, load_default_etfs
    from src.features import FeatureEngineer
    from src.strategies import EqualWeightStrategy, MeanVarianceStrategy
    from src.backtest import PortfolioBacktest
    from src.metrics import calculate_all_metrics
    from src.visualization import plot_equity_curves
    print("   ✓ All modules imported successfully")
except Exception as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test data loading
print("\n2. Testing data loading...")
try:
    loader = ETFDataLoader()
    print("   ✓ ETFDataLoader initialized")
except Exception as e:
    print(f"   ✗ Data loader error: {e}")
    sys.exit(1)

# Test strategy initialization
print("\n3. Testing strategy initialization...")
try:
    ew = EqualWeightStrategy()
    mv = MeanVarianceStrategy()
    print("   ✓ Strategies initialized")
except Exception as e:
    print(f"   ✗ Strategy error: {e}")
    sys.exit(1)

# Test feature engineer
print("\n4. Testing feature engineer...")
try:
    fe = FeatureEngineer(lookback_window=30)
    print("   ✓ Feature engineer initialized")
except Exception as e:
    print(f"   ✗ Feature engineer error: {e}")
    sys.exit(1)

# Test backtest engine
print("\n5. Testing backtest engine...")
try:
    backtest = PortfolioBacktest()
    print("   ✓ Backtest engine initialized")
except Exception as e:
    print(f"   ✗ Backtest error: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("✓ All tests passed! System is ready to use.")
print("="*80)
print("\nNext steps:")
print("  1. Install dependencies: pip install -r requirements.txt")
print("  2. Run demo notebook: jupyter notebook notebooks/baseline_demo.ipynb")
print("  3. Or test data download: python src/data.py")
print("\nNote: Data will be downloaded on first use (requires internet connection)")
