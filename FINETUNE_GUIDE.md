# FinBERT Fine-Tuning Guide

This guide walks you through fine-tuning FinBERT on ETF-specific financial news for better sentiment predictions.

## Quick Start

### Step 1: Install Additional Dependencies

```bash
pip install datasets>=2.14.0 accelerate>=0.20.0
```

(These should already be in `requirements.txt`)

### Step 2: Run Fine-Tuning Script

```bash
cd /Users/vincent/Programming_Projects/ETF-Optimization
python src/finetune_finbert.py
```

**What it does:**
1. **Fetches historical news** from GDELT for 6 core ETFs (2020-2024)
2. **Downloads ETF prices** to compute forward returns
3. **Creates labels** based on price movements:
   - Positive (2): ETF went up >1% in next 5 days
   - Neutral (1): ETF moved between -1% and +1%
   - Negative (0): ETF went down <-1%
4. **Fine-tunes FinBERT** on ~18K labeled headlines
5. **Saves model** to `models/finbert-etf-finetuned/`

**Expected runtime:** 1-2 hours (depends on GPU)

### Step 3: Use Fine-Tuned Model

After fine-tuning, use the model in your sentiment pipeline:

```python
from sentiment import compute_sentiment_bigquery

# Compute sentiment using fine-tuned model
sentiment_df = compute_sentiment_bigquery(
    start_date='2015-01-01',
    end_date='2025-12-07',
    use_gkg_tone=False,  # Use FinBERT, not GKG tone
    finbert_model_path='models/finbert-etf-finetuned'  # Your fine-tuned model
)
```

## Configuration Options

Edit `src/finetune_finbert.py` to customize:

```python
config = FineTuneConfig(
    start_date="2020-01-01",         # Training data start
    end_date="2024-12-31",           # Training data end
    max_headlines_per_etf=3000,      # Headlines per ETF
    min_return_threshold=0.01,       # 1% threshold for labels

    num_epochs=3,                    # Training epochs
    batch_size=16,                   # Batch size (reduce if OOM)
    learning_rate=2e-5,              # Learning rate
)
```

## ETF Coverage

**Default ETFs** (fine-tuning dataset):
- SPY, QQQ, TLT, GLD, VTI, BND

**To add more ETFs**, edit line 461 in `finetune_finbert.py`:
```python
tickers = ['SPY', 'QQQ', 'TLT', 'GLD', 'VTI', 'BND', 'VEA', 'VWO', 'IWM', 'XLE']
```

## Troubleshooting

### Out of Memory (OOM) Errors
```python
config.batch_size = 8  # Reduce from 16
```

### GDELT Query Fails
Check BigQuery authentication:
```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=gen-lang-client-0939375020
```

### No Headlines Fetched
- Normal for older dates (many URLs are dead)
- Script will skip ETFs with insufficient data
- Try reducing `start_date` to more recent dates

### Training Takes Too Long
```python
config.max_headlines_per_etf = 1000  # Reduce from 3000
config.num_epochs = 2                # Reduce from 3
```

## Dataset Caching

The script caches the labeled dataset at:
```
data/finbert_training_data.csv
```

To rebuild dataset from scratch:
```bash
rm data/finbert_training_data.csv
python src/finetune_finbert.py
```

## Validation

After fine-tuning, test the model:

```python
from sentiment import FinBertSentiment

# Load fine-tuned model
finbert = FinBertSentiment(model_path='models/finbert-etf-finetuned')

# Test examples
test_texts = [
    "Stock market rallies to new highs on strong earnings",
    "Market crashes amid recession fears",
    "Federal Reserve holds interest rates steady",
]

for text in test_texts:
    score = finbert.score(text)
    print(f"Score: {score:+.3f} | {text}")
```

Expected improved scores:
- Rally news → positive score (>0.5)
- Crash news → negative score (<-0.5)
- Neutral news → near-zero score

## Timeline

For **Dec 8 deadline**:

**Option A - Fast track (4 hours total):**
1. Fine-tune on 6 ETFs, 2020-2024 (1-2 hours)
2. Generate sentiment for all 10 ETFs (1 hour)
3. Run experiments notebook (1 hour)

**Option B - Baseline first (recommended):**
1. Run experiments WITHOUT sentiment (1 hour)
2. Fine-tune overnight (1-2 hours)
3. Add sentiment results tomorrow (2 hours)

## Next Steps

After fine-tuning:

1. **Compare models** in experiments notebook:
   - Pre-trained FinBERT
   - Fine-tuned FinBERT
   - GDELT GKG tone

2. **Ablation study**:
   - Ridge vs LightGBM (no sentiment)
   - Ridge vs LightGBM (with sentiment)

3. **Feature importance**:
   - SHAP values for sentiment features
   - Correlation between sentiment and returns
