"""
Generate sentiment scores using pre-trained FinBERT (no fine-tuning).

This runs Option A: Use the pre-trained ProsusAI/finbert model directly
to score news sentiment for all ETFs in the portfolio.
"""

import pandas as pd
from sentiment import compute_sentiment_bigquery
from datetime import datetime

def main():
    print("="*80)
    print("Generating Sentiment with Pre-Trained FinBERT")
    print("="*80)
    print("Model: ProsusAI/finbert (pre-trained)")
    print("Date range: 2015-01-01 to 2025-12-07")
    print("ETFs: SPY, QQQ, VTI, TLT, BND, GLD, VEA, VWO, IWM, XLE")
    print("Method: GDELT BigQuery + FinBERT scoring")
    print("\nThis will take 30-60 minutes depending on data volume...")
    print("="*80)

    # Generate sentiment using pre-trained FinBERT
    print("\nFetching news and computing sentiment...")
    sentiment_df = compute_sentiment_bigquery(
        start_date='2015-01-01',
        end_date='2025-12-07',
        use_gkg_tone=False,  # Use FinBERT, not GKG tone
        finbert_model_path=None  # Use pre-trained model (not fine-tuned)
    )

    # Ensure date is a column (compute_sentiment_bigquery returns DatetimeIndex)
    if sentiment_df.index.name != 'date':
        sentiment_df.index.name = 'date'
    sentiment_df = sentiment_df.reset_index()

    # Save results
    output_path = 'data/sentiment_pretrained.csv'
    sentiment_df.to_csv(output_path, index=False)

    print("\n" + "="*80)
    print("Sentiment generation complete!")
    print("="*80)
    print(f"Output saved to: {output_path}")
    print(f"Total rows: {len(sentiment_df)}")
    print(f"\nDate range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")
    print(f"\nColumns: {list(sentiment_df.columns)}")
    print("\nSample data:")
    print(sentiment_df.head(10))
    print("\nSummary statistics:")
    print(sentiment_df.describe())
    print("="*80)

    return sentiment_df

if __name__ == "__main__":
    main()
