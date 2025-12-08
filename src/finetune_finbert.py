"""
Fine-tune FinBERT on ETF-specific financial news data.

Creates pseudo-labeled dataset from historical news + ETF returns,
then fine-tunes FinBERT for better sentiment predictions.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from tqdm import tqdm

# Transformers imports
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Local imports
from sentiment import GDELTBigQueryLoader, HeadlineScraper, ETF_SEARCH_TERMS, FINANCE_DOMAINS
import yfinance as yf


@dataclass
class FineTuneConfig:
    """Configuration for fine-tuning."""
    # Data parameters
    start_date: str = "2020-01-01"  # Start date for training data
    end_date: str = "2024-12-31"    # End date for training data
    max_headlines_per_etf: int = 5000  # Max headlines per ETF
    min_return_threshold: float = 0.01  # 1% - threshold for strong signal

    # Model parameters
    model_name: str = "ProsusAI/finbert"
    max_length: int = 128

    # Training parameters
    output_dir: str = "models/finbert-etf-finetuned"
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    eval_steps: int = 500
    save_steps: int = 500

    # Hardware
    device: str = None  # Auto-detect

    def __post_init__(self):
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


class ETFSentimentDatasetBuilder:
    """
    Build labeled sentiment dataset from historical news + ETF returns.

    Strategy: Fetch headlines, compute forward returns, assign labels based on returns.
    - Positive label: ETF went up significantly after news
    - Negative label: ETF went down significantly after news
    - Neutral label: Small price movement
    """

    def __init__(self, config: FineTuneConfig):
        self.config = config
        self.bq_loader = GDELTBigQueryLoader()
        self.scraper = HeadlineScraper(max_workers=20)

    def _fetch_etf_prices(self, ticker: str) -> pd.Series:
        """Download ETF prices for labeling."""
        print(f"  Downloading {ticker} prices...")
        data = yf.download(
            ticker,
            start=self.config.start_date,
            end=self.config.end_date,
            progress=False
        )
        return data['Close']

    def _compute_forward_returns(
        self,
        prices: pd.Series,
        dates: pd.DatetimeIndex,
        horizon: int = 5  # 5-day forward return
    ) -> pd.Series:
        """
        Compute forward returns for sentiment labeling.

        Args:
            prices: Price series
            dates: Dates for which to compute returns
            horizon: Number of days forward to look

        Returns:
            Series of forward returns indexed by date
        """
        forward_returns = []

        for date in dates:
            try:
                # Find price at date
                if date not in prices.index:
                    # Use nearest prior date
                    prior_dates = prices.index[prices.index <= date]
                    if len(prior_dates) == 0:
                        forward_returns.append(np.nan)
                        continue
                    date = prior_dates[-1]

                current_price = prices[date]

                # Find price horizon days later
                future_dates = prices.index[prices.index > date]
                if len(future_dates) < horizon:
                    forward_returns.append(np.nan)
                    continue

                future_date = future_dates[min(horizon-1, len(future_dates)-1)]
                future_price = prices[future_date]

                # Compute return
                ret = (future_price - current_price) / current_price
                forward_returns.append(ret)

            except Exception:
                forward_returns.append(np.nan)

        return pd.Series(forward_returns, index=dates)

    def _assign_labels(self, returns: pd.Series) -> pd.Series:
        """
        Assign sentiment labels based on forward returns.

        Labels:
        - 2 (positive): return > threshold
        - 1 (neutral): -threshold <= return <= threshold
        - 0 (negative): return < -threshold
        """
        threshold = self.config.min_return_threshold

        labels = pd.Series(index=returns.index, dtype=int)
        labels[returns > threshold] = 2  # Positive
        labels[returns < -threshold] = 0  # Negative
        labels[(returns >= -threshold) & (returns <= threshold)] = 1  # Neutral

        return labels

    def build_dataset_for_ticker(self, ticker: str) -> pd.DataFrame:
        """
        Build labeled dataset for a single ETF ticker.

        Returns:
            DataFrame with columns: headline, label, ticker, date, return
        """
        print(f"\n{'='*80}")
        print(f"Building dataset for {ticker}")
        print(f"{'='*80}")

        # Fetch news from GDELT
        search_terms = ETF_SEARCH_TERMS.get(ticker, [ticker])

        print(f"  Querying GDELT for {ticker} ({', '.join(search_terms)})...")
        news_data = self.bq_loader.query_events(
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            search_terms=search_terms,
            domains=FINANCE_DOMAINS,
            max_results=self.config.max_headlines_per_etf * 2  # Get extra, will filter
        )

        if news_data.empty:
            print(f"  No news found for {ticker}")
            return pd.DataFrame(columns=['headline', 'label', 'ticker', 'date', 'return'])

        print(f"  Found {len(news_data)} articles")

        # Fetch headlines
        print(f"  Fetching headlines...")
        urls = news_data['url'].tolist()
        headlines = self.scraper.fetch_headlines_batch(
            urls,
            progress_callback=lambda done, total: print(f"    {done}/{total} headlines fetched", end='\r')
        )
        print(f"\n  Got {len(headlines)} headlines")

        # Map headlines back to dates
        news_data['headline'] = news_data['url'].map(headlines)
        news_data = news_data.dropna(subset=['headline'])

        if news_data.empty:
            print(f"  No valid headlines for {ticker}")
            return pd.DataFrame(columns=['headline', 'label', 'ticker', 'date', 'return'])

        # Download ETF prices
        prices = self._fetch_etf_prices(ticker)

        # Compute forward returns for labeling
        print(f"  Computing forward returns...")

        # Compute returns row by row to handle duplicate dates
        returns_list = []
        for idx, row in news_data.iterrows():
            date = row['date']
            try:
                # Find price at date
                if date not in prices.index:
                    prior_dates = prices.index[prices.index <= date]
                    if len(prior_dates) == 0:
                        returns_list.append(np.nan)
                        continue
                    date = prior_dates[-1]

                current_price = prices[date]

                # Find price 5 days later
                future_dates = prices.index[prices.index > date]
                if len(future_dates) < 5:
                    returns_list.append(np.nan)
                    continue

                future_date = future_dates[min(4, len(future_dates)-1)]
                future_price = prices[future_date]

                # Compute return
                ret = (future_price - current_price) / current_price
                returns_list.append(ret)

            except Exception:
                returns_list.append(np.nan)

        news_data['return'] = returns_list

        # Remove NaN returns
        news_data = news_data.dropna(subset=['return'])

        if news_data.empty:
            print(f"  No valid returns for {ticker}")
            return pd.DataFrame(columns=['headline', 'label', 'ticker', 'date', 'return'])

        # Assign labels
        news_data['label'] = self._assign_labels(news_data['return'])
        news_data['ticker'] = ticker

        # Sample to limit size
        if len(news_data) > self.config.max_headlines_per_etf:
            # Stratified sample to keep balanced labels
            news_data = news_data.groupby('label', group_keys=False).apply(
                lambda x: x.sample(min(len(x), self.config.max_headlines_per_etf // 3))
            )

        print(f"  Final dataset size: {len(news_data)}")
        print(f"  Label distribution:")
        print(f"    Negative (0): {(news_data['label'] == 0).sum()}")
        print(f"    Neutral (1):  {(news_data['label'] == 1).sum()}")
        print(f"    Positive (2): {(news_data['label'] == 2).sum()}")

        return news_data[['headline', 'label', 'ticker', 'date', 'return']]

    def build_full_dataset(self, tickers: List[str]) -> pd.DataFrame:
        """Build labeled dataset for all tickers."""
        print(f"\nBuilding labeled dataset for {len(tickers)} ETFs...")

        all_data = []
        for ticker in tickers:
            ticker_data = self.build_dataset_for_ticker(ticker)
            if not ticker_data.empty:
                all_data.append(ticker_data)

        if not all_data:
            raise ValueError("No data collected for any ticker!")

        full_dataset = pd.concat(all_data, ignore_index=True)

        print(f"\n{'='*80}")
        print(f"Full Dataset Summary")
        print(f"{'='*80}")
        print(f"Total headlines: {len(full_dataset)}")
        print(f"Label distribution:")
        print(f"  Negative (0): {(full_dataset['label'] == 0).sum()} ({(full_dataset['label'] == 0).sum() / len(full_dataset) * 100:.1f}%)")
        print(f"  Neutral (1):  {(full_dataset['label'] == 1).sum()} ({(full_dataset['label'] == 1).sum() / len(full_dataset) * 100:.1f}%)")
        print(f"  Positive (2): {(full_dataset['label'] == 2).sum()} ({(full_dataset['label'] == 2).sum() / len(full_dataset) * 100:.1f}%)")
        print(f"Date range: {full_dataset['date'].min()} to {full_dataset['date'].max()}")

        return full_dataset


class FinBERTFineTuner:
    """Fine-tune FinBERT on labeled sentiment data."""

    def __init__(self, config: FineTuneConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = None

    def prepare_dataset(self, df: pd.DataFrame) -> Tuple[Dataset, Dataset]:
        """
        Prepare Hugging Face datasets for training.

        Args:
            df: DataFrame with 'headline' and 'label' columns

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        print("\nPreparing datasets for training...")

        # Split train/eval
        train_df, eval_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df['label']  # Maintain label distribution
        )

        print(f"Train size: {len(train_df)}")
        print(f"Eval size: {len(eval_df)}")

        # Convert to Hugging Face Dataset format
        train_dataset = Dataset.from_pandas(train_df[['headline', 'label']])
        eval_dataset = Dataset.from_pandas(eval_df[['headline', 'label']])

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples['headline'],
                padding='max_length',
                truncation=True,
                max_length=self.config.max_length
            )

        print("Tokenizing...")
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        eval_dataset = eval_dataset.map(tokenize_function, batched=True)

        # Set format for PyTorch
        train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

        return train_dataset, eval_dataset

    def train(self, train_dataset: Dataset, eval_dataset: Dataset):
        """
        Fine-tune FinBERT on the labeled dataset.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
        """
        print(f"\nInitializing model from {self.config.model_name}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=3,  # negative, neutral, positive
            id2label={0: "negative", 1: "neutral", 2: "positive"},
            label2id={"negative": 0, "neutral": 1, "positive": 2}
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=100,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),  # Use mixed precision on GPU
            report_to="none",  # Disable wandb/tensorboard
        )

        # Metric computation
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = predictions.argmax(axis=1)

            accuracy = (predictions == labels).mean()

            # Per-class accuracy
            metrics = {"accuracy": accuracy}
            for i, label_name in enumerate(["negative", "neutral", "positive"]):
                mask = labels == i
                if mask.sum() > 0:
                    metrics[f"accuracy_{label_name}"] = (predictions[mask] == labels[mask]).mean()

            return metrics

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        print(f"\nStarting training on {self.config.device}...")
        print(f"{'='*80}")

        # Train
        trainer.train()

        print(f"\n{'='*80}")
        print("Training complete!")
        print(f"{'='*80}")

        # Evaluate
        print("\nEvaluating on validation set...")
        metrics = trainer.evaluate()

        print("\nFinal metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        # Save final model
        print(f"\nSaving model to {self.config.output_dir}...")
        trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)

        print("Model saved!")

        return trainer


def main():
    """Main fine-tuning pipeline."""
    # Configuration
    config = FineTuneConfig(
        start_date="2020-01-01",  # 5 years of data
        end_date="2024-12-31",
        max_headlines_per_etf=3000,  # 3k headlines per ETF
        num_epochs=3,
        batch_size=16,
    )

    print(f"{'='*80}")
    print("FinBERT Fine-Tuning for ETF Sentiment Analysis")
    print(f"{'='*80}")
    print(f"Date range: {config.start_date} to {config.end_date}")
    print(f"Device: {config.device}")
    print(f"Output: {config.output_dir}")

    # ETFs to train on
    tickers = ['SPY', 'QQQ', 'TLT', 'GLD', 'VTI', 'BND']

    # Step 1: Build labeled dataset
    print("\n" + "="*80)
    print("STEP 1: Building Labeled Dataset")
    print("="*80)

    builder = ETFSentimentDatasetBuilder(config)

    # Check if we have a cached dataset
    cache_file = Path("data/finbert_training_data.csv")

    if cache_file.exists():
        print(f"\nFound cached dataset at {cache_file}")
        response = input("Use cached dataset? (y/n): ")
        if response.lower() == 'y':
            labeled_data = pd.read_csv(cache_file, parse_dates=['date'])
            print(f"Loaded {len(labeled_data)} labeled examples from cache")
        else:
            labeled_data = builder.build_full_dataset(tickers)
            labeled_data.to_csv(cache_file, index=False)
            print(f"Saved dataset to {cache_file}")
    else:
        labeled_data = builder.build_full_dataset(tickers)
        labeled_data.to_csv(cache_file, index=False)
        print(f"Saved dataset to {cache_file}")

    # Step 2: Fine-tune model
    print("\n" + "="*80)
    print("STEP 2: Fine-Tuning FinBERT")
    print("="*80)

    fine_tuner = FinBERTFineTuner(config)
    train_dataset, eval_dataset = fine_tuner.prepare_dataset(labeled_data)
    trainer = fine_tuner.train(train_dataset, eval_dataset)

    print("\n" + "="*80)
    print("Fine-tuning complete!")
    print(f"Model saved to: {config.output_dir}")
    print(f"To use in sentiment pipeline, update FinBertSentiment to load from:")
    print(f"  {config.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
