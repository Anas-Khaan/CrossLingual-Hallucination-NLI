"""
Data Preprocessing
==================
Convert translated CSV files to NLI format for training:
  - Premise  = evidence (source text)
  - Hypothesis = claim (generated text)
  - Label = 0 (Supported), 1 (Not Supported), 2 (Not Enough Info)

Creates PyTorch Datasets and DataLoaders with XLM-RoBERTa tokenization.
"""

import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer

from src.utils import FINAL_DIR, FRIENDLY_LABELS


# ============================================
# Constants
# ============================================

DEFAULT_MODEL_NAME = "xlm-roberta-large"
DEFAULT_MAX_LENGTH = 512


# ============================================
# PyTorch Dataset
# ============================================

class HallucinationDataset(Dataset):
    """
    PyTorch Dataset for hallucination detection (NLI-style).

    Each sample is a pair: (evidence, claim) → label.
    Tokenized using XLM-RoBERTa tokenizer as:
        [CLS] evidence [SEP] claim [SEP]
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        """
        Args:
            dataframe: DataFrame with columns [claim, evidence, label].
            tokenizer: Hugging Face tokenizer.
            max_length: Max token length for truncation.
        """
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]

        evidence = str(row["evidence"]) if pd.notna(row["evidence"]) else ""
        claim = str(row["claim"]) if pd.notna(row["claim"]) else ""
        label = int(row["label"])

        # Tokenize as NLI pair: [CLS] evidence [SEP] claim [SEP]
        encoding = self.tokenizer(
            evidence,
            claim,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ============================================
# Data Loading Functions
# ============================================

def load_variant_csv(split: str, variant: str) -> pd.DataFrame:
    """
    Load a specific variant CSV file.

    Args:
        split: "train" or "dev"
        variant: "pure_urdu", "mixed", or "roman_urdu"

    Returns:
        DataFrame with columns [claim, evidence, label, label_text]
    """
    path = FINAL_DIR / f"{split}_{variant}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. "
            "Please run 'python main.py translate' first."
        )

    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"  ✓ Loaded {path.name}: {len(df):,} rows")
    return df


def load_combined_dataset(
    split: str,
    variants: List[str] = None,
) -> pd.DataFrame:
    """
    Load and combine multiple language variants into a single DataFrame.

    Args:
        split: "train" or "dev"
        variants: List of variants to include.
                   Default: all three ["pure_urdu", "mixed", "roman_urdu"]

    Returns:
        Combined DataFrame with a 'variant' column.
    """
    if variants is None:
        variants = ["pure_urdu", "mixed", "roman_urdu"]

    dfs = []
    for variant in variants:
        df = load_variant_csv(split, variant)
        df["variant"] = variant
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    print(f"\n  📊 Combined {split} dataset: {len(combined):,} rows "
          f"({len(variants)} variants)")
    print(f"     Label distribution:")
    for label_id, label_name in FRIENDLY_LABELS.items():
        count = (combined["label"] == label_id).sum()
        print(f"       {label_name:20s}: {count:,}")

    return combined


# ============================================
# DataLoader Factory
# ============================================

def create_dataloaders(
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = 16,
    variants: List[str] = None,
) -> Tuple[DataLoader, DataLoader, AutoTokenizer]:
    """
    Create train and dev DataLoaders with tokenization.

    Args:
        model_name: Hugging Face model name for tokenizer.
        max_length: Max token sequence length.
        batch_size: Batch size for DataLoaders.
        variants: Language variants to include.

    Returns:
        (train_loader, dev_loader, tokenizer)
    """
    print(f"\n{'=' * 60}")
    print(f"  Creating DataLoaders")
    print(f"  Model: {model_name}")
    print(f"  Max length: {max_length} | Batch size: {batch_size}")
    print(f"{'=' * 60}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load data
    train_df = load_combined_dataset("train", variants)
    dev_df = load_combined_dataset("dev", variants)

    # Create datasets
    train_dataset = HallucinationDataset(train_df, tokenizer, max_length)
    dev_dataset = HallucinationDataset(dev_df, tokenizer, max_length)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"\n  ✓ Train DataLoader: {len(train_loader):,} batches")
    print(f"  ✓ Dev DataLoader:   {len(dev_loader):,} batches")

    return train_loader, dev_loader, tokenizer


def get_tokenizer(model_name: str = DEFAULT_MODEL_NAME) -> AutoTokenizer:
    """Load and return the tokenizer."""
    return AutoTokenizer.from_pretrained(model_name)
