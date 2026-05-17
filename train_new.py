"""
Training Script for New Data (English + Pure Urdu)
====================================================
Fine-tune XLM-RoBERTa-Large on the 20K dataset from new_data/.
Combines English + Urdu variants for cross-lingual training.
Includes class weights to fix the "Not Supported" F1=0 issue.

Usage:
    python train_new.py
    python train_new.py --epochs 6 --batch-size 16
    python train_new.py --variants urdu          # Urdu only
    python train_new.py --variants english       # English only
    python train_new.py --variants english,urdu  # Both (default)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ============================================
# Paths & Config
# ============================================
BASE_DIR = Path(__file__).parent
NEW_DATA_DIR = BASE_DIR / "new_data"
ENGLISH_DIR = NEW_DATA_DIR / "english"
URDU_DIR = NEW_DATA_DIR / "urdu"
OUTPUT_DIR = BASE_DIR / "models" / "checkpoints" / "xlm-roberta-new"

MODEL_NAME = "xlm-roberta-large"
NUM_LABELS = 3
MAX_LENGTH = 512

LABEL_NAMES = ["Supported", "Not Supported", "Not Enough Info"]


# ============================================
# Dataset
# ============================================
class HallucinationDataset(Dataset):
    """PyTorch Dataset for NLI-style hallucination detection."""

    def __init__(self, df, tokenizer, max_length=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        evidence = str(row["evidence"]) if pd.notna(row["evidence"]) else ""
        claim = str(row["claim"]) if pd.notna(row["claim"]) else ""

        # Handle "nan" string evidence (NOT ENOUGH INFO class)
        if evidence.strip().lower() == "nan":
            evidence = ""

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
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }


# ============================================
# Weighted Trainer (fixes Not Supported F1=0)
# ============================================
class WeightedTrainer(Trainer):
    """Custom Trainer that applies class weights to the loss function."""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ============================================
# Metrics
# ============================================
def compute_metrics(eval_pred):
    """Compute classification metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "macro_precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, predictions, average="macro", zero_division=0),
    }

    # Per-class F1
    per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
    for i, label_name in enumerate(LABEL_NAMES):
        safe_name = label_name.lower().replace(" ", "_")
        if i < len(per_class_f1):
            metrics[f"f1_{safe_name}"] = per_class_f1[i]

    return metrics


# ============================================
# Data Loading
# ============================================
def load_data(split: str, variants: list = None):
    """
    Load and combine data from specified variants.

    Args:
        split: "train", "val", or "test"
        variants: List of variants to load, e.g. ["english", "urdu"]. Default: both.
    """
    if variants is None:
        variants = ["english", "urdu"]

    variant_dirs = {
        "english": ENGLISH_DIR,
        "urdu": URDU_DIR,
    }

    dfs = []
    for variant in variants:
        path = variant_dirs[variant] / f"{split}.csv"
        if path.exists():
            df = pd.read_csv(path, encoding="utf-8-sig")
            print(f"  ✓ Loaded {variant}/{split}.csv: {len(df):,} rows")
            dfs.append(df)
        else:
            print(f"  ⚠ Missing: {path}")

    if not dfs:
        raise FileNotFoundError(f"No data found for split '{split}'")

    combined = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  → Combined {split}: {len(combined):,} rows")
    return combined


def compute_class_weights(df):
    """
    Compute inverse-frequency class weights.
    Gives higher weight to underrepresented classes (like REFUTES).
    """
    label_counts = Counter(df["label"].values)
    total = len(df)
    num_classes = len(label_counts)

    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        # Inverse frequency: total / (num_classes * count)
        w = total / (num_classes * count)
        weights.append(w)

    print(f"\n  Class weights (inverse frequency):")
    for i, (name, w) in enumerate(zip(LABEL_NAMES, weights)):
        count = label_counts.get(i, 0)
        print(f"    {name:20s}: weight={w:.4f}  (count={count:,})")

    return weights


# ============================================
# Training
# ============================================
def train_model(
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 4,
    fp16: bool = True,
    variants: list = None,
    use_class_weights: bool = True,
    early_stopping_patience: int = 2,
):
    """Fine-tune XLM-RoBERTa-Large on the new dataset."""

    output_dir = str(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    if variants is None:
        variants = ["english", "urdu"]

    print("\n" + "=" * 60)
    print("  🏋️ Training — Hallucination Detector (New Data)")
    print("=" * 60)
    print(f"  Model:          {MODEL_NAME}")
    print(f"  Variants:       {', '.join(variants)}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Learning rate:  {learning_rate}")
    print(f"  Epochs:         {num_epochs}")
    print(f"  Max length:     {MAX_LENGTH}")
    print(f"  FP16:           {fp16}")
    print(f"  Class weights:  {use_class_weights}")
    print(f"  Output dir:     {output_dir}")
    print(f"  Device:         {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  GPU:            {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:           {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    # ---- Load tokenizer & data ----
    print("\n📄 Loading tokenizer and datasets...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_df = load_data("train", variants)
    val_df = load_data("val", variants)

    train_dataset = HallucinationDataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset = HallucinationDataset(val_df, tokenizer, MAX_LENGTH)

    print(f"\n  Train samples: {len(train_dataset):,}")
    print(f"  Val samples:   {len(val_dataset):,}")

    # ---- Class weights ----
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(train_df)

    # ---- Model ----
    print(f"\n🤖 Loading {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )

    # ---- Training args ----
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    # ---- Callbacks ----
    callbacks = []
    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        )

    # ---- Trainer (with class weights) ----
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # ---- Train ----
    print("\n🏋️ Starting training...\n")
    train_result = trainer.train()

    # ---- Save best model ----
    best_model_path = os.path.join(output_dir, "best_model")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)

    # ---- Print results ----
    print("\n" + "=" * 60)
    print("  ✅ Training Complete!")
    print("=" * 60)
    print(f"  Train loss:     {train_result.training_loss:.4f}")

    eval_results = trainer.evaluate()
    for key, value in sorted(eval_results.items()):
        if key.startswith("eval_"):
            print(f"  {key:25s}: {value:.4f}")

    print(f"\n  📁 Best model saved to: {best_model_path}")
    print("=" * 60 + "\n")


# ============================================
# Main
# ============================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train on new 20K dataset")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Disable class weights (not recommended)")
    parser.add_argument("--variants", type=str, default="english,urdu",
                        help="Comma-separated: english,urdu (default: both)")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",")]

    train_model(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        fp16=not args.no_fp16,
        variants=variants,
        use_class_weights=not args.no_class_weights,
    )
