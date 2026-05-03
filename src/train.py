"""
Training Script
================
Fine-tune XLM-RoBERTa-Large on the FEVER Urdu hallucination detection task
using Hugging Face Trainer API with FP16 mixed precision.

Optimized for RTX 4090 (24GB VRAM).
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.model import create_model, DEFAULT_MODEL_NAME, NUM_LABELS
from src.preprocess import (
    HallucinationDataset,
    load_combined_dataset,
    DEFAULT_MAX_LENGTH,
)
from src.utils import MODELS_DIR, FRIENDLY_LABELS, ensure_dirs


# ============================================
# Metrics Computation
# ============================================

def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute classification metrics for the Trainer.

    Returns:
        Dict with accuracy, macro_f1, macro_precision, macro_recall,
        and per-class F1 scores.
    """
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
    for i, f1_val in enumerate(per_class_f1):
        label_name = FRIENDLY_LABELS.get(i, f"class_{i}")
        metrics[f"f1_{label_name.replace(' ', '_').lower()}"] = f1_val

    return metrics


# ============================================
# Training Function
# ============================================

def train_model(
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 4,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    fp16: bool = True,
    variants: List[str] = None,
    output_dir: str = None,
    early_stopping_patience: int = 2,
    gradient_accumulation_steps: int = 1,
) -> str:
    """
    Fine-tune XLM-RoBERTa on the FEVER Urdu hallucination detection task.

    Args:
        model_name: Pretrained model name.
        max_length: Max token sequence length.
        batch_size: Training batch size (16 fits RTX 4090).
        learning_rate: Peak learning rate.
        num_epochs: Number of training epochs.
        warmup_ratio: Fraction of steps for LR warmup.
        weight_decay: L2 regularization weight.
        fp16: Use mixed precision (FP16) for faster training.
        variants: Language variants to include in training data.
        output_dir: Directory to save checkpoints.
        early_stopping_patience: Stop after N evals without improvement.
        gradient_accumulation_steps: Accumulate gradients over N steps.

    Returns:
        Path to the best saved checkpoint.
    """
    ensure_dirs()

    if output_dir is None:
        output_dir = str(MODELS_DIR / "xlm-roberta-hallucination")

    print("\n" + "=" * 60)
    print("  🚀 TRAINING — Hallucination Detector")
    print("=" * 60)
    print(f"  Model:          {model_name}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Learning rate:  {learning_rate}")
    print(f"  Epochs:         {num_epochs}")
    print(f"  Max length:     {max_length}")
    print(f"  FP16:           {fp16}")
    print(f"  Output dir:     {output_dir}")
    print(f"  Device:         {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  GPU:            {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:           {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    # ---- Load tokenizer & data ----
    print("\n📄 Loading tokenizer and datasets...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_df = load_combined_dataset("train", variants)
    dev_df = load_combined_dataset("dev", variants)

    train_dataset = HallucinationDataset(train_df, tokenizer, max_length)
    dev_dataset = HallucinationDataset(dev_df, tokenizer, max_length)

    print(f"\n  Train samples: {len(train_dataset):,}")
    print(f"  Dev samples:   {len(dev_dataset):,}")

    # ---- Load model ----
    model = create_model(model_name=model_name)

    # ---- Training arguments ----
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        fp16=fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",  # Disable wandb/tensorboard
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

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
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

    # Evaluate on dev set
    eval_results = trainer.evaluate()
    for key, value in sorted(eval_results.items()):
        if key.startswith("eval_"):
            print(f"  {key:25s}: {value:.4f}")

    print(f"\n  📁 Best model saved to: {best_model_path}")
    print("=" * 60 + "\n")

    return best_model_path
