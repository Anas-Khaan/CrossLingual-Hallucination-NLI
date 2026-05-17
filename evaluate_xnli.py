"""
XNLI Evaluation Script (Pure Urdu)
====================================
Evaluate the Pure-Urdu fine-tuned model on the XNLI Pure Urdu dataset
for cross-domain zero-shot performance.

Usage:
    python evaluate_xnli.py
    python evaluate_xnli.py --batch-size 32
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, matthews_corrcoef,
)
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ============================================
# Paths & Config
# ============================================
BASE_DIR = Path(__file__).parent
XNLI_PATH = BASE_DIR / "new_data" / "xnli_pure_urdu.csv"
MODEL_DIR = BASE_DIR / "models" / "checkpoints" / "xlm-roberta-new" / "best_model"

NUM_LABELS = 3
MAX_LENGTH = 512
LABEL_NAMES = ["Supported", "Not Supported", "Not Enough Info"]


# ============================================
# Dataset
# ============================================
class XNLIDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # XNLI uses 'Claim' (capital C) and 'evidence'
        claim = str(row["claim"]) if pd.notna(row["claim"]) else ""
        evidence = str(row["evidence"]) if pd.notna(row["evidence"]) else ""

        if evidence.strip().lower() == "nan":
            evidence = ""

        encoding = self.tokenizer(
            evidence, claim,
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
# Evaluation
# ============================================
def evaluate_xnli(batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("  📊 EVALUATION — Cross-Domain Zero-Shot (XNLI Pure Urdu)")
    print("=" * 60)
    print(f"  Model:    {MODEL_DIR}")
    print(f"  Dataset:  {XNLI_PATH}")
    print(f"  Device:   {device}")
    print("=" * 60)

    # Load model
    print("\n  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()

    # Load XNLI data
    print("  Loading XNLI Pure Urdu data...")
    df = pd.read_csv(XNLI_PATH, encoding="utf-8-sig")

    # Normalize column names (XNLI uses 'Claim' with capital C)
    df.columns = [c.lower() for c in df.columns]

    print(f"  ✓ Loaded: {len(df):,} rows")
    print(f"  Label distribution:")
    for label_text, count in df["label_text"].value_counts().items():
        print(f"    {label_text:20s}: {count:,}")

    # Create dataset & loader
    dataset = XNLIDataset(df, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"\n  {'─' * 50}")
    print(f"  Evaluating: XNLI Pure Urdu ({len(df):,} samples)")
    print(f"  {'─' * 50}")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Eval XNLI", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_p = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_r = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)

    print(f"\n  Results (XNLI Pure Urdu):")
    print(f"    Accuracy:         {acc:.4f} ({acc*100:.2f}%)")
    print(f"    Macro Precision:  {macro_p:.4f}")
    print(f"    Macro Recall:     {macro_r:.4f}")
    print(f"    Macro F1:         {macro_f1:.4f}")
    print(f"    Matthews Corr:    {mcc:.4f}")

    # Classification report
    print(f"\n  Classification Report (XNLI Pure Urdu):")
    report = classification_report(
        all_labels, all_preds,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )
    for line in report.split("\n"):
        print(f"    {line}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n  Confusion Matrix (XNLI Pure Urdu):")
    print(f"    {'':20s} Pred:Sup  Pred:Not  Pred:NEI")
    for i, label in enumerate(LABEL_NAMES):
        row = "    ".join(f"{v:8d}" for v in cm[i])
        print(f"    {label:20s} {row}")

    print("\n" + "=" * 60)
    print("  ✅ XNLI Evaluation Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate on XNLI Pure Urdu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    evaluate_xnli(batch_size=args.batch_size)
