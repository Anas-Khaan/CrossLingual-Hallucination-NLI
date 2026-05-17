"""
Visualization Script — Real Model Predictions
================================================
Generates publication-ready figures from actual model predictions
on FEVER Urdu (in-domain) and XNLI Pure Urdu (cross-domain).

Outputs:
  reports/figures/confusion_matrices_comparison.png
  reports/figures/per_class_f1_comparison.png
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    matthews_corrcoef, accuracy_score,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ============================================
# Paths
# ============================================
BASE_DIR = Path(__file__).parent
FEVER_TEST_PATH = BASE_DIR / "new_data" / "urdu" / "test.csv"
XNLI_PATH = BASE_DIR / "new_data" / "xnli_pure_urdu.csv"
MODEL_DIR = BASE_DIR / "models" / "checkpoints" / "xlm-roberta-new" / "best_model"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

MAX_LENGTH = 512
LABELS = ["Supported", "Not Supported", "Not Enough Info"]


# ============================================
# Dataset
# ============================================
class NLIDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
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
# Inference
# ============================================
def get_predictions(model, tokenizer, df, device, batch_size=32, desc="Predicting"):
    """Run inference and return (y_true, y_pred) arrays."""
    dataset = NLIDataset(df, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


# ============================================
# Visualizations
# ============================================
def plot_confusion_matrices(y_true_fever, y_pred_fever, y_true_xnli, y_pred_xnli):
    """Side-by-side normalized confusion matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.set_theme(style="white")

    # FEVER Heatmap
    cm_fever = confusion_matrix(y_true_fever, y_pred_fever, normalize='true')
    sns.heatmap(cm_fever, annot=True, fmt='.2f', cmap='Blues', ax=axes[0],
                xticklabels=LABELS, yticklabels=LABELS, cbar=False,
                annot_kws={"size": 13})
    axes[0].set_title('Urdu-FEVER (In-Domain)', fontsize=14, fontweight='bold', pad=10)
    axes[0].set_ylabel('Actual Ground Truth', fontsize=12)
    axes[0].set_xlabel('Model Prediction', fontsize=12)

    # XNLI Heatmap
    cm_xnli = confusion_matrix(y_true_xnli, y_pred_xnli, normalize='true')
    sns.heatmap(cm_xnli, annot=True, fmt='.2f', cmap='Oranges', ax=axes[1],
                xticklabels=LABELS, yticklabels=LABELS, cbar=False,
                annot_kws={"size": 13})
    axes[1].set_title('Urdu-XNLI (Zero-Shot Cross-Domain)', fontsize=14, fontweight='bold', pad=10)
    axes[1].set_ylabel('', fontsize=12)
    axes[1].set_xlabel('Model Prediction', fontsize=12)

    plt.tight_layout()
    save_path = FIGURES_DIR / "confusion_matrices_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {save_path}")
    plt.close()


def plot_f1_comparison(y_true_fever, y_pred_fever, y_true_xnli, y_pred_xnli):
    """Per-class F1-score grouped bar chart."""
    _, _, f1_fever, _ = precision_recall_fscore_support(y_true_fever, y_pred_fever, average=None, zero_division=0)
    _, _, f1_xnli, _ = precision_recall_fscore_support(y_true_xnli, y_pred_xnli, average=None, zero_division=0)

    x = np.arange(len(LABELS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, f1_fever, width, label='Urdu-FEVER (In-Domain)', color='#2b5c8f')
    rects2 = ax.bar(x + width/2, f1_xnli, width, label='Urdu-XNLI (Zero-Shot)', color='#e67e22')

    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Per-Class F1-Score: In-Domain vs Zero-Shot', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Add value labels on bars
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    save_path = FIGURES_DIR / "per_class_f1_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {save_path}")
    plt.close()


def print_metrics(y_true, y_pred, name):
    """Print the 5 key metrics."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"  {name}:")
    print(f"    Accuracy:        {acc*100:.2f}%")
    print(f"    Macro Precision: {p:.4f}")
    print(f"    Macro Recall:    {r:.4f}")
    print(f"    Macro F1:        {f:.4f}")
    print(f"    MCC:             {mcc:.4f}")


# ============================================
# Main
# ============================================
def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("  📊 Generating Visualizations (Real Data)")
    print("=" * 60)
    print(f"  Device: {device}")

    # Load model
    print("\n  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()

    # --- FEVER Test Predictions ---
    print("\n  Loading FEVER Urdu test data...")
    fever_df = pd.read_csv(FEVER_TEST_PATH, encoding="utf-8-sig")
    print(f"  ✓ FEVER test: {len(fever_df):,} samples")

    y_true_fever, y_pred_fever = get_predictions(
        model, tokenizer, fever_df, device, batch_size=32, desc="  FEVER Test"
    )

    # --- XNLI Predictions ---
    print("\n  Loading XNLI Pure Urdu data...")
    xnli_df = pd.read_csv(XNLI_PATH, encoding="utf-8-sig")
    xnli_df.columns = [c.lower() for c in xnli_df.columns]
    print(f"  ✓ XNLI:       {len(xnli_df):,} samples")

    y_true_xnli, y_pred_xnli = get_predictions(
        model, tokenizer, xnli_df, device, batch_size=32, desc="  XNLI Urdu"
    )

    # --- Generate Figures ---
    print("\n  Generating figures...")
    plot_confusion_matrices(y_true_fever, y_pred_fever, y_true_xnli, y_pred_xnli)
    plot_f1_comparison(y_true_fever, y_pred_fever, y_true_xnli, y_pred_xnli)

    # --- Print Metrics ---
    print("\n" + "=" * 60)
    print("  METRICS SUMMARY")
    print("=" * 60)
    print_metrics(y_true_fever, y_pred_fever, "FEVER Urdu (In-Domain)")
    print()
    print_metrics(y_true_xnli, y_pred_xnli, "XNLI Pure Urdu (Zero-Shot)")
    print("=" * 60)

    print(f"\n  📁 Figures saved to: {FIGURES_DIR}")
    print("  ✅ Done!\n")


if __name__ == "__main__":
    main()
