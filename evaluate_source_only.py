"""
Ablation Test: Source-Only (Evidence-Only) Evaluation
=====================================================
Evaluates the model on the FEVER Urdu test set, but completely removes the claim.
If the model still performs well, it proves Premise Bias (e.g., memorizing negative 
words in the source text to automatically guess 'Not Supported').
Outputs a bar chart comparing:
1. FEVER (Full Input) - 92.70%
2. FEVER (Source-Only Input) - calculated here
3. XNLI (Full Input) - 40.70%
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
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
MODEL_DIR = BASE_DIR / "models" / "checkpoints" / "xlm-roberta-new" / "best_model"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

MAX_LENGTH = 512

# ============================================
# Dataset (Ablated)
# ============================================
class SourceOnlyDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        evidence = str(row["evidence"]) if pd.notna(row["evidence"]) else ""
        
        # ABLATION: Force claim to be empty string
        claim = ""

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
def get_ablation_accuracy(model, tokenizer, df, device, batch_size=32):
    dataset = SourceOnlyDataset(df, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Source-Only Eval", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return accuracy_score(all_labels, all_preds) * 100.0

# ============================================
# Visualization
# ============================================
def plot_ablation_chart(source_only_acc):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_theme(style="whitegrid")

    categories = ['FEVER\n(Full Input)', 'FEVER\n(Source-Only Input)', 'XNLI\n(Full Input)']
    accuracies = [92.70, source_only_acc, 40.70]
    colors = ['#2ca02c', '#d62728', '#1f77b4'] # Green, Red, Blue

    bars = ax.bar(categories, accuracies, color=colors, width=0.6)

    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Proof of Premise Bias (Source-Only Ablation)', fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)

    # Add a dashed line at 33.3% (random guess for 3 classes)
    ax.axhline(33.33, color='black', linestyle='--', alpha=0.7, label="Random Guess (33.3%)")
    ax.legend(fontsize=11)

    # Add text labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    save_path = FIGURES_DIR / "source_only_ablation_proof.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved proof chart to: {save_path}")
    plt.close()

# ============================================
# Main
# ============================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("  🧪 ABLATION TEST: Source-Only (Evidence-Only) Evaluation")
    print("=" * 60)
    
    # Load model
    print("\n  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()

    # Load Data
    print("\n  Loading FEVER Urdu test data...")
    df = pd.read_csv(FEVER_TEST_PATH, encoding="utf-8-sig")
    print(f"  ✓ Total claims: {len(df):,}")

    print("\n  Running model with Claim REMOVED...")
    source_only_acc = get_ablation_accuracy(model, tokenizer, df, device, batch_size=32)

    print(f"\n  Result: The model achieved {source_only_acc:.1f}% accuracy with NO claim!")
    
    if source_only_acc > 70.0:
        print("  ⚠️ CONCLUSION: Strong Premise Bias detected. The model relies heavily on source text statistics.")
    else:
        print("  ✅ CONCLUSION: Model did not suffer from massive Premise Bias. It requires the claim.")

    print("\n  Generating visual proof...")
    plot_ablation_chart(source_only_acc)
    
    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
