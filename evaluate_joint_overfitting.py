"""
Joint Overfitting Test: Evidence-Shuffling
==========================================
Methodology:
1. Load FEVER Urdu test data.
2. Filter for samples labeled "Supported" (label == 0).
3. Sample 1,000 rows.
4. Keep the claims intact but randomly shuffle the `evidence` texts among these rows.
   (This breaks the factual link while keeping the text style perfectly matched).
5. Run inference on this corrupted dataset.
6. A model that truly understands reasoning should NOT predict "Supported".
   If it still predicts "Supported" for a large percentage, it's jointly overfitting.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
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
SAMPLE_SIZE = 1000

# Mapping (for reference, though we assume 0: Supported, 1: Not Supported, 2: Not Enough Info)
LABEL_MAP = {0: "Supported", 1: "Not Supported", 2: "Not Enough Info"}

# ============================================
# Dataset
# ============================================
class ShuffledEvidenceDataset(Dataset):
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
def run_inference(model, tokenizer, df, device, batch_size=32):
    dataset = ShuffledEvidenceDataset(df, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating Shuffled Data", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)

    return np.array(all_preds)

# ============================================
# Visualization
# ============================================
def plot_joint_overfitting(preds, title="Joint Overfitting Test: Evidence Shuffling"):
    # Count predictions for each class
    counts = {0: 0, 1: 0, 2: 0}
    for p in preds:
        counts[p] += 1
        
    total = len(preds)
    percentages = {k: (v / total) * 100 for k, v in counts.items()}
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_theme(style="whitegrid")

    categories = ['Supported', 'Not Supported', 'Not Enough Info']
    values = [percentages[0], percentages[1], percentages[2]]
    colors = ['#2ca02c', '#d62728', '#1f77b4'] # Green, Red, Blue

    bars = ax.bar(categories, values, color=colors, width=0.6)

    ax.set_ylabel('Percentage of Predictions (%)', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)

    # Add text labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add a descriptive text box
    desc_text = (
        "Input: 1,000 'Supported' claims with SHUFFLED evidence.\n"
        "Expected: Model should NOT predict 'Supported'.\n"
        "If 'Supported' is high -> Memorized spurious text style."
    )
    plt.figtext(0.5, -0.05, desc_text, wrap=True, horizontalalignment='center', fontsize=11, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    save_path = FIGURES_DIR / "joint_overfitting_proof.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved chart to: {save_path}")
    plt.close()

# ============================================
# Main
# ============================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("  🧪 JOINT OVERFITTING TEST (Evidence Shuffling)")
    print("=" * 60)
    
    # 1. Load Data
    print("\n  Loading FEVER Urdu test data...")
    df = pd.read_csv(FEVER_TEST_PATH, encoding="utf-8-sig")
    
    # 2. Filter for 'Supported' (label == 0)
    # Assuming labels: 0=Supported, 1=Not Supported, 2=Not Enough Info
    supported_df = df[df["label"] == 0].copy()
    print(f"  ✓ Found {len(supported_df):,} 'Supported' samples.")
    
    # 3. Sample 1000 rows (or max available)
    sample_size = min(SAMPLE_SIZE, len(supported_df))
    shuffled_df = supported_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    print(f"  ✓ Sampled {sample_size:,} rows.")
    
    # 4. Shuffle the Evidence
    print("  ✓ Shuffling evidence completely...")
    # To ensure no evidence stays with its original claim, we can use a permutation
    # A simple numpy permutation
    shuffled_evidence = shuffled_df["evidence"].values.copy()
    np.random.seed(42)
    np.random.shuffle(shuffled_evidence)
    
    # Check if any randomly stayed the same and fix if strictness is required (optional)
    # But simple shuffle is usually sufficient for 1000 items (chance of staying is ~1/1000)
    shuffled_df["evidence"] = shuffled_evidence

    # 5. Load model
    print("\n  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()

    # 6. Run Inference
    print(f"\n  Running model on {sample_size} disconnected Claim-Evidence pairs...")
    preds = run_inference(model, tokenizer, shuffled_df, device, batch_size=32)

    # 7. Analyze
    supported_preds = np.sum(preds == 0)
    not_supported_preds = np.sum(preds == 1)
    nei_preds = np.sum(preds == 2)
    
    supported_pct = (supported_preds / sample_size) * 100
    not_supported_pct = (not_supported_preds / sample_size) * 100
    nei_pct = (nei_preds / sample_size) * 100
    
    print("\n  ================ PREDICTION RESULTS ================")
    print(f"  Supported:       {supported_pct:.1f}% ({supported_preds}/{sample_size})")
    print(f"  Not Supported:   {not_supported_pct:.1f}% ({not_supported_preds}/{sample_size})")
    print(f"  Not Enough Info: {nei_pct:.1f}% ({nei_preds}/{sample_size})")
    print("  ====================================================")
    
    if supported_pct > 30.0:
        print("\n  ⚠️ CONCLUSION: High degree of Joint Overfitting detected.")
        print("  The model blindly predicted 'Supported' because the text style matched,")
        print("  even though the facts were completely unrelated.")
    else:
        print("\n  ✅ CONCLUSION: Model did NOT overfit to joint text styles.")
        print("  It recognized that the claims were no longer supported by the shuffled evidence.")

    print("\n  Generating visual proof...")
    plot_joint_overfitting(preds)
    
    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
