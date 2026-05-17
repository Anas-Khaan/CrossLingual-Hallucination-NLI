"""
t-SNE Feature Space Visualization
=================================
Extracts [CLS] embeddings from the fine-tuned XLM-RoBERTa model for 500
FEVER samples and 500 XNLI samples. Projects them to 2D using t-SNE to
visually demonstrate Domain Shift.
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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ============================================
# Paths
# ============================================
BASE_DIR = Path(__file__).parent
FEVER_PATH = BASE_DIR / "new_data" / "urdu" / "test.csv"
XNLI_PATH = BASE_DIR / "new_data" / "xnli_pure_urdu.csv"
MODEL_DIR = BASE_DIR / "models" / "checkpoints" / "xlm-roberta-new" / "best_model"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

MAX_LENGTH = 512
SAMPLE_SIZE = 500

# ============================================
# Dataset
# ============================================
class EmbeddingsDataset(Dataset):
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
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }

# ============================================
# Extraction
# ============================================
def extract_embeddings(model, tokenizer, df, device, batch_size=16, desc="Extracting"):
    dataset = EmbeddingsDataset(df, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Request hidden states to get the raw representations before classification head
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            # Get the last hidden state (batch_size, sequence_length, hidden_size)
            last_hidden_state = outputs.hidden_states[-1]
            
            # Extract the [CLS] token representation (index 0)
            cls_embeddings = last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(cls_embeddings)
            
    return np.array(embeddings)

# ============================================
# Main
# ============================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("  🌌 EXTRACTING EMBEDDINGS FOR t-SNE")
    print("=" * 60)

    # Load model
    print("\n  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()

    # Load and sample Data
    print("\n  Sampling data...")
    fever_df = pd.read_csv(FEVER_PATH, encoding="utf-8-sig")
    if len(fever_df) > SAMPLE_SIZE:
        fever_df = fever_df.sample(n=SAMPLE_SIZE, random_state=42)
        
    xnli_df = pd.read_csv(XNLI_PATH, encoding="utf-8-sig")
    xnli_df.columns = [c.lower() for c in xnli_df.columns]
    if len(xnli_df) > SAMPLE_SIZE:
        xnli_df = xnli_df.sample(n=SAMPLE_SIZE, random_state=42)

    # Extract embeddings
    print("\n  Computing FEVER embeddings...")
    fever_embeds = extract_embeddings(model, tokenizer, fever_df, device, desc="FEVER")
    
    print("\n  Computing XNLI embeddings...")
    xnli_embeds = extract_embeddings(model, tokenizer, xnli_df, device, desc="XNLI")

    # Combine data for t-SNE
    print("\n  Running t-SNE dimensionality reduction (this may take a minute)...")
    X = np.vstack([fever_embeds, xnli_embeds])
    labels = ['FEVER (In-Domain)'] * len(fever_embeds) + ['XNLI (Out-of-Domain)'] * len(xnli_embeds)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_2d = tsne.fit_transform(X)
    
    # Plotting
    print("\n  Plotting scatter plot...")
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    
    # Create dataframe for plotting
    plot_df = pd.DataFrame({
        'Dim 1': X_2d[:, 0],
        'Dim 2': X_2d[:, 1],
        'Dataset': labels
    })
    
    sns.scatterplot(
        data=plot_df,
        x='Dim 1',
        y='Dim 2',
        hue='Dataset',
        palette={'FEVER (In-Domain)': '#2ca02c', 'XNLI (Out-of-Domain)': '#1f77b4'},
        alpha=0.7,
        s=60,
        edgecolor=None
    )
    
    plt.title('t-SNE Visualization of [CLS] Token Embeddings\nVisual Proof of Domain Shift', fontsize=16, fontweight='bold', pad=20)
    plt.legend(title='Dataset Origin', title_fontsize='12', fontsize='11', loc='best')
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    save_path = FIGURES_DIR / "tsne_domain_shift.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"\n  ✅ Saved t-SNE plot to: {save_path}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
