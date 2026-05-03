"""
Evaluation Module
=================
Evaluate the trained hallucination detector on dev/test sets.
Produces: accuracy, precision, recall, F1 (per-class & macro),
confusion matrix visualization, and error analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

from src.preprocess import HallucinationDataset, load_combined_dataset, load_variant_csv
from src.utils import FRIENDLY_LABELS, MODELS_DIR


# ============================================
# Evaluation Function
# ============================================

def evaluate_model(
    model_path: str = None,
    split: str = "test",
    variants: List[str] = None,
    batch_size: int = 16,
    max_length: int = 512,
    save_plots: bool = True,
    output_dir: str = None,
) -> Dict[str, float]:
    """
    Evaluate the trained model on a dataset split.

    Args:
        model_path: Path to the saved model checkpoint.
        split: "dev" or "test".
        variants: Language variants to evaluate on. None = all three.
        batch_size: Evaluation batch size.
        max_length: Max token length.
        save_plots: Whether to save confusion matrix plots.
        output_dir: Directory to save evaluation outputs.

    Returns:
        Dictionary of evaluation metrics.
    """
    if model_path is None:
        model_path = str(MODELS_DIR / "xlm-roberta-hallucination" / "best_model")

    if output_dir is None:
        output_dir = str(Path(model_path).parent / "eval_results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("  📊 EVALUATION — Hallucination Detector")
    print("=" * 60)
    print(f"  Model:    {model_path}")
    print(f"  Split:    {split}")
    print(f"  Variants: {variants or 'all'}")
    print(f"  Device:   {device}")
    print("=" * 60)

    # Load model and tokenizer
    print("\n  Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Overall evaluation (combined)
    print("\n" + "-" * 40)
    print("  Combined Evaluation (all variants)")
    print("-" * 40)
    combined_df = load_combined_dataset(split, variants)
    all_metrics = _evaluate_split(
        model, tokenizer, combined_df, device, batch_size, max_length,
        save_plots, output_dir, "combined"
    )

    # Per-variant evaluation
    eval_variants = variants or ["pure_urdu", "mixed", "roman_urdu"]
    for variant in eval_variants:
        print(f"\n" + "-" * 40)
        print(f"  Variant: {variant}")
        print("-" * 40)
        try:
            variant_df = load_variant_csv(split, variant)
            _evaluate_split(
                model, tokenizer, variant_df, device, batch_size, max_length,
                save_plots, output_dir, variant
            )
        except FileNotFoundError:
            print(f"  ⚠ Skipping {variant} — file not found")

    return all_metrics


def _evaluate_split(
    model,
    tokenizer,
    df: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    max_length: int,
    save_plots: bool,
    output_dir: str,
    name: str,
) -> Dict[str, float]:
    """Run evaluation on a single DataFrame and print results."""

    dataset = HallucinationDataset(df, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {name}", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Classification report
    label_names = [FRIENDLY_LABELS[i] for i in range(3)]
    report = classification_report(
        all_labels, all_preds,
        target_names=label_names,
        digits=4,
        zero_division=0,
    )
    print(f"\n{report}")

    # Metrics dict
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }

    # Confusion matrix
    if save_plots:
        _plot_confusion_matrix(all_labels, all_preds, label_names, output_dir, name)

    # Error analysis — top misclassified examples
    _error_analysis(df, all_preds, all_labels, output_dir, name)

    return metrics


# ============================================
# Visualization
# ============================================

def _plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    label_names: List[str],
    output_dir: str,
    name: str,
) -> None:
    """Plot and save a confusion matrix heatmap."""

    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {name}", fontsize=14)
    plt.tight_layout()

    plot_path = Path(output_dir) / f"confusion_matrix_{name}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  📈 Confusion matrix saved: {plot_path.name}")


# ============================================
# Error Analysis
# ============================================

def _error_analysis(
    df: pd.DataFrame,
    preds: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    name: str,
    top_n: int = 10,
) -> None:
    """Show top misclassified examples for error analysis."""

    errors_mask = preds != labels
    error_count = errors_mask.sum()
    total = len(labels)

    print(f"\n  🔍 Error Analysis ({name}): {error_count}/{total} misclassified "
          f"({100 * error_count / total:.1f}%)")

    if error_count == 0:
        print("  🎉 No errors!")
        return

    error_df = df.iloc[np.where(errors_mask)[0]].copy()
    error_df = error_df.head(top_n)
    error_preds = preds[errors_mask][:top_n]

    print(f"\n  Top {min(top_n, len(error_df))} misclassified examples:")
    print("  " + "-" * 70)

    for i, (idx, row) in enumerate(error_df.iterrows()):
        true_label = FRIENDLY_LABELS.get(int(row["label"]), "?")
        pred_label = FRIENDLY_LABELS.get(int(error_preds[i]), "?")
        claim = str(row.get("claim", ""))[:80]
        evidence = str(row.get("evidence", ""))[:80]

        print(f"  [{i+1}] True: {true_label} | Pred: {pred_label}")
        print(f"      Claim:    {claim}")
        print(f"      Evidence: {evidence}")
        print("  " + "-" * 70)

    # Save errors to CSV
    errors_path = Path(output_dir) / f"errors_{name}.csv"
    full_error_df = df.iloc[np.where(errors_mask)[0]].copy()
    full_error_df["predicted_label"] = preds[errors_mask]
    full_error_df.to_csv(errors_path, index=False, encoding="utf-8-sig")
    print(f"  📄 Full error log saved: {errors_path.name}")
