"""
New Data Preparation Script
=============================
Prepares a clean 20,000-sample dataset with only 2 language variants:
  1. Pure English  (original FEVER text)
  2. Pure Urdu     (translated via Google Translate)

Split:
  - Train:      12,000 samples
  - Validation:  4,000 samples
  - Test:        4,000 samples

All output goes to: new_data/
"""

import os
import sys
import time
import re
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from deep_translator import GoogleTranslator

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ============================================
# Paths
# ============================================
BASE_DIR = Path(__file__).parent
INTERIM_DIR = BASE_DIR / "data" / "interim"
NEW_DATA_DIR = BASE_DIR / "new_data"
ENGLISH_DIR = NEW_DATA_DIR / "english"
URDU_DIR = NEW_DATA_DIR / "urdu"

# ============================================
# Config
# ============================================
TOTAL_SAMPLES = 20000
TRAIN_SIZE = 12000
VAL_SIZE = 4000
TEST_SIZE = 4000
RANDOM_STATE = 42

RATE_LIMIT_DELAY = 0.1  # seconds between Google Translate calls
BATCH_SAVE_INTERVAL = 200  # save progress every N rows


# ============================================
# Step 1: Sample & Split English Data
# ============================================

def prepare_english_splits():
    """
    Load all mapped English data, sample 20,000 rows (stratified by label),
    and split into train/val/test.
    """
    print("\n" + "=" * 60)
    print("  📊 Step 1: Preparing English Data Splits")
    print("=" * 60)

    # Load all mapped data
    dfs = []
    for name in ["train_mapped.csv", "dev_mapped.csv", "test_mapped.csv"]:
        path = INTERIM_DIR / name
        if path.exists():
            df = pd.read_csv(path, encoding="utf-8-sig")
            dfs.append(df)
            print(f"  ✓ Loaded {name}: {len(df):,} rows")
        else:
            print(f"  ⚠ Missing: {name}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total available: {len(combined):,} rows")

    # Drop rows with missing claim (keep "nan" evidence — that's NOT ENOUGH INFO class)
    combined = combined.dropna(subset=["claim"]).reset_index(drop=True)
    print(f"  After cleaning:  {len(combined):,} rows")

    # Stratified sample of 20,000
    if len(combined) < TOTAL_SAMPLES:
        print(f"  ⚠ Only {len(combined)} rows available, using all")
        sampled = combined
    else:
        sampled, _ = train_test_split(
            combined,
            train_size=TOTAL_SAMPLES,
            stratify=combined["label"],
            random_state=RANDOM_STATE,
        )
        sampled = sampled.reset_index(drop=True)

    print(f"  Sampled: {len(sampled):,} rows")
    print(f"  Label distribution:")
    for label, count in sampled["label_text"].value_counts().items():
        print(f"    {label:20s}: {count:,}")

    # Split: 12k train / 4k val / 4k test
    train_df, temp_df = train_test_split(
        sampled,
        train_size=TRAIN_SIZE,
        stratify=sampled["label"],
        random_state=RANDOM_STATE,
    )
    val_df, test_df = train_test_split(
        temp_df,
        train_size=VAL_SIZE,
        stratify=temp_df["label"],
        random_state=RANDOM_STATE,
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save
    os.makedirs(ENGLISH_DIR, exist_ok=True)
    train_df.to_csv(ENGLISH_DIR / "train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(ENGLISH_DIR / "val.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(ENGLISH_DIR / "test.csv", index=False, encoding="utf-8-sig")

    print(f"\n  ✅ English splits saved to: {ENGLISH_DIR}")
    print(f"    Train: {len(train_df):,}")
    print(f"    Val:   {len(val_df):,}")
    print(f"    Test:  {len(test_df):,}")

    # Print label distribution per split
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n  {name} label distribution:")
        for label, count in df["label_text"].value_counts().items():
            print(f"    {label:20s}: {count:,}")


# ============================================
# Step 2: Translate to Pure Urdu
# ============================================

def translate_to_urdu(text: str) -> str:
    """Translate English text to Pure Urdu via Google Translate."""
    if not text or not text.strip():
        return ""
    try:
        translator = GoogleTranslator(source="en", target="ur")
        if len(text) > 4500:
            parts = _split_text(text, max_len=4500)
            translated = []
            for part in parts:
                result = translator.translate(part)
                translated.append(result if result else "")
                time.sleep(RATE_LIMIT_DELAY)
            return " ".join(translated)
        else:
            result = translator.translate(text)
            return result if result else ""
    except Exception as e:
        print(f"  ⚠ Translation error: {e}")
        return ""


def _split_text(text: str, max_len: int = 4500):
    """Split text into chunks respecting sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    parts = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > max_len:
            if current:
                parts.append(current)
            current = sent
        else:
            current = f"{current} {sent}".strip()
    if current:
        parts.append(current)
    return parts if parts else [text]


def translate_split(split: str):
    """
    Translate one English split (train/val/test) to Pure Urdu.
    Supports resume if interrupted.
    """
    english_path = ENGLISH_DIR / f"{split}.csv"
    urdu_path = URDU_DIR / f"{split}.csv"
    progress_file = URDU_DIR / f".progress_{split}.txt"
    partial_file = URDU_DIR / f".partial_{split}.csv"

    os.makedirs(URDU_DIR, exist_ok=True)

    # Skip if already done
    if urdu_path.exists() and not progress_file.exists():
        print(f"\n  ✓ {split} Urdu translation already exists — skipping")
        return

    # Load English data
    df = pd.read_csv(english_path, encoding="utf-8-sig")
    total = len(df)

    print(f"\n{'=' * 60}")
    print(f"  🌐 Translating {split} to Pure Urdu — {total:,} rows")
    print(f"{'=' * 60}\n")

    # Resume support
    start_idx = 0
    if progress_file.exists():
        start_idx = int(progress_file.read_text().strip())
        print(f"  ↻ Resuming from row {start_idx:,}")
        if partial_file.exists():
            df = pd.read_csv(partial_file, encoding="utf-8-sig")

    if start_idx == 0:
        df["claim_urdu"] = ""
        df["evidence_urdu"] = ""

    # Translate row by row
    for idx in tqdm(range(start_idx, total), desc=f"Translating {split}", unit="row"):
        claim_en = str(df.at[idx, "claim"])
        evidence_en = str(df.at[idx, "evidence"])

        claim_urdu = translate_to_urdu(claim_en)
        time.sleep(RATE_LIMIT_DELAY)

        evidence_urdu = translate_to_urdu(evidence_en) if evidence_en else ""
        time.sleep(RATE_LIMIT_DELAY)

        df.at[idx, "claim_urdu"] = claim_urdu
        df.at[idx, "evidence_urdu"] = evidence_urdu

        # Periodic save
        if (idx + 1) % BATCH_SAVE_INTERVAL == 0:
            progress_file.write_text(str(idx + 1))
            df.to_csv(partial_file, index=False, encoding="utf-8-sig")
            print(f"  💾 Progress saved at row {idx + 1:,}")

    # Save final Urdu CSV
    df_urdu = df[["claim_urdu", "evidence_urdu", "label", "label_text"]].copy()
    df_urdu.columns = ["claim", "evidence", "label", "label_text"]
    df_urdu.to_csv(urdu_path, index=False, encoding="utf-8-sig")

    # Cleanup
    for f in [progress_file, partial_file]:
        if f.exists():
            f.unlink()

    print(f"\n  ✅ Saved: {urdu_path} ({len(df_urdu):,} rows)")


def translate_all_splits():
    """Translate all 3 splits to Pure Urdu."""
    for split in ["train", "val", "test"]:
        translate_split(split)


# ============================================
# Main
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare new 20K dataset (English + Urdu)")
    parser.add_argument("--step", choices=["split", "translate", "all"], default="all",
                        help="Which step to run: 'split' (English only), 'translate' (Urdu), or 'all'")
    args = parser.parse_args()

    if args.step in ("split", "all"):
        prepare_english_splits()

    if args.step in ("translate", "all"):
        translate_all_splits()

    print("\n" + "=" * 60)
    print("  🎉 New data preparation complete!")
    print("=" * 60)
    print(f"  📁 English data: {ENGLISH_DIR}")
    print(f"  📁 Urdu data:    {URDU_DIR}")
    print(f"\n  Structure:")
    print(f"    new_data/")
    print(f"    ├── english/")
    print(f"    │   ├── train.csv  (12,000 rows)")
    print(f"    │   ├── val.csv    ( 4,000 rows)")
    print(f"    │   └── test.csv   ( 4,000 rows)")
    print(f"    └── urdu/")
    print(f"        ├── train.csv  (12,000 rows)")
    print(f"        ├── val.csv    ( 4,000 rows)")
    print(f"        └── test.csv   ( 4,000 rows)")
    print("=" * 60 + "\n")
