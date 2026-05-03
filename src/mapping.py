"""
Evidence Mapping
================
Parse the FEVER wiki dump and map evidence sentences to claims.
Produces interim CSV files with (claim, evidence, label) triples.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from src.utils import (
    RAW_DIR, WIKI_DIR, INTERIM_DIR,
    LABEL_MAP, clean_text,
    load_jsonl, load_jsonl_lazy,
    ensure_dirs,
)


# ============================================
# Wiki Dump Parsing
# ============================================

def build_wiki_lookup(wiki_dir: Path = WIKI_DIR, max_files: int = None) -> Dict[str, Dict[int, str]]:
    """
    Build a lookup dictionary from the FEVER wiki dump.

    Structure: {page_title: {sentence_id: sentence_text}}

    Args:
        wiki_dir: Path to the directory containing wiki-XXX.jsonl files.
        max_files: Max number of wiki files to load (None = all). Useful for testing.

    Returns:
        Nested dict mapping page titles to their sentences.
    """
    wiki_lookup = {}

    # Find all wiki JSONL files (they may be inside a subdirectory)
    # Filter out __MACOSX junk directory from zip extraction
    wiki_files = sorted(
        f for f in wiki_dir.rglob("*.jsonl")
        if "__MACOSX" not in str(f)
    )
    if not wiki_files:
        raise FileNotFoundError(
            f"No .jsonl files found in {wiki_dir}. "
            "Please run 'python main.py download' first."
        )

    if max_files:
        wiki_files = wiki_files[:max_files]

    print(f"\n📖 Loading wiki dump ({len(wiki_files)} files)...")

    for fpath in tqdm(wiki_files, desc="Loading wiki pages", unit="file"):
        for record in load_jsonl_lazy(fpath):
            page_id = record.get("id", "")
            lines_text = record.get("lines", "")

            if not lines_text:
                continue

            sentences = {}
            for line in lines_text.split("\n"):
                parts = line.split("\t")
                if len(parts) >= 2:
                    try:
                        sent_id = int(parts[0])
                        sent_text = parts[1]
                        if sent_text.strip():
                            sentences[sent_id] = clean_text(sent_text)
                    except (ValueError, IndexError):
                        continue

            if sentences:
                wiki_lookup[page_id] = sentences

    print(f"  ✓ Loaded {len(wiki_lookup):,} wiki pages")
    return wiki_lookup


# ============================================
# Evidence Extraction
# ============================================

def extract_evidence_text(
    claim_record: Dict,
    wiki_lookup: Dict[str, Dict[int, str]]
) -> Optional[str]:
    """
    Extract the evidence text for a single FEVER claim.

    Args:
        claim_record: A single FEVER claim record (dict).
        wiki_lookup: The wiki page lookup dictionary.

    Returns:
        Concatenated evidence sentences, or None if not found.
    """
    evidence_sets = claim_record.get("evidence", [])
    if not evidence_sets:
        return None

    evidence_sentences = []

    for evidence_set in evidence_sets:
        for annotation in evidence_set:
            # FEVER format: [annotation_id, evidence_id, page_title, sentence_id]
            if len(annotation) < 4:
                continue

            page_title = annotation[2]
            sentence_id = annotation[3]

            if page_title is None or sentence_id is None:
                continue

            # Normalize page title (underscores to match wiki dump format)
            page_title = str(page_title).replace(" ", "_")

            if page_title in wiki_lookup:
                sent = wiki_lookup[page_title].get(sentence_id, None)
                if sent and sent not in evidence_sentences:
                    evidence_sentences.append(sent)

    if evidence_sentences:
        return " ".join(evidence_sentences)
    return None


# ============================================
# Main Mapping Pipeline
# ============================================

def map_claims_to_evidence(
    split: str = "train",
    max_claims: int = None,
    max_wiki_files: int = None,
) -> pd.DataFrame:
    """
    Map FEVER claims to their evidence sentences.

    Args:
        split: "train" or "dev"
        max_claims: Max number of claims to process (None = all).
        max_wiki_files: Max wiki files to load (None = all).

    Returns:
        DataFrame with columns: [claim, evidence, label, label_text]
    """
    ensure_dirs()

    # Determine input file
    if split == "train":
        claims_path = RAW_DIR / "train.jsonl"
    elif split == "dev":
        claims_path = RAW_DIR / "shared_task_dev.jsonl"
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train' or 'dev'.")

    if not claims_path.exists():
        raise FileNotFoundError(
            f"Claims file not found: {claims_path}. "
            "Please run 'python main.py download' first."
        )

    # Build wiki lookup
    wiki_lookup = build_wiki_lookup(WIKI_DIR, max_files=max_wiki_files)

    # Load claims
    print(f"\n📝 Processing {split} claims from {claims_path.name}...")
    claims = load_jsonl(claims_path)

    if max_claims:
        claims = claims[:max_claims]

    print(f"  → {len(claims):,} claims to process")

    # Map each claim to evidence
    mapped_data = []
    skipped = 0

    for record in tqdm(claims, desc=f"Mapping {split}", unit="claim"):
        claim_text = clean_text(record.get("claim", ""))
        label_text = record.get("label", "")
        label_id = LABEL_MAP.get(label_text, -1)

        if label_id == -1:
            skipped += 1
            continue

        if label_text == "NOT ENOUGH INFO":
            # No evidence for NEI — use empty evidence
            evidence_text = ""
        else:
            evidence_text = extract_evidence_text(record, wiki_lookup)
            if evidence_text is None:
                # Could not find evidence in wiki dump
                evidence_text = ""

        mapped_data.append({
            "claim": claim_text,
            "evidence": evidence_text,
            "label": label_id,
            "label_text": label_text,
        })

    # Create DataFrame
    df = pd.DataFrame(mapped_data)

    # Save to interim
    output_path = INTERIM_DIR / f"{split}_mapped.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"  Mapping Summary — {split}")
    print(f"{'=' * 50}")
    print(f"  Total mapped:   {len(df):,}")
    print(f"  Skipped:        {skipped:,}")
    print(f"  Label distribution:")
    for label_text, count in df["label_text"].value_counts().items():
        print(f"    {label_text:20s}: {count:,}")
    print(f"  Empty evidence: {(df['evidence'] == '').sum():,}")
    print(f"  Saved to:       {output_path}")
    print(f"{'=' * 50}\n")

    return df


def map_all(max_claims: int = None, max_wiki_files: int = None, test_ratio: float = 0.2) -> None:
    """
    Map both train and dev splits, then split dev into dev + test.

    The FEVER test set labels are not publicly available, so we create
    our own test set by splitting the dev set:
      - dev  (80%) → validation during training (early stopping, best checkpoint)
      - test (20%) → held-out final evaluation (never seen during training)

    Args:
        max_claims: Max claims to process per split.
        max_wiki_files: Max wiki dump files to load.
        test_ratio: Fraction of dev set to hold out as test (default: 0.2).
    """
    from sklearn.model_selection import train_test_split

    # Map both splits
    for split in ["train", "dev"]:
        map_claims_to_evidence(
            split=split,
            max_claims=max_claims,
            max_wiki_files=max_wiki_files,
        )

    # Split dev into dev + test
    dev_path = INTERIM_DIR / "dev_mapped.csv"
    if dev_path.exists():
        print(f"\n{'=' * 50}")
        print(f"  Splitting dev set into dev + test")
        print(f"{'=' * 50}")

        df_dev_full = pd.read_csv(dev_path, encoding="utf-8-sig")

        df_dev, df_test = train_test_split(
            df_dev_full,
            test_size=test_ratio,
            random_state=42,
            stratify=df_dev_full["label"],  # preserve label distribution
        )

        # Save split files
        dev_split_path = INTERIM_DIR / "dev_mapped.csv"
        test_split_path = INTERIM_DIR / "test_mapped.csv"

        df_dev.to_csv(dev_split_path, index=False, encoding="utf-8-sig")
        df_test.to_csv(test_split_path, index=False, encoding="utf-8-sig")

        print(f"  Original dev set: {len(df_dev_full):,} rows")
        print(f"  → Dev (validation): {len(df_dev):,} rows ({100*(1-test_ratio):.0f}%)")
        print(f"  → Test (held-out):  {len(df_test):,} rows ({100*test_ratio:.0f}%)")
        print(f"  Saved: {dev_split_path.name}, {test_split_path.name}")
        print(f"{'=' * 50}\n")
