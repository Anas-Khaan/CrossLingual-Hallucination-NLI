"""
Translator Module
=================
Translates claims and evidence from English to three variants:
  1. Pure Urdu        — full Urdu script (اردو)
  2. Urdu-English Mixed — Urdu script with some English words retained
  3. Roman Urdu       — Urdu transliterated into Roman/English letters

Uses free Google Translate via `deep-translator` with rate limiting and caching.
"""

import os
import re
import time
import random
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

from deep_translator import GoogleTranslator

from src.utils import (
    INTERIM_DIR, FINAL_DIR,
    ensure_dirs, load_progress, save_progress,
)

# Attempt to import transliteration library
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as indic_transliterate
    HAS_INDIC_TRANSLIT = True
except ImportError:
    HAS_INDIC_TRANSLIT = False
    print("⚠ indic-transliteration not installed. Roman Urdu will use basic mapping.")


# ============================================
# Constants
# ============================================

# Pause between API calls to respect rate limits
RATE_LIMIT_DELAY = 0.1  # seconds (Google free API tolerates ~10 req/s)
BATCH_SAVE_INTERVAL = 200  # save progress every N rows

# Common English words to retain in mixed variant
# (proper nouns, technical terms, entity names)
RETAIN_ENGLISH_PATTERNS = [
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized words (likely proper nouns)
    r'\b\d+\b',                                 # Numbers
]

# Basic Roman Urdu transliteration map (fallback if indic_transliteration unavailable)
URDU_TO_ROMAN_MAP = {
    'ا': 'a', 'آ': 'aa', 'ب': 'b', 'پ': 'p', 'ت': 't', 'ٹ': 'T',
    'ث': 's', 'ج': 'j', 'چ': 'ch', 'ح': 'h', 'خ': 'kh', 'د': 'd',
    'ڈ': 'D', 'ذ': 'z', 'ر': 'r', 'ڑ': 'R', 'ز': 'z', 'ژ': 'zh',
    'س': 's', 'ش': 'sh', 'ص': 's', 'ض': 'z', 'ط': 't', 'ظ': 'z',
    'ع': 'a', 'غ': 'gh', 'ف': 'f', 'ق': 'q', 'ک': 'k', 'گ': 'g',
    'ل': 'l', 'م': 'm', 'ن': 'n', 'ں': 'n', 'و': 'o', 'ہ': 'h',
    'ھ': 'h', 'ء': "'", 'ی': 'i', 'ے': 'e', 'ئ': 'i',
    '۔': '.', '،': ',', '؟': '?', '؛': ';',
    'ة': 'a', 'إ': 'i', 'أ': 'a', 'ؤ': 'o',
}


# ============================================
# Translation Functions
# ============================================

def translate_to_urdu(text: str) -> str:
    """
    Translate English text to Pure Urdu using Google Translate.

    Args:
        text: English text to translate.

    Returns:
        Pure Urdu translation.
    """
    if not text or not text.strip():
        return ""

    try:
        translator = GoogleTranslator(source="en", target="ur")
        # Google Translate has a ~5000 char limit per request
        if len(text) > 4500:
            # Split long text and translate in parts
            parts = _split_text(text, max_len=4500)
            translated_parts = []
            for part in parts:
                result = translator.translate(part)
                translated_parts.append(result if result else "")
                time.sleep(RATE_LIMIT_DELAY)
            return " ".join(translated_parts)
        else:
            result = translator.translate(text)
            return result if result else ""
    except Exception as e:
        print(f"  ⚠ Translation error: {e}")
        return ""


def create_mixed_variant(english_text: str, urdu_text: str) -> str:
    """
    Create Urdu-English mixed text by keeping some English words
    within the Urdu translation.

    Strategy:
      1. Identify proper nouns and key English words from original text.
      2. Randomly replace ~20-30% of Urdu words with their English counterparts.

    Args:
        english_text: Original English text.
        urdu_text: Pure Urdu translation.

    Returns:
        Mixed text like: "یہ ایک bili ہے"
    """
    if not urdu_text or not english_text:
        return urdu_text

    # Extract English words to potentially retain
    english_words = english_text.split()
    urdu_words = urdu_text.split()

    if len(urdu_words) <= 2:
        return urdu_text

    # Find proper nouns and important English words
    retain_words = []
    for pattern in RETAIN_ENGLISH_PATTERNS:
        matches = re.findall(pattern, english_text)
        retain_words.extend(matches)

    # Also randomly select some common English words to mix in
    common_english = [w for w in english_words if w.isalpha() and len(w) > 3]
    if common_english:
        num_to_mix = max(1, len(common_english) // 4)  # ~25%
        random_picks = random.sample(
            common_english,
            min(num_to_mix, len(common_english))
        )
        retain_words.extend(random_picks)

    if not retain_words:
        # If no good candidates, replace 1-2 random Urdu words with English ones
        if len(english_words) >= 2 and len(urdu_words) >= 3:
            insert_idx = random.randint(1, len(urdu_words) - 1)
            eng_word = random.choice(english_words)
            urdu_words.insert(insert_idx, eng_word)
            return " ".join(urdu_words)
        return urdu_text

    # Insert retained English words at random positions in the Urdu text
    mixed_words = urdu_words.copy()
    for word in retain_words[:3]:  # Limit to 3 mixed words
        if len(mixed_words) > 2:
            insert_pos = random.randint(1, len(mixed_words) - 1)
            mixed_words.insert(insert_pos, word)

    return " ".join(mixed_words)


def transliterate_to_roman(urdu_text: str) -> str:
    """
    Convert Urdu script text to Roman Urdu.

    Example: "یہ ایک بلی ہے" → "yeh ek bili hai"

    Args:
        urdu_text: Text in Urdu script.

    Returns:
        Roman Urdu text.
    """
    if not urdu_text or not urdu_text.strip():
        return ""

    # Use basic character-level mapping
    roman_chars = []
    for char in urdu_text:
        if char in URDU_TO_ROMAN_MAP:
            roman_chars.append(URDU_TO_ROMAN_MAP[char])
        elif char.isascii():
            roman_chars.append(char)
        elif char == ' ':
            roman_chars.append(' ')
        else:
            # Skip unknown Urdu characters or diacritics
            pass

    result = "".join(roman_chars)

    # Clean up: collapse multiple spaces, strip
    result = re.sub(r'\s+', ' ', result).strip()

    return result


# ============================================
# Helper
# ============================================

def _split_text(text: str, max_len: int = 4500) -> List[str]:
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


# ============================================
# Batch Translation Pipeline
# ============================================

def translate_dataset(
    split: str = "train",
    max_rows: int = None,
    resume: bool = True,
) -> None:
    """
    Translate a mapped dataset (from interim/) to 3 language variants.

    Produces:
      - data/final/{split}_pure_urdu.csv
      - data/final/{split}_mixed.csv
      - data/final/{split}_roman_urdu.csv

    Args:
        split: "train" or "dev"
        max_rows: Max number of rows to translate (None = all).
        resume: If True, resume from last saved progress.
    """
    ensure_dirs()

    # Skip if already completed (all 3 final CSVs exist and no progress file)
    progress_file = FINAL_DIR / f".progress_{split}.txt"
    all_done = all(
        (FINAL_DIR / f"{split}_{v}.csv").exists()
        for v in ["pure_urdu", "mixed", "roman_urdu"]
    )
    if all_done and not progress_file.exists():
        print(f"\n  ✓ {split} already translated — skipping")
        return

    # Load interim mapped data
    input_path = INTERIM_DIR / f"{split}_mapped.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Mapped file not found: {input_path}. "
            "Please run 'python main.py map' first."
        )

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    if max_rows:
        df = df.head(max_rows)

    total = len(df)
    print(f"\n{'=' * 60}")
    print(f"  Translating {split} dataset — {total:,} rows")
    print(f"  Variants: Pure Urdu | Urdu-English Mixed | Roman Urdu")
    print(f"{'=' * 60}\n")

    # Resume support
    progress_file = FINAL_DIR / f".progress_{split}.txt"
    start_idx = load_progress(progress_file) if resume else 0

    if start_idx > 0:
        print(f"  ↻ Resuming from row {start_idx:,}")

    # Initialize output columns if starting fresh
    if start_idx == 0:
        df["claim_urdu"] = ""
        df["evidence_urdu"] = ""
        df["claim_mixed"] = ""
        df["evidence_mixed"] = ""
        df["claim_roman"] = ""
        df["evidence_roman"] = ""
    else:
        # Load partially translated data
        partial_path = FINAL_DIR / f".partial_{split}.csv"
        if partial_path.exists():
            df = pd.read_csv(partial_path, encoding="utf-8-sig")
            if max_rows:
                df = df.head(max_rows)

    # Translate row by row
    for idx in tqdm(range(start_idx, total), desc=f"Translating {split}", unit="row"):
        row = df.iloc[idx]
        claim_en = str(row["claim"])
        evidence_en = str(row["evidence"])

        # 1. Pure Urdu
        claim_urdu = translate_to_urdu(claim_en)
        evidence_urdu = translate_to_urdu(evidence_en) if evidence_en else ""
        time.sleep(RATE_LIMIT_DELAY)

        # 2. Urdu-English Mixed
        claim_mixed = create_mixed_variant(claim_en, claim_urdu)
        evidence_mixed = create_mixed_variant(evidence_en, evidence_urdu) if evidence_en else ""

        # 3. Roman Urdu
        claim_roman = transliterate_to_roman(claim_urdu)
        evidence_roman = transliterate_to_roman(evidence_urdu) if evidence_urdu else ""

        # Store results
        df.at[idx, "claim_urdu"] = claim_urdu
        df.at[idx, "evidence_urdu"] = evidence_urdu
        df.at[idx, "claim_mixed"] = claim_mixed
        df.at[idx, "evidence_mixed"] = evidence_mixed
        df.at[idx, "claim_roman"] = claim_roman
        df.at[idx, "evidence_roman"] = evidence_roman

        # Periodic save
        if (idx + 1) % BATCH_SAVE_INTERVAL == 0:
            save_progress(progress_file, idx + 1)
            df.to_csv(FINAL_DIR / f".partial_{split}.csv", index=False, encoding="utf-8-sig")
            print(f"  💾 Progress saved at row {idx + 1:,}")

    # Save final outputs
    _save_final_csvs(df, split)

    # Clean up progress files
    for f in [progress_file, FINAL_DIR / f".partial_{split}.csv"]:
        if f.exists():
            f.unlink()

    print(f"\n✅ Translation complete for {split}!\n")


def _save_final_csvs(df: pd.DataFrame, split: str) -> None:
    """Save the translated DataFrame into 3 separate CSV files."""

    # Pure Urdu
    df_urdu = df[["claim_urdu", "evidence_urdu", "label", "label_text"]].copy()
    df_urdu.columns = ["claim", "evidence", "label", "label_text"]
    urdu_path = FINAL_DIR / f"{split}_pure_urdu.csv"
    df_urdu.to_csv(urdu_path, index=False, encoding="utf-8-sig")
    print(f"  ✓ Saved: {urdu_path.name} ({len(df_urdu):,} rows)")

    # Urdu-English Mixed
    df_mixed = df[["claim_mixed", "evidence_mixed", "label", "label_text"]].copy()
    df_mixed.columns = ["claim", "evidence", "label", "label_text"]
    mixed_path = FINAL_DIR / f"{split}_mixed.csv"
    df_mixed.to_csv(mixed_path, index=False, encoding="utf-8-sig")
    print(f"  ✓ Saved: {mixed_path.name} ({len(df_mixed):,} rows)")

    # Roman Urdu
    df_roman = df[["claim_roman", "evidence_roman", "label", "label_text"]].copy()
    df_roman.columns = ["claim", "evidence", "label", "label_text"]
    roman_path = FINAL_DIR / f"{split}_roman_urdu.csv"
    df_roman.to_csv(roman_path, index=False, encoding="utf-8-sig")
    print(f"  ✓ Saved: {roman_path.name} ({len(df_roman):,} rows)")


def translate_all(max_rows: int = None) -> None:
    """Translate train, dev, and test splits."""
    for split in ["train", "dev", "test"]:
        translate_dataset(split=split, max_rows=max_rows)
