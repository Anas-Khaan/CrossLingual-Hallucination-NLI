"""
Utility Functions
=================
JSONL I/O, file downloads with progress bars, text cleaning, and chunking helpers.
"""

import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import requests
from tqdm import tqdm


# ============================================
# Constants
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
WIKI_DIR = DATA_DIR / "wiki_dump"
INTERIM_DIR = DATA_DIR / "interim"
FINAL_DIR = DATA_DIR / "final"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"

# FEVER Dataset URLs (official — fever.ai)
FEVER_URLS = {
    "train": "https://fever.ai/download/fever/train.jsonl",
    "dev": "https://fever.ai/download/fever/shared_task_dev.jsonl",
    "wiki": "https://fever.ai/download/fever/wiki-pages.zip",
}

# Label mapping
LABEL_MAP = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT ENOUGH INFO": 2,
}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}
FRIENDLY_LABELS = {
    0: "Supported",
    1: "Not Supported",
    2: "Not Enough Info",
}


# ============================================
# Directory Setup
# ============================================

def ensure_dirs() -> None:
    """Create all required project directories if they don't exist."""
    for d in [RAW_DIR, WIKI_DIR, INTERIM_DIR, FINAL_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ============================================
# Download Helpers
# ============================================

def download_file(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    """
    Download a file from `url` to `dest` with a progress bar.
    Skips download if the file already exists.

    Args:
        url: URL to download from.
        dest: Destination file path.
        chunk_size: Bytes per chunk for streaming download.

    Returns:
        Path to the downloaded file.
    """
    if dest.exists():
        print(f"  ✓ Already downloaded: {dest.name}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ Downloading: {url}")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=dest.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"  ✓ Saved: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def unzip_file(zip_path: Path, extract_to: Path) -> None:
    """
    Unzip a .zip file to the specified directory.
    Skips if the directory already has contents.
    """
    if extract_to.exists() and any(extract_to.iterdir()):
        print(f"  ✓ Already extracted: {extract_to.name}/")
        return

    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ Extracting: {zip_path.name} → {extract_to.name}/")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc="Extracting", unit="file"):
            zf.extract(member, extract_to)

    print(f"  ✓ Extracted {len(members)} files to {extract_to.name}/")


def download_fever_dataset() -> None:
    """
    Download all FEVER dataset files:
      - train.jsonl       → data/raw/
      - shared_task_dev.jsonl → data/raw/
      - wiki-pages.zip    → data/raw/ (then unzipped to data/wiki_dump/)
    """
    ensure_dirs()
    print("\n" + "=" * 60)
    print("  FEVER Dataset Download")
    print("=" * 60)

    # Download train and dev JSONL
    for name, url in FEVER_URLS.items():
        if name == "wiki":
            dest = RAW_DIR / "wiki-pages.zip"
        else:
            filename = url.split("/")[-1]
            dest = RAW_DIR / filename
        download_file(url, dest)

    # Unzip wiki pages
    wiki_zip = RAW_DIR / "wiki-pages.zip"
    if wiki_zip.exists():
        unzip_file(wiki_zip, WIKI_DIR)

    print("\n✅ FEVER dataset download complete!\n")


# ============================================
# JSONL I/O
# ============================================

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load all records from a JSONL file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_jsonl_lazy(path: Path) -> Generator[Dict[str, Any], None, None]:
    """Lazily load records from a JSONL file (memory-efficient for large files)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_jsonl(data: List[Dict[str, Any]], path: Path) -> None:
    """Save a list of dicts to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  ✓ Saved {len(data)} records → {path.name}")


# ============================================
# Text Cleaning
# ============================================

def clean_text(text: str) -> str:
    """
    Clean and normalize text:
      - Strip leading/trailing whitespace
      - Collapse multiple spaces into one
      - Remove HTML tags
      - Normalize common Unicode characters
    """
    if not text:
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Replace special tokens like -LRB-, -RRB- (used in FEVER wiki dump)
    text = text.replace("-LRB-", "(")
    text = text.replace("-RRB-", ")")
    text = text.replace("-LSB-", "[")
    text = text.replace("-RSB-", "]")
    text = text.replace("-COLON-", ":")

    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================
# Chunking Utilities (for long texts)
# ============================================

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using basic punctuation rules.
    Handles both English and Urdu sentence boundaries.
    """
    if not text:
        return []

    # Split on common sentence-ending punctuation
    # Includes Urdu full stop (۔) and Arabic question mark (؟)
    sentences = re.split(r'(?<=[.!?۔؟])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, max_tokens: int = 400, overlap: int = 50) -> List[str]:
    """
    Split long text into overlapping chunks for processing.
    Uses word-level splitting as a proxy for token count.

    Args:
        text: Input text to chunk.
        max_tokens: Approximate max words per chunk.
        overlap: Number of overlapping words between chunks.

    Returns:
        List of text chunks.
    """
    words = text.split()
    if len(words) <= max_tokens:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + max_tokens
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap

    return chunks


# ============================================
# Progress Tracking (for resumable operations)
# ============================================

def load_progress(progress_file: Path) -> int:
    """Load the last processed index from a progress file."""
    if progress_file.exists():
        with open(progress_file, "r") as f:
            return int(f.read().strip())
    return 0


def save_progress(progress_file: Path, index: int) -> None:
    """Save the current progress index to a file."""
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, "w") as f:
        f.write(str(index))
