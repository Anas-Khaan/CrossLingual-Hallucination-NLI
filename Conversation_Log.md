# FEVER Urdu Hallucination Detector — Conversation Log

**Date:** April 24, 2026  
**Project:** 3rd Semester NLP — Hallucination Detection for Urdu  
**Environment:** Windows, Python 3.13.9 venv, PyTorch 2.11.0+cu126, NVIDIA RTX 4090 (25.8 GB VRAM)

---

## Table of Contents

1. [Project Overview & Status](#1-project-overview--status)
2. [Environment Setup](#2-environment-setup)
3. [GPU & CUDA Verification](#3-gpu--cuda-verification)
4. [Understanding the Pipeline Flow](#4-understanding-the-pipeline-flow)
5. [What is the Mapping Step?](#5-what-is-the-mapping-step)
6. [Train / Dev / Test Split](#6-train--dev--test-split)
7. [Understanding the Data Folder](#7-understanding-the-data-folder)
8. [Understanding the Models Folder & Checkpoints](#8-understanding-the-models-folder--checkpoints)
9. [Understanding Every File in the Project](#9-understanding-every-file-in-the-project)
10. [Complete Execution Flow](#10-complete-execution-flow)
11. [Code Changes Made During This Session](#11-code-changes-made-during-this-session)

---

## 1. Project Overview & Status

### 🎯 Ultimate Goal

Build a **hallucination detection system** for **Urdu and Urdu-English mixed language text**. Given a **source text** (evidence) and a **claim** (generated text), the system predicts whether the claim is:

| Label | Meaning |
|---|---|
| **Supported** | The claim is backed by the source |
| **Not Supported** | The claim contradicts the source |
| **Not Enough Info** | The source doesn't have enough info to verify |

It works across **3 language variants**:

| Variant | Example |
|---|---|
| **Pure Urdu** | یہ ایک بلی ہے |
| **Urdu-English Mixed** | یہ ایک bili ہے |
| **Roman Urdu** | yeh ak bili hai |

### Architecture

- **Model**: XLM-RoBERTa-Large (560M params, supports 100+ languages)
- **Task**: Natural Language Inference (NLI) — 3-class classification
- **Training data**: FEVER dataset (English) → translated into 3 Urdu variants
- **Hardware**: NVIDIA RTX 4090 (24GB VRAM), FP16 mixed precision
- **Framework**: HuggingFace Transformers + Trainer API

### Status at Start of This Session

| What | Status |
|---|---|
| Complete codebase | ✅ All 9 source modules fully written |
| Virtual environment | ✅ Created (Python venv) |
| All dependencies | ✅ Installed (PyTorch+CUDA, HuggingFace, etc.) |
| CLI pipeline | ✅ Designed with `main.py` |
| Data download | ❌ Not done (all data folders empty) |
| Mapping | ❌ Not done |
| Translation | ❌ Not done |
| Training | ❌ Not done |
| Evaluation | ❌ Not done |

**Summary**: The architecture and code are complete; execution hasn't started.

---

## 2. Environment Setup

### Virtual Environment

The project uses a **Python venv** (not conda). The venv is located at:

```
Hallucination_Detector/Hallucination_detector/
```

**Activation:**
```powershell
.\Hallucination_detector\Scripts\Activate.ps1
```

You'll see `(Hallucination_detector)` appear in your prompt.

> **Note:** This is a Python venv, NOT a conda environment. `conda activate` will not work.

### Installed Dependencies

| Category | Packages | Version |
|---|---|---|
| **PyTorch + CUDA** | `torch`, `torchvision`, `torchaudio` | `2.11.0+cu126` |
| **HuggingFace** | `transformers`, `datasets`, `accelerate`, `tokenizers` | `5.5.4`, `4.8.4`, `1.13.0` |
| **Translation** | `deep-translator`, `indic_transliteration` | `1.11.4`, `2.3.82` |
| **Data** | `pandas`, `numpy`, `tqdm` | `3.0.2`, `2.4.3`, `4.67.3` |
| **ML/Metrics** | `scikit-learn` | `1.8.0` |
| **Visualization** | `matplotlib`, `seaborn` | `3.10.8`, `0.13.2` |
| **Utilities** | `python-dotenv`, `requests` | `1.2.2`, `2.33.1` |
| **Notebooks** | `jupyter`, `jupyterlab`, `ipykernel` | Installed |

---

## 3. GPU & CUDA Verification

### How PyTorch + CUDA Works

1. **PyTorch is installed** in the venv as `torch 2.11.0+cu126` — the `+cu126` means it was built for CUDA 12.6
2. **You do NOT install CUDA separately** — PyTorch's pip package **bundles the CUDA runtime libraries** inside itself
3. **What you DO need** at the system level is just the **NVIDIA GPU driver** (already installed on the PC)

### The Chain

```
Your Python venv (pip packages)     System level (OS)
┌─────────────────────────┐         ┌──────────────────┐
│  PyTorch + CUDA runtime │ ──────► │  NVIDIA GPU Driver│ ──────► GPU Hardware
│  (bundled inside torch) │         │  (already on PC)  │         (RTX 4090)
└─────────────────────────┘         └──────────────────┘
```

### Verification Result

```
PyTorch:       2.11.0+cu126
CUDA available: True
CUDA version:  12.6
GPU count:     1
GPU 0:         NVIDIA GeForce RTX 4090
VRAM:          25.8 GB
```

✅ Everything connected and working.

### Common Questions

| Question | Answer |
|---|---|
| How is PyTorch installed? | Via `pip install torch --index-url .../cu126` — a special CUDA-enabled build |
| Can I install CUDA in a Python venv? | You don't need to — PyTorch bundles CUDA runtime inside the pip package |
| How does it connect to the GPU? | PyTorch's bundled CUDA talks to the NVIDIA driver (system-wide), which controls the GPU |

---

## 4. Understanding the Pipeline Flow

### The 6-Step Pipeline

```
Step 1: DOWNLOAD          Step 2: MAP                Step 3: TRANSLATE
┌──────────────┐      ┌──────────────────┐      ┌──────────────────────┐
│ FEVER Dataset│      │ Claims + Wiki    │      │ English → 3 variants │
│ (English)    │─────►│ Dump → Extract   │─────►│ • Pure Urdu          │
│ • train.jsonl│      │ evidence for     │      │ • Urdu-English Mixed │
│ • dev.jsonl  │      │ each claim       │      │ • Roman Urdu         │
│ • wiki dump  │      │                  │      │                      │
│   (~2GB)     │      │ Output: CSV with │      │ Output: 9 CSV files  │
└──────────────┘      │ (claim,evidence, │      │ (3 splits × 3)       │
                      │  label)          │      └──────────┬───────────┘
                      └──────────────────┘                 │
                                                           ▼
Step 6: INFER             Step 5: EVALUATE         Step 4: TRAIN
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│ Give source+claim│    │ Test on test set  │    │ Fine-tune            │
│ Get: Supported / │◄───│ Accuracy, F1,    │◄───│ XLM-RoBERTa-Large   │
│ Not Supported /  │    │ Confusion Matrix │    │ on translated data   │
│ Not Enough Info  │    │ Error Analysis   │    │ (FP16, RTX 4090)     │
└──────────────────┘    └──────────────────┘    └──────────────────────┘
```

### Commands

```bash
python main.py download                         # Step 1
python main.py map --max-claims 50000           # Step 2
python main.py translate --max-rows 50000       # Step 3
python main.py train --epochs 4 --batch-size 16 # Step 4
python main.py evaluate                         # Step 5
python main.py infer                            # Step 6
```

---

## 5. What is the Mapping Step?

The FEVER dataset comes in **two separate pieces** that don't directly connect:

**Piece 1: Claims file** (`train.jsonl`) — has claims with **pointers** to evidence, not actual text:

```json
{
  "claim": "Barack Obama was born in Hawaii",
  "label": "SUPPORTS",
  "evidence": [
    [null, null, "Barack_Obama", 0],
    [null, null, "Barack_Obama", 3]
  ]
}
```

The evidence field says: *"go to page Barack_Obama, get sentence #0 and #3"*

**Piece 2: Wiki dump** (thousands of `.jsonl` files) — has the actual Wikipedia text:

```json
{
  "id": "Barack_Obama",
  "lines": "0\tBarack Obama is an American politician born in Honolulu, Hawaii.\n1\tHe served as the 44th president.\n3\tObama was born on August 4, 1961 in Hawaii."
}
```

### The Mapping Step connects them:

```
BEFORE mapping:                          AFTER mapping (clean CSV):
┌────────────────────────┐               ┌──────────┬───────────────────────────┬─────────┐
│ claim: "Obama born     │               │ claim    │ evidence                  │ label   │
│         in Hawaii"     │      MAP      ├──────────┼───────────────────────────┼─────────┤
│ evidence: go to page   │ ───────────►  │ "Obama   │ "Barack Obama is an      │SUPPORTS │
│   "Barack_Obama",      │               │  born in │  American politician     │         │
│   sentence 0 and 3     │               │  Hawaii" │  born in Honolulu,       │         │
│ label: SUPPORTS        │               │          │  Hawaii. Obama was born  │         │
└────────────────────────┘               │          │  on Aug 4, 1961..."      │         │
                                         └──────────┴───────────────────────────┴─────────┘
```

**In simple words:**

1. Read the claim → "Barack Obama was born in Hawaii"
2. Follow the pointer → go to wiki page `Barack_Obama`, get sentence `#0` and `#3`
3. Extract the actual text → "Barack Obama is an American politician born in Honolulu, Hawaii. Obama was born on August 4, 1961 in Hawaii."
4. Save as a clean row: `(claim, evidence_text, label)`

Without mapping, you'd just have address-like pointers — you can't train a model on pointers. You need the actual text.

---

## 6. Train / Dev / Test Split

### The Problem

The FEVER shared task keeps **test set labels hidden** (for competition fairness). So we only have:

- `train.jsonl` → 145K claims (with labels)
- `shared_task_dev.jsonl` → 19K claims (with labels)
- `test.jsonl` → No labels available ❌

### The Solution

We **split the dev set** into two parts:

```
┌─────────────────────────────────────────────────┐
│              FEVER dev set (19K)                 │
│                                                 │
│   ┌──────────────────┐  ┌────────────────────┐  │
│   │  Dev/Validation   │  │   Test (held-out)  │  │
│   │  (80% = ~15K)     │  │   (20% = ~4K)      │  │
│   │                   │  │                    │  │
│   │  • Early stopping │  │  • Final metrics   │  │
│   │  • Best checkpoint│  │  • Report results  │  │
│   └──────────────────┘  └────────────────────┘  │
└─────────────────────────────────────────────────┘
```

| Split | Source | Purpose | Used When |
|---|---|---|---|
| **Train** | `train.jsonl` | Learn weights | During training |
| **Dev** | 80% of `dev.jsonl` | Validation, early stopping | After each epoch |
| **Test** | 20% of `dev.jsonl` | Final evaluation & reporting | After training is done |

### Key Points

- The split is **stratified** (same label distribution in dev and test)
- Uses `random_state=42` for reproducibility
- The model **never sees** the test set during training

### Updated Pipeline Flow

```
train.jsonl ──map──► train_mapped.csv ──translate──► train_*.csv ──► TRAIN
                                                                        │
dev.jsonl ──map──► dev_full ──80/20 split──┐                           │
                                            ├──► dev_mapped.csv ──translate──► dev_*.csv ──► VALIDATE
                                            └──► test_mapped.csv ──translate──► test_*.csv ──► FINAL EVAL
```

---

## 7. Understanding the Data Folder

Each subfolder holds data at a different stage of the pipeline:

```
data/
├── raw/          ← Step 1: Downloaded (untouched from internet)
├── wiki_dump/    ← Step 1: Unzipped Wikipedia pages
├── interim/      ← Step 2: Mapped (English, cleaned, connected)
└── final/        ← Step 3: Translated (Urdu variants, ready for training)
```

### `raw/` — Raw downloaded files (untouched)

Straight from the internet, exactly as FEVER provides them.

```
raw/
├── train.jsonl            ← 145K claims with evidence pointers
├── shared_task_dev.jsonl  ← 19K claims with evidence pointers
└── wiki-pages.zip         ← Compressed Wikipedia dump (~2GB)
```

### `wiki_dump/` — Unzipped Wikipedia pages

The `wiki-pages.zip` gets extracted here. Contains thousands of small files.

```
wiki_dump/
└── wiki-pages/
    ├── wiki-001.jsonl    ← hundreds of Wikipedia articles
    ├── wiki-002.jsonl
    ├── ...
    └── wiki-109.jsonl    ← ~5.4 million Wikipedia pages total
```

### `interim/` — After mapping (English, clean CSVs)

The mapping step reads `raw/` + `wiki_dump/`, connects the pointers to actual text.

```
interim/
├── train_mapped.csv    ← (claim, evidence_text, label) — English
├── dev_mapped.csv      ← 80% of FEVER dev set — for validation
└── test_mapped.csv     ← 20% of FEVER dev set — held-out for testing
```

Each CSV looks like:

| claim | evidence | label | label_text |
|---|---|---|---|
| Obama was born in Hawaii | Barack Obama is an American politician born in Honolulu... | 0 | SUPPORTS |

### `final/` — After translation (ready for training)

The translation step translates each CSV into 3 Urdu variants. This is what the model trains on.

```
final/
├── train_pure_urdu.csv      ← Pure Urdu
├── train_mixed.csv          ← Urdu-English mixed
├── train_roman_urdu.csv     ← Roman Urdu
├── dev_pure_urdu.csv
├── dev_mixed.csv
├── dev_roman_urdu.csv
├── test_pure_urdu.csv
├── test_mixed.csv
└── test_roman_urdu.csv      ← 9 files total (3 splits × 3 variants)
```

### Visual Flow

```
DOWNLOAD          UNZIP              MAP                TRANSLATE
   │                │                  │                    │
   ▼                ▼                  ▼                    ▼
┌───────┐     ┌──────────┐      ┌──────────┐         ┌─────────┐
│ raw/  │     │wiki_dump/│      │ interim/ │         │ final/  │
│       │     │          │      │          │         │         │
│.jsonl │────►│ wiki     │─────►│ English  │────────►│ 3 Urdu  │───► MODEL
│.zip   │     │ pages    │      │ CSVs     │         │ variant │    TRAINING
│       │     │ (text)   │      │(mapped)  │         │ CSVs    │
└───────┘     └──────────┘      └──────────┘         └─────────┘
```

---

## 8. Understanding the Models Folder & Checkpoints

This is where the **trained model weights** get saved after training (Step 4). Currently empty.

After training, it will look like:

```
models/
└── checkpoints/
    └── xlm-roberta-hallucination/
        ├── checkpoint-1000/          ← saved after 1000 steps
        ├── checkpoint-2000/          ← saved after 2000 steps
        ├── checkpoint-3000/          ← saved after 3000 steps
        └── best_model/              ← ⭐ THE BEST one (used for inference)
            ├── model.safetensors     ← final trained weights (~2.2 GB)
            ├── config.json           ← model architecture config
            ├── tokenizer.json        ← tokenizer
            └── special_tokens_map.json
```

### What are Checkpoints?

Think of them as **save points in a video game**:

| Concept | Analogy |
|---|---|
| **Checkpoint** | A save point — snapshot of the model at a specific moment during training |
| **Why multiple?** | If training crashes at hour 5, you can resume from the last checkpoint instead of starting over |
| **`best_model/`** | The checkpoint with the **highest F1 score** on the dev set — this is the one you actually use |

### How it works during training:

```
Epoch 1 → evaluate on dev → F1: 0.72 → save checkpoint-1
Epoch 2 → evaluate on dev → F1: 0.81 → save checkpoint-2 ⭐ (best so far)
Epoch 3 → evaluate on dev → F1: 0.83 → save checkpoint-3 ⭐ (new best!)
Epoch 4 → evaluate on dev → F1: 0.82 → save checkpoint-4 (worse → early stopping)

                                         → copy checkpoint-3 → best_model/ ✅
```

---

## 9. Understanding Every File in the Project

### Notebooks (optional, for exploration)

| Notebook | Purpose |
|---|---|
| `01_data_mapping.ipynb` | Interactive walkthrough of how evidence mapping works — load a few claims, show pointers, show extracted text |
| `02_translation_test.ipynb` | Test translation quality — translate a few sentences, compare all 3 variants side by side |
| `03_eda.ipynb` | Exploratory Data Analysis — label distribution, text length stats, class balance charts |

> **Note:** Notebooks are NOT part of the pipeline. The pipeline runs entirely through `main.py`.

### `src/` — Source Modules

#### `__init__.py`
Makes `src/` a Python package. Contains project version (`0.1.0`). No logic.

#### `utils.py` — Foundation (used by EVERY other module)
Shared constants, helpers, and I/O functions:
- All folder paths (`DATA_DIR`, `RAW_DIR`, `INTERIM_DIR`, etc.)
- FEVER download URLs
- Label mappings (`SUPPORTS→0`, `REFUTES→1`, `NOT ENOUGH INFO→2`)
- `download_fever_dataset()` — download all FEVER files
- `load_jsonl()` / `save_jsonl()` — read/write JSONL files
- `clean_text()` — remove HTML, fix special tokens
- `split_into_sentences()`, `chunk_text()` — text splitting
- `load_progress()` / `save_progress()` — resume support

#### `mapping.py` — Evidence extraction (Step 2)
- `build_wiki_lookup()` — Read all wiki dump files into a dictionary: `{page_title: {sentence_id: text}}`
- `extract_evidence_text()` — For one claim, follow its pointers to get actual text
- `map_claims_to_evidence()` — Process all claims in a split
- `map_all()` — Map train + dev, then split dev into dev (80%) + test (20%)

**Input:** `data/raw/*.jsonl` + `data/wiki_dump/`  
**Output:** `data/interim/*.csv`

#### `translator.py` — 3-variant translation (Step 3)
- `translate_to_urdu(text)` — English → Pure Urdu via Google Translate
- `create_mixed_variant(eng, urdu)` — Insert English words into Urdu text (~25% mixing)
- `transliterate_to_roman(urdu)` — Urdu script → Roman letters
- `translate_dataset(split)` — Translate entire CSV with rate limiting + resume support
- `translate_all()` — Translate all 3 splits

**Input:** `data/interim/*.csv`  
**Output:** `data/final/*.csv` (9 files)

#### `preprocess.py` — Tokenization & DataLoaders (Step 4 prep)
- `HallucinationDataset` — PyTorch Dataset: tokenizes as `[CLS] evidence [SEP] claim [SEP]`
- `load_variant_csv()` — Load one CSV file
- `load_combined_dataset()` — Load & merge all 3 variants, shuffled
- `create_dataloaders()` — Create train + dev DataLoaders

**Input:** CSVs from `data/final/`  
**Output:** PyTorch DataLoaders

#### `model.py` — Model definition (Step 4)
- `create_model()` — Load XLM-RoBERTa-Large with a 3-class classification head (or from checkpoint)
- `get_model_info()` — Return model summary

**Size:** ~560M parameters, ~2.2GB on disk

#### `train.py` — Training loop (Step 4)
- `compute_metrics()` — Calculate accuracy, macro F1, per-class F1
- `train_model()` — Fine-tune with HuggingFace Trainer:
  - Batch size 16, LR 2e-5, FP16, AdamW
  - Early stopping (patience 2)
  - Save best model based on macro F1

**Output:** `models/checkpoints/xlm-roberta-hallucination/best_model/`

#### `evaluate.py` — Evaluation (Step 5)
- `evaluate_model()` — Run on test set (combined + per-variant)
- `_plot_confusion_matrix()` — Heatmap visualization
- `_error_analysis()` — Top misclassified examples + error CSV

**Output:** Confusion matrix plots, error CSVs, printed metrics

#### `inference.py` — Live predictions (Step 6)
- `HallucinationDetector` class with 3 modes:
  - `predict(source, claim)` — Single sentence
  - `predict_paragraph(source, claim)` — Split into sentences → majority vote
  - `predict_document(source, claim)` — Chunk long texts → cross-check → majority vote

### Root Files

#### `main.py` — CLI Orchestrator
The single entry point. Each command imports only the module it needs:

| Command | Calls |
|---|---|
| `python main.py gpu` | `check_gpu()` |
| `python main.py download` | `src.utils.download_fever_dataset()` |
| `python main.py map` | `src.mapping.map_all()` |
| `python main.py translate` | `src.translator.translate_all()` |
| `python main.py train` | `src.train.train_model()` |
| `python main.py evaluate` | `src.evaluate.evaluate_model()` |
| `python main.py infer` | `src.inference.HallucinationDetector()` |

#### `verify_install.py` — Installation checker
Tries to import every required package, prints OK/FAIL for each, checks GPU.

---

## 10. Complete Execution Flow

```
YOU RUN                    WHAT HAPPENS                           FILES INVOLVED
═══════                    ════════════                           ══════════════

Step 0:
python verify_install.py   → Check all packages + GPU             verify_install.py

Step 1:
python main.py download    → Downloads train.jsonl, dev.jsonl      main.py → utils.py
                           → Downloads + unzips wiki dump
                           → Output: data/raw/ + data/wiki_dump/

Step 2:
python main.py map         → Builds wiki lookup dictionary         main.py → mapping.py → utils.py
                           → Maps each claim to evidence text
                           → Splits dev into dev(80%) + test(20%)
                           → Output: data/interim/*.csv

Step 3:
python main.py translate   → Translates each row to 3 variants    main.py → translator.py → utils.py
                           → Rate limiting + resume support
                           → Output: data/final/*.csv (9 files)

Step 4:
python main.py train       → Loads data + tokenizes                main.py → train.py
                           → Creates XLM-RoBERTa model                     → preprocess.py
                           → Trains 4 epochs with FP16                     → model.py
                           → Validates on dev after each epoch             → utils.py
                           → Saves best checkpoint
                           → Output: models/checkpoints/best_model/

Step 5:
python main.py evaluate    → Loads best model                     main.py → evaluate.py
                           → Runs on held-out test set                     → preprocess.py
                           → Prints metrics + confusion matrix             → utils.py
                           → Output: eval_results/

Step 6:
python main.py infer       → Loads best model                     main.py → inference.py → utils.py
                           → Interactive: type source + claim
                           → Output: label + confidence score
```

### Module Dependency

```
main.py (entry point)
  ├── utils.py ← foundation (paths, I/O, constants) — used by ALL modules
  ├── mapping.py ← uses utils.py
  ├── translator.py ← uses utils.py
  ├── train.py ← uses preprocess.py, model.py, utils.py
  ├── evaluate.py ← uses preprocess.py, utils.py
  └── inference.py ← uses utils.py
      preprocess.py ← uses utils.py
      model.py ← uses utils.py
```

---

## 11. Code Changes Made During This Session

### Change 1: Added Train/Dev/Test Split

**Files modified:**

| File | Change |
|---|---|
| `mapping.py` | `map_all()` now splits dev into dev (80%) + test (20%) with stratified sampling |
| `translator.py` | `translate_all()` now translates 3 splits (train, dev, test) instead of 2 |
| `evaluate.py` | Default split changed from `"dev"` → `"test"` |
| `main.py` | CLI `--split` default updated to `"test"` |
| `README.md` | Updated documentation to reflect the new split |

**Why:** The FEVER test set labels are not publicly available, so we created our own held-out test set from the dev split. This ensures the model is never evaluated on data it was validated against during training.

---

*End of Conversation Log*
