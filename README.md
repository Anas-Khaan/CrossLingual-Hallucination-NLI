# FEVER Urdu Hallucination Detector

A hallucination detection system trained on the [FEVER dataset](https://fever.ai/) that verifies whether a **claim** is supported by a **source text**. Supports **three language variants**:

| Variant | Example |
|---------|---------|
| **Pure Urdu** | یہ ایک بلی ہے |
| **Urdu-English Mixed** | یہ ایک bili ہے |
| **Roman Urdu** | yeh ak bili hai |

## Architecture

- **Model**: XLM-RoBERTa-Large (560M params, supports 100+ languages)
- **Task**: Natural Language Inference (NLI) — 3-class classification
- **Labels**: Supported · Not Supported · Not Enough Info
- **Hardware**: Optimized for NVIDIA RTX 4090 (24GB VRAM)

## Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment
.\Hallucination_detector\Scripts\activate

# Install PyTorch with CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python main.py gpu
```

### 2. Run the Pipeline

```bash
# Step 1: Download FEVER dataset (~2GB)
python main.py download

# Step 2: Map evidence to claims + split dev into dev/test
python main.py map --max-claims 50000

# Step 3: Translate all 3 splits to 3 language variants
python main.py translate --max-rows 50000

# Step 4: Train the model (validates on dev set)
python main.py train --epochs 4 --batch-size 16

# Step 5: Evaluate on held-out test set
python main.py evaluate

# Step 6: Interactive inference
python main.py infer
```

### 3. Single Prediction

```bash
python main.py infer --source "یہ ایک بلی ہے" --claim "یہ ایک کتا ہے"
```

## Project Structure

```
FEVER_Urdu_Hallucination/
├── data/
│   ├── raw/                       # Auto-downloaded FEVER .jsonl files
│   ├── wiki_dump/                 # Wikipedia pages (unzipped)
│   ├── interim/                   # Mapped CSVs: train, dev (80%), test (20%)
│   └── final/                     # Translated CSVs (3 variants × train/dev/test)
├── notebooks/
│   ├── 01_data_mapping.ipynb      # Evidence extraction walkthrough
│   ├── 02_translation_test.ipynb  # Translation quality testing
│   └── 03_eda.ipynb               # Dataset statistics & EDA
├── src/
│   ├── utils.py                   # Download, JSONL I/O, text cleaning
│   ├── mapping.py                 # Wiki dump → evidence mapping
│   ├── translator.py              # 3-variant translation pipeline
│   ├── preprocess.py              # NLI format, tokenization, DataLoaders
│   ├── model.py                   # XLM-RoBERTa model definition
│   ├── train.py                   # Training with HF Trainer (FP16)
│   ├── evaluate.py                # Metrics, confusion matrix, error analysis
│   └── inference.py               # Inference engine (sentence/paragraph/doc)
├── models/checkpoints/            # Saved model weights
├── main.py                        # CLI pipeline orchestration
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Dataset

- **Source**: [FEVER Shared Task](https://fever.ai/)
- **Training**: ~50K claims (subset of 145K)
- **Validation (dev)**: 80% of FEVER dev set (~15K) — used during training
- **Test (held-out)**: 20% of FEVER dev set (~4K) — used for final evaluation
- **License**: Creative Commons Attribution-ShareAlike

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | xlm-roberta-large |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Epochs | 4 |
| Max sequence length | 512 |
| Mixed precision | FP16 |
| Optimizer | AdamW |
| Early stopping | Patience 2 |

## License

This project uses the FEVER dataset under the Creative Commons Attribution-ShareAlike License.
