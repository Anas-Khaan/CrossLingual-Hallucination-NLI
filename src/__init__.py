"""
FEVER Urdu Hallucination Detector — Source Package
===================================================
Modules:
    - utils.py        : JSONL I/O, download helpers, text cleaning
    - mapping.py      : Parse wiki dump & map evidence to claims
    - translator.py   : 3-variant translation (Pure Urdu, Mixed, Roman Urdu)
    - preprocess.py   : NLI format conversion, tokenization, DataLoaders
    - model.py        : XLM-RoBERTa + 3-class classification head
    - train.py        : Training loop with HF Trainer
    - evaluate.py     : Metrics, confusion matrix, error analysis
    - inference.py    : Source + claim → verdict + confidence
"""

__version__ = "0.1.0"
