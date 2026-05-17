# Pure Urdu Hallucination Detector — Evaluation Results

## Model Details
- **Model:** XLM-RoBERTa-Large (fine-tuned)
- **Training Data:** 12,000 Pure Urdu samples from FEVER dataset
- **Validation Data:** 4,000 Pure Urdu samples from FEVER dataset
- **Training:** 4 epochs, batch_size=16, lr=2e-5, FP16, class-weighted loss
- **GPU:** NVIDIA GeForce RTX 4090 (24 GB VRAM)
- **Training Time:** ~13.5 minutes

---

## 1. In-Domain Evaluation — Urdu-FEVER Test Set (4,000 samples)

| Metric | Value |
|---|---|
| **Classification Accuracy** | **92.70%** |
| **Macro Precision** | **0.9241** |
| **Macro Recall** | **0.9249** |
| **Macro F1-Score** | **0.9245** |
| **Matthews Corr. (MCC)** | **0.8843** |

### Per-Class Breakdown (FEVER)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Supported | 0.9265 | 0.9241 | 0.9253 | 1,950 |
| Not Supported | 0.8568 | 0.8506 | 0.8537 | 964 |
| Not Enough Info | 0.9891 | 1.0000 | 0.9945 | 1,086 |

---

## 2. Cross-Domain Zero-Shot Evaluation — XNLI Pure Urdu (372,316 samples)

| Metric | Value |
|---|---|
| **Classification Accuracy** | **40.70%** |
| **Macro Precision** | **0.3882** |
| **Macro Recall** | **0.4069** |
| **Macro F1-Score** | **0.3645** |
| **Matthews Corr. (MCC)** | **0.1229** |

### Per-Class Breakdown (XNLI)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Supported | 0.3816 | 0.6913 | 0.4917 | 124,093 |
| Not Supported | 0.5017 | 0.4443 | 0.4713 | 124,287 |
| Not Enough Info | 0.2812 | 0.0849 | 0.1305 | 123,936 |

---

## 3. Comparison Table

| Metric | In-Domain (FEVER Urdu) | Cross-Domain (XNLI Urdu) | Δ Drop |
|---|---|---|---|
| **Classification Accuracy** | 92.70% | 40.70% | −52.00% |
| **Macro Precision** | 0.9241 | 0.3882 | −0.5359 |
| **Macro Recall** | 0.9249 | 0.4069 | −0.5180 |
| **Macro F1-Score** | 0.9245 | 0.3645 | −0.5600 |
| **Matthews Corr. (MCC)** | 0.8843 | 0.1229 | −0.7614 |
