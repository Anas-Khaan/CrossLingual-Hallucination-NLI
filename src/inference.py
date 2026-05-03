"""
Inference Module
================
Run hallucination detection on new (source, claim) pairs.
Supports single sentences, paragraphs, and documents.
Works with Pure Urdu, Urdu-English Mixed, and Roman Urdu inputs.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils import MODELS_DIR, FRIENDLY_LABELS, split_into_sentences, chunk_text


# ============================================
# Inference Engine
# ============================================

class HallucinationDetector:
    """
    Hallucination Detection Inference Engine.

    Usage:
        detector = HallucinationDetector("models/checkpoints/xlm-roberta-hallucination/best_model")
        result = detector.predict(
            source="یہ ایک بلی ہے",
            claim="یہ ایک کتا ہے"
        )
        # → {"label": "Not Supported", "confidence": 0.94, "scores": {...}}
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        max_length: int = 512,
    ):
        """
        Initialize the detector.

        Args:
            model_path: Path to the saved model checkpoint.
            device: "cuda" or "cpu". Auto-detects if None.
            max_length: Max token length for the model.
        """
        if model_path is None:
            model_path = str(MODELS_DIR / "xlm-roberta-hallucination" / "best_model")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.max_length = max_length

        print(f"\n🔍 Loading Hallucination Detector...")
        print(f"   Model:  {model_path}")
        print(f"   Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        print(f"   ✓ Model loaded and ready!\n")

    def predict(
        self,
        source: str,
        claim: str,
    ) -> Dict[str, Union[str, float, Dict]]:
        """
        Predict whether a claim is supported by the source text.

        Args:
            source: The reference/source text (evidence).
            claim: The claim or generated text to verify.

        Returns:
            Dict with:
              - label: "Supported", "Not Supported", or "Not Enough Info"
              - confidence: Probability of the predicted label (0-1)
              - scores: Probabilities for all three labels
        """
        # Tokenize
        encoding = self.tokenizer(
            source,
            claim,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        predicted_label_id = int(np.argmax(probs))
        confidence = float(probs[predicted_label_id])

        return {
            "label": FRIENDLY_LABELS[predicted_label_id],
            "confidence": round(confidence, 4),
            "scores": {
                FRIENDLY_LABELS[i]: round(float(probs[i]), 4)
                for i in range(len(probs))
            },
        }

    def predict_paragraph(
        self,
        source: str,
        claim: str,
        aggregation: str = "majority",
    ) -> Dict[str, Union[str, float, List]]:
        """
        Predict for paragraph-level inputs by splitting into sentences
        and aggregating predictions.

        Args:
            source: Source paragraph/text.
            claim: Claim paragraph/text.
            aggregation: "majority" (majority vote) or "average" (avg probabilities).

        Returns:
            Dict with aggregated label, confidence, and per-sentence details.
        """
        # Split claim into sentences
        claim_sentences = split_into_sentences(claim)
        if not claim_sentences:
            claim_sentences = [claim]

        # Get prediction for each sentence
        sentence_results = []
        for sent in claim_sentences:
            result = self.predict(source=source, claim=sent)
            sentence_results.append({
                "sentence": sent,
                **result,
            })

        # Aggregate
        if aggregation == "majority":
            label_counts = {}
            for r in sentence_results:
                label = r["label"]
                label_counts[label] = label_counts.get(label, 0) + 1

            final_label = max(label_counts, key=label_counts.get)
            final_confidence = label_counts[final_label] / len(sentence_results)

        elif aggregation == "average":
            avg_scores = {FRIENDLY_LABELS[i]: 0.0 for i in range(3)}
            for r in sentence_results:
                for label, score in r["scores"].items():
                    avg_scores[label] += score
            for label in avg_scores:
                avg_scores[label] /= len(sentence_results)

            final_label = max(avg_scores, key=avg_scores.get)
            final_confidence = avg_scores[final_label]
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        return {
            "label": final_label,
            "confidence": round(final_confidence, 4),
            "num_sentences": len(claim_sentences),
            "sentence_results": sentence_results,
        }

    def predict_document(
        self,
        source: str,
        claim: str,
        max_chunk_words: int = 400,
        overlap: int = 50,
    ) -> Dict[str, Union[str, float, List]]:
        """
        Predict for document-level inputs by chunking both source and claim.

        Args:
            source: Source document text.
            claim: Claim document text.
            max_chunk_words: Max words per chunk.
            overlap: Word overlap between chunks.

        Returns:
            Dict with aggregated label, confidence, and per-chunk details.
        """
        # Chunk source and claim
        source_chunks = chunk_text(source, max_tokens=max_chunk_words, overlap=overlap)
        claim_chunks = chunk_text(claim, max_tokens=max_chunk_words, overlap=overlap)

        # Cross-check each claim chunk against all source chunks
        all_results = []
        for c_chunk in claim_chunks:
            chunk_scores = {FRIENDLY_LABELS[i]: 0.0 for i in range(3)}

            for s_chunk in source_chunks:
                result = self.predict(source=s_chunk, claim=c_chunk)
                for label, score in result["scores"].items():
                    chunk_scores[label] = max(chunk_scores[label], score)

            best_label = max(chunk_scores, key=chunk_scores.get)
            all_results.append({
                "claim_chunk": c_chunk[:100] + "...",
                "label": best_label,
                "confidence": round(chunk_scores[best_label], 4),
            })

        # Aggregate via majority vote
        label_counts = {}
        for r in all_results:
            label = r["label"]
            label_counts[label] = label_counts.get(label, 0) + 1

        final_label = max(label_counts, key=label_counts.get)
        final_confidence = label_counts[final_label] / len(all_results)

        return {
            "label": final_label,
            "confidence": round(final_confidence, 4),
            "num_chunks": len(all_results),
            "chunk_results": all_results,
        }


# ============================================
# Quick Inference Function
# ============================================

def predict(
    source: str,
    claim: str,
    model_path: str = None,
    mode: str = "sentence",
) -> Dict:
    """
    Quick inference function (creates detector on first call).

    Args:
        source: Source/reference text.
        claim: Claim/generated text to verify.
        model_path: Path to model checkpoint.
        mode: "sentence", "paragraph", or "document".

    Returns:
        Prediction result dict.
    """
    detector = HallucinationDetector(model_path=model_path)

    if mode == "sentence":
        return detector.predict(source, claim)
    elif mode == "paragraph":
        return detector.predict_paragraph(source, claim)
    elif mode == "document":
        return detector.predict_document(source, claim)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'sentence', 'paragraph', or 'document'.")
