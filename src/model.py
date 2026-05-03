"""
Model Definition
=================
XLM-RoBERTa-Large with a 3-class sequence classification head
for hallucination detection (NLI-style).

Labels:
  0 = Supported
  1 = Not Supported (Refuted)
  2 = Not Enough Info
"""

from transformers import AutoModelForSequenceClassification, AutoConfig

from src.utils import FRIENDLY_LABELS


# ============================================
# Constants
# ============================================

DEFAULT_MODEL_NAME = "xlm-roberta-large"
NUM_LABELS = 3

# Maps used during inference
ID2LABEL = {i: name for i, name in FRIENDLY_LABELS.items()}
LABEL2ID = {name: i for i, name in FRIENDLY_LABELS.items()}


# ============================================
# Model Factory
# ============================================

def create_model(
    model_name: str = DEFAULT_MODEL_NAME,
    num_labels: int = NUM_LABELS,
    from_checkpoint: str = None,
):
    """
    Create or load an XLM-RoBERTa model for sequence classification.

    Args:
        model_name: Hugging Face model name (default: xlm-roberta-large).
        num_labels: Number of output classes (default: 3).
        from_checkpoint: Path to a saved checkpoint to load from.
                         If None, loads the pretrained base model.

    Returns:
        AutoModelForSequenceClassification instance.
    """
    if from_checkpoint:
        print(f"\n  📦 Loading model from checkpoint: {from_checkpoint}")
        model = AutoModelForSequenceClassification.from_pretrained(
            from_checkpoint,
            num_labels=num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
    else:
        print(f"\n  📦 Loading pretrained model: {model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  ✓ Model loaded successfully")
    print(f"    Total parameters:     {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")
    print(f"    Labels: {ID2LABEL}")

    return model


def get_model_info(model) -> dict:
    """Return a summary dict of the model's configuration."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "model_name": model.config._name_or_path,
        "num_labels": model.config.num_labels,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "hidden_size": model.config.hidden_size,
        "num_layers": model.config.num_hidden_layers,
        "vocab_size": model.config.vocab_size,
    }
