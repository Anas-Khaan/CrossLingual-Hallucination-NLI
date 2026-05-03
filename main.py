"""
Main Orchestration Script
==========================
CLI interface to run the full hallucination detection pipeline.

Usage:
    python main.py download      # Download FEVER dataset
    python main.py map           # Map evidence to claims
    python main.py translate     # Translate to 3 language variants
    python main.py train         # Fine-tune XLM-RoBERTa
    python main.py evaluate      # Evaluate on dev set
    python main.py infer         # Run inference on custom text
    python main.py gpu           # Check GPU status
"""

import argparse
import sys
import torch

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def check_gpu():
    """Print GPU detection status."""
    print("\n" + "=" * 60)
    print("  🖥️  GPU Status Check")
    print("=" * 60)
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available:  {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  CUDA version:    {torch.version.cuda}")
        print(f"  GPU count:       {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"    Compute: {props.major}.{props.minor}")
        print("\n  ✅ GPU is ready for training!")
    else:
        print("\n  ❌ No GPU detected!")
        print("  Fix: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126")
    print("=" * 60 + "\n")


def cmd_download(args):
    """Download FEVER dataset."""
    from src.utils import download_fever_dataset
    download_fever_dataset()


def cmd_map(args):
    """Map evidence to claims."""
    from src.mapping import map_all
    map_all(
        max_claims=args.max_claims,
        max_wiki_files=args.max_wiki_files,
    )


def cmd_translate(args):
    """Translate to 3 language variants."""
    from src.translator import translate_all
    translate_all(max_rows=args.max_rows)


def cmd_train(args):
    """Train the model."""
    from src.train import train_model

    variants = None
    if args.variants:
        variants = args.variants.split(",")

    train_model(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        fp16=not args.no_fp16,
        variants=variants,
    )


def cmd_evaluate(args):
    """Evaluate the model."""
    from src.evaluate import evaluate_model

    variants = None
    if args.variants:
        variants = args.variants.split(",")

    evaluate_model(
        model_path=args.model_path,
        split=args.split,
        variants=variants,
    )


def cmd_infer(args):
    """Run inference on custom text."""
    from src.inference import HallucinationDetector

    detector = HallucinationDetector(model_path=args.model_path)

    if args.source and args.claim:
        # Single prediction from command line
        if args.mode == "sentence":
            result = detector.predict(args.source, args.claim)
        elif args.mode == "paragraph":
            result = detector.predict_paragraph(args.source, args.claim)
        elif args.mode == "document":
            result = detector.predict_document(args.source, args.claim)

        print("\n" + "=" * 60)
        print("  🔍 Prediction Result")
        print("=" * 60)
        print(f"  Source:     {args.source[:100]}")
        print(f"  Claim:      {args.claim[:100]}")
        print(f"  Label:      {result['label']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        if "scores" in result:
            print(f"  Scores:")
            for label, score in result["scores"].items():
                bar = "█" * int(score * 30)
                print(f"    {label:20s}: {score:.4f} {bar}")
        print("=" * 60 + "\n")
    else:
        # Interactive mode
        print("\n🔍 Interactive Hallucination Detection")
        print("Type 'quit' to exit.\n")

        while True:
            source = input("📄 Source text: ").strip()
            if source.lower() == "quit":
                break

            claim = input("💬 Claim text:  ").strip()
            if claim.lower() == "quit":
                break

            result = detector.predict(source, claim)

            print(f"\n  → Label:      {result['label']}")
            print(f"  → Confidence: {result['confidence']:.4f}")
            print(f"  → Scores:")
            for label, score in result["scores"].items():
                bar = "█" * int(score * 30)
                print(f"      {label:20s}: {score:.4f} {bar}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="FEVER Urdu Hallucination Detector — Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py gpu                              # Check GPU status
  python main.py download                         # Download FEVER dataset
  python main.py map --max-claims 50000           # Map 50K claims
  python main.py translate --max-rows 50000       # Translate 50K rows
  python main.py train --epochs 4 --batch-size 16 # Train model
  python main.py evaluate                         # Evaluate on dev set
  python main.py infer                            # Interactive inference
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline commands")

    # GPU check
    subparsers.add_parser("gpu", help="Check GPU status")

    # Download
    subparsers.add_parser("download", help="Download FEVER dataset")

    # Map
    map_parser = subparsers.add_parser("map", help="Map evidence to claims")
    map_parser.add_argument("--max-claims", type=int, default=None,
                            help="Max claims to process (default: all)")
    map_parser.add_argument("--max-wiki-files", type=int, default=None,
                            help="Max wiki dump files to load (default: all)")

    # Translate
    translate_parser = subparsers.add_parser("translate", help="Translate to 3 variants")
    translate_parser.add_argument("--max-rows", type=int, default=None,
                                  help="Max rows to translate (default: all)")

    # Train
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--lr", type=float, default=2e-5)
    train_parser.add_argument("--epochs", type=int, default=4)
    train_parser.add_argument("--no-fp16", action="store_true",
                              help="Disable FP16 mixed precision")
    train_parser.add_argument("--variants", type=str, default=None,
                              help="Comma-separated variants: pure_urdu,mixed,roman_urdu")

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--model-path", type=str, default=None)
    eval_parser.add_argument("--split", type=str, default="test")
    eval_parser.add_argument("--variants", type=str, default=None)

    # Infer
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--model-path", type=str, default=None)
    infer_parser.add_argument("--source", type=str, default=None,
                              help="Source text")
    infer_parser.add_argument("--claim", type=str, default=None,
                              help="Claim text")
    infer_parser.add_argument("--mode", type=str, default="sentence",
                              choices=["sentence", "paragraph", "document"])

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Dispatch
    commands = {
        "gpu": lambda a: check_gpu(),
        "download": cmd_download,
        "map": cmd_map,
        "translate": cmd_translate,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "infer": cmd_infer,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
