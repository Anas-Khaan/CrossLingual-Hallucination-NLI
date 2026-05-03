"""Quick verification script — checks all packages and GPU."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 55)
print("  INSTALLATION VERIFICATION")
print("=" * 55)

packages = {
    "torch": "PyTorch",
    "torchvision": "TorchVision",
    "torchaudio": "TorchAudio",
    "transformers": "Transformers (HF)",
    "datasets": "Datasets (HF)",
    "accelerate": "Accelerate (HF)",
    "deep_translator": "Deep Translator",
    "indic_transliteration": "Indic Transliteration",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "tqdm": "TQDM",
    "sklearn": "Scikit-Learn",
    "matplotlib": "Matplotlib",
    "seaborn": "Seaborn",
    "requests": "Requests",
}

all_ok = True
for module, name in packages.items():
    try:
        mod = __import__(module)
        ver = getattr(mod, "__version__", "ok")
        print(f"  [OK]   {name:25s} {ver}")
    except ImportError:
        print(f"  [FAIL] {name:25s} NOT INSTALLED")
        all_ok = False

print()
print("=" * 55)
print("  GPU STATUS")
print("=" * 55)

import torch
print(f"  PyTorch version:  {torch.__version__}")
print(f"  CUDA available:   {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  CUDA version:     {torch.version.cuda}")
    print(f"  GPU:              {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  VRAM:             {props.total_memory / 1e9:.1f} GB")
    print(f"\n  >>> GPU READY FOR TRAINING! <<<")
else:
    print(f"\n  >>> NO GPU DETECTED <<<")

print("=" * 55)

if all_ok:
    print("\n  ALL PACKAGES INSTALLED SUCCESSFULLY!\n")
else:
    print("\n  Some packages are missing.\n")
