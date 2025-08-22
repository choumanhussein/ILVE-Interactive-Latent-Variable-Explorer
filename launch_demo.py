#!/usr/bin/env python
"""
Continuous Latent Variables Project Launcher

This script checks requirements, trains a basic VAE if no checkpoints are
present, and launches the Streamlit demo.

Usage
-----
python launch_demo.py              # full interactive workflow
python launch_demo.py --quick      # run quick self-tests only
python launch_demo.py --demo-only  # skip checks and open Streamlit
"""

import os
import sys
import subprocess
import importlib.util          #  ← NEW
import torch                   # (used later for torch.cuda checks, etc.)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Requirement check
# ──────────────────────────────────────────────────────────────────────────────
def check_requirements() -> bool:
    """Return True if *all* required libraries are importable."""
    print("🔍 Checking requirements...")

    # Friendly-name  →  actual import-name
    required = {
        "torch"        : "torch",
        "torchvision"  : "torchvision",
        "matplotlib"   : "matplotlib",
        "numpy"        : "numpy",
        "streamlit"    : "streamlit",
        "plotly"       : "plotly",
        "Pillow"       : "PIL",        # PyPI pillow → import PIL
        "scikit-learn" : "sklearn"     # PyPI scikit-learn → import sklearn
    }

    missing = []
    for printable, mod in required.items():
        if importlib.util.find_spec(mod) is None:
            print(f"❌ {printable}")
            missing.append(printable)
        else:
            print(f"✅ {printable}")

    if missing:
        print(f"\n🚨 Missing packages: {', '.join(missing)}")
        print("Please install with: pip install -r requirements.txt")
        return False

    print("✅ All requirements satisfied!")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 2. Model checkpoint utilities
# ──────────────────────────────────────────────────────────────────────────────
def check_models() -> bool:
    """Detect whether trained .pth checkpoints exist."""
    print("\n🧠 Checking for trained models...")

    ckpt_dir = "experiments/checkpoints"
    if not os.path.isdir(ckpt_dir):
        print("❌ No checkpoint directory found")
        return False

    models = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
    if not models:
        print("❌ No trained models found")
        return False

    print(f"✅ Found {len(models)} trained models:")
    for m in models:
        print(f"   📦 {m}")
    return True


def train_basic_model() -> bool:
    """Train a small 2-D VAE used by the demo."""
    print("\n🏋️  Training a basic model for demonstration… (≈5–10 min)")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "train_basic_vae.py",
                "--epochs", "15",
                "--latent-dim", "2",
                "--no-wandb"
            ],
            capture_output=True,
            text=True,
            timeout=600            # 10 min
        )
    except subprocess.TimeoutExpired:
        print("❌ Training timed out")
        return False
    except Exception as exc:       # e.g. FileNotFoundError
        print(f"❌ Training error: {exc}")
        return False

    if result.returncode != 0:
        print("❌ Training failed:")
        print(result.stderr)
        return False

    print("✅ Basic model trained successfully!")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 3. Demo launchers
# ──────────────────────────────────────────────────────────────────────────────
def launch_streamlit() -> None:
    """Spawn the Streamlit web app."""
    print("\n🚀 Launching Streamlit demo…")
    print("The demo will open in your browser.  (Ctrl+C to quit)\n")

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "webapp/app.py"]
        )
    except KeyboardInterrupt:
        print("\n👋 Demo stopped.")
    except Exception as exc:
        print(f"❌ Failed to launch demo: {exc}")


def quick_demo() -> None:
    """Import-time smoke tests for the VAE codebase (no training)."""
    print("🎯 Quick Demo Mode")
    print("=" * 30)

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        print("✅ Basic import test passed")
    except ImportError as exc:
        print(f"❌ Missing requirement: {exc}")
        return

    sys.path.append("src")
    try:
        from models.base_vae import VAE, test_vae
        from models.beta_vae import test_beta_vae

        print("\n🧠 Testing VAE implementation…")
        test_vae()

        print("\n🎯 Testing β-VAE implementation…")
        test_beta_vae()

        print("\n✅ Quick demo completed successfully!")
        print("\nNext steps:")
        print("  1. Train models: python train_basic_vae.py --epochs 10 --no-wandb")
        print("  2. Run experiments: python run_beta_experiments.py --quick --no-wandb")
        print("  3. Launch demo: streamlit run webapp/app.py")
    except Exception as exc:
        print(f"❌ Demo failed: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Main interactive workflow
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("🚀 Continuous Latent Variables Project Launcher")
    print("=" * 50)

    # 1. Requirement check
    if not check_requirements():
        return

    # 2. Check for checkpoints
    have_models = check_models()
    if not have_models:
        print("\n🎯 No trained models found. Let’s train one.")
        if input("Train a basic model now? (y/n): ").strip().lower() == "y":
            if not train_basic_model():
                return
        else:
            print("\n📚 Manual training examples:")
            print("  python train_basic_vae.py --epochs 15 --latent-dim 2 --no-wandb")
            print("  python run_beta_experiments.py --quick --no-wandb")
            return

    # 3. Launch demo
    print("\n🎮 Ready to launch the interactive demo!")
    if input("Launch web demo now? (y/n): ").strip().lower() == "y":
        launch_streamlit()
    else:
        print("\n📋 Manual launch options:")
        print("  🌐 Web demo: streamlit run webapp/app.py")
        print("  🎮 Explorer:  python notebooks/interactive_latent_explorer.py")
        print("  🧪 Experiments: python run_beta_experiments.py --quick --no-wandb")


# ──────────────────────────────────────────────────────────────────────────────
# 5. CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--quick":
            quick_demo()
        elif arg == "--demo-only":
            launch_streamlit()
        elif arg == "--help":
            print(__doc__)
        else:
            print("Unknown option. Use --help for usage.")
    else:
        main()
