"""
Training launcher script for the transliteration model.
Wraps OpenNMT-py CLI commands for vocabulary building and training.

Usage:
    python train/train.py                        # Full training
    python train/train.py --build_vocab_only      # Only build vocabulary
    python train/train.py --max_steps 100         # Quick smoke test
    python train/train.py --config configs/transliteration.yaml  # Custom config
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd: list, description: str):
    """Run a shell command with error handling."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\nERROR: {description} failed with return code {result.returncode}")
        sys.exit(1)
    return result


def build_vocab(config_path: str):
    """Build vocabulary from training data."""
    # Create models directory
    Path("models").mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "onmt.bin.build_vocab",
        "-config", config_path,
        "-n_sample", "-1",  # Use all data for vocab
    ]
    run_command(cmd, "Building Vocabulary")


def train_model(config_path: str, max_steps: int = None, gpu: int = 0):
    """Train the transliteration model."""
    Path("models").mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "onmt.bin.train",
        "-config", config_path,
    ]

    if max_steps is not None:
        cmd.extend(["-train_steps", str(max_steps)])

    # GPU configuration
    if gpu >= 0:
        cmd.extend(["-world_size", "1", "-gpu_ranks", str(gpu)])
    else:
        # CPU training
        cmd.extend(["-world_size", "0", "-gpu_ranks", ""])

    run_command(cmd, f"Training Model ({'GPU ' + str(gpu) if gpu >= 0 else 'CPU'})")


def main():
    parser = argparse.ArgumentParser(description="Train transliteration model with OpenNMT-py")
    parser.add_argument("--config", type=str, default="configs/transliteration.yaml",
                        help="Path to OpenNMT-py config file")
    parser.add_argument("--build_vocab_only", action="store_true",
                        help="Only build vocabulary, don't train")
    parser.add_argument("--skip_vocab", action="store_true",
                        help="Skip vocabulary building (use existing vocab)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max training steps (e.g., 100 for smoke test)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU rank to use (-1 for CPU, 0 for first GPU)")
    args = parser.parse_args()

    # Verify config exists
    if not Path(args.config).exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    # Verify preprocessed data exists
    data_dir = Path("data/processed")
    if not data_dir.exists() or not (data_dir / "src-train.txt").exists():
        print("ERROR: Preprocessed data not found. Run 'python data/preprocess.py' first.")
        sys.exit(1)

    # Step 1: Build vocabulary
    if not args.skip_vocab:
        build_vocab(args.config)
        if args.build_vocab_only:
            print("\nVocabulary building complete. Exiting.")
            return

    # Step 2: Train
    print("\nStarting model training...")
    train_model(args.config, max_steps=args.max_steps, gpu=args.gpu)

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Checkpoints saved in: models/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
