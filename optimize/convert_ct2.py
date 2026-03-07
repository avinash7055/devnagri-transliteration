"""
Convert trained OpenNMT-py model to CTranslate2 format for optimized inference.
Supports int8 quantization for reduced model size and faster inference.

Usage:
    python optimize/convert_ct2.py
    python optimize/convert_ct2.py --model models/transliteration_model_step_30000.pt
    python optimize/convert_ct2.py --quantization int8
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path


def find_best_model(models_dir: str = "models") -> str:
    """Find the latest model checkpoint."""
    model_dir = Path(models_dir)
    checkpoints = sorted(model_dir.glob("transliteration_model_step_*.pt"))
    if not checkpoints:
        print("ERROR: No model checkpoints found in models/")
        sys.exit(1)
    best = str(checkpoints[-1])
    print(f"Using model: {best}")
    return best


def get_dir_size(path: Path) -> float:
    """Get directory size in MB."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def get_file_size(path: str) -> float:
    """Get file size in MB."""
    return os.path.getsize(path) / (1024 * 1024)


def convert_model(model_path: str, output_dir: str, quantization: str = "int8"):
    """Convert OpenNMT-py model to CTranslate2 format."""
    output_path = Path(output_dir)

    # Remove existing output directory
    if output_path.exists():
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    # Get original model size
    original_size = get_file_size(model_path)
    print(f"\nOriginal model size: {original_size:.2f} MB")

    # Run CTranslate2 converter
    cmd = [
        "ct2-opennmt-py-converter",
        "--model_path", model_path,
        "--output_dir", output_dir,
        "--quantization", quantization,
    ]

    print(f"\nConverting model with {quantization} quantization...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"\nERROR: Conversion failed!")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)

    if result.stdout:
        print(result.stdout)

    # Get optimized model size
    optimized_size = get_dir_size(output_path)
    reduction = (1 - optimized_size / original_size) * 100 if original_size > 0 else 0

    print(f"\n{'='*60}")
    print(f"Conversion Complete!")
    print(f"{'='*60}")
    print(f"  Original model:  {original_size:.2f} MB")
    print(f"  Optimized model: {optimized_size:.2f} MB")
    print(f"  Size reduction:  {reduction:.1f}%")
    print(f"  Quantization:    {quantization}")
    print(f"  Output:          {output_dir}")
    print(f"{'='*60}")

    return {
        "original_size_mb": round(original_size, 2),
        "optimized_size_mb": round(optimized_size, 2),
        "size_reduction_pct": round(reduction, 1),
        "quantization": quantization,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert OpenNMT-py model to CTranslate2")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint (default: latest)")
    parser.add_argument("--output_dir", type=str, default="models/ct2_model",
                        help="Output directory for CTranslate2 model")
    parser.add_argument("--quantization", type=str, default="int8",
                        choices=["int8", "int16", "float16", "float32"],
                        help="Quantization type (default: int8)")
    args = parser.parse_args()

    model_path = args.model or find_best_model()
    result = convert_model(model_path, args.output_dir, args.quantization)

    # Save conversion info
    import json
    info_file = Path("results/conversion_info.json")
    info_file.parent.mkdir(parents=True, exist_ok=True)
    with open(info_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nConversion info saved to: {info_file}")


if __name__ == "__main__":
    main()
