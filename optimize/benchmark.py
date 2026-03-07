"""
Benchmark comparison between OpenNMT-py and CTranslate2 models.
Compares inference speed, model size, and transliteration quality.

Usage:
    python optimize/benchmark.py
    python optimize/benchmark.py --num_samples 500
"""

import os
import sys
import json
import time
import argparse
import editdistance
from pathlib import Path

# Try importing ctranslate2
try:
    import ctranslate2
except ImportError:
    print("ERROR: ctranslate2 not installed. Run: pip install ctranslate2")
    sys.exit(1)


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


def load_test_data(src_file: str, tgt_file: str, num_samples: int) -> tuple:
    """Load test data."""
    with open(src_file, "r", encoding="utf-8") as f:
        sources = [line.strip() for line in f]
    with open(tgt_file, "r", encoding="utf-8") as f:
        references = [line.strip() for line in f]

    if num_samples and len(sources) > num_samples:
        sources = sources[:num_samples]
        references = references[:num_samples]

    return sources, references


def tokenize_for_ct2(lines: list) -> list:
    """Tokenize lines into token lists for CTranslate2."""
    return [line.split() for line in lines]


def chars_to_word(char_sequence: str) -> str:
    """Convert space-separated characters back to a word."""
    return char_sequence.replace(" ", "").strip()


def compute_accuracy_cer(predictions: list, references: list) -> dict:
    """Compute accuracy and CER."""
    correct = 0
    total_cer_dist = 0
    total_cer_len = 0

    for pred, ref in zip(predictions, references):
        pred_word = chars_to_word(pred)
        ref_word = chars_to_word(ref)

        if pred_word == ref_word:
            correct += 1

        char_dist = editdistance.eval(list(pred_word), list(ref_word))
        total_cer_dist += char_dist
        total_cer_len += len(ref_word)

    total = len(predictions)
    accuracy = correct / total * 100 if total > 0 else 0
    cer = total_cer_dist / total_cer_len * 100 if total_cer_len > 0 else 0

    return {
        "accuracy": round(accuracy, 2),
        "cer": round(cer, 2),
    }


def benchmark_ctranslate2(ct2_model_dir: str, sources: list, references: list,
                          beam_size: int = 5) -> dict:
    """Benchmark CTranslate2 model."""
    print("\n--- Benchmarking CTranslate2 Model ---")

    # Load model
    translator = ctranslate2.Translator(ct2_model_dir, device="cpu")

    # Tokenize
    source_tokens = tokenize_for_ct2(sources)

    # Warm-up
    print("  Warming up...")
    warmup_batch = source_tokens[:min(10, len(source_tokens))]
    for _ in range(3):
        translator.translate_batch(warmup_batch, beam_size=beam_size)

    # Benchmark
    print(f"  Translating {len(source_tokens)} samples...")
    start_time = time.perf_counter()

    results = translator.translate_batch(
        source_tokens,
        beam_size=beam_size,
        max_decoding_length=150,
    )

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Extract predictions
    predictions = []
    total_tokens = 0
    for result in results:
        tokens = result.hypotheses[0]
        predictions.append(" ".join(tokens))
        total_tokens += len(tokens)

    # Compute metrics
    quality = compute_accuracy_cer(predictions, references)

    latency_per_sample = (total_time / len(sources)) * 1000  # ms
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0

    # Model size
    model_size = get_dir_size(Path(ct2_model_dir))

    benchmark = {
        "engine": "CTranslate2",
        "device": "CPU",
        "total_time_s": round(total_time, 3),
        "latency_ms_per_sample": round(latency_per_sample, 2),
        "tokens_per_second": round(tokens_per_second, 1),
        "model_size_mb": round(model_size, 2),
        "accuracy": quality["accuracy"],
        "cer": quality["cer"],
        "num_samples": len(sources),
    }

    print(f"  Time: {total_time:.3f}s")
    print(f"  Latency: {latency_per_sample:.2f} ms/sample")
    print(f"  Throughput: {tokens_per_second:.1f} tokens/s")
    print(f"  Model size: {model_size:.2f} MB")
    print(f"  Accuracy: {quality['accuracy']:.2f}%")
    print(f"  CER: {quality['cer']:.2f}%")

    return benchmark


def benchmark_opennmt(model_path: str, src_file: str, references: list,
                      beam_size: int = 5) -> dict:
    """Benchmark OpenNMT-py model via CLI."""
    import subprocess
    import tempfile

    print("\n--- Benchmarking OpenNMT-py Model ---")

    # Create temp output file
    output_file = Path("results/benchmark_opennmt_output.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "onmt.bin.translate",
        "-model", model_path,
        "-src", src_file,
        "-output", str(output_file),
        "-beam_size", str(beam_size),
        "-replace_unk",
        "-verbose",
    ]

    print(f"  Translating...")
    start_time = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.perf_counter()
    total_time = end_time - start_time

    if result.returncode != 0:
        print(f"  WARNING: OpenNMT translation failed: {result.stderr}")
        return None

    # Read predictions
    with open(output_file, "r", encoding="utf-8") as f:
        predictions = [line.strip() for line in f]

    # Compute metrics
    quality = compute_accuracy_cer(predictions, references)

    total_tokens = sum(len(p.split()) for p in predictions)
    latency_per_sample = (total_time / len(references)) * 1000
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0

    model_size = get_file_size(model_path)

    benchmark = {
        "engine": "OpenNMT-py",
        "device": "CPU",
        "total_time_s": round(total_time, 3),
        "latency_ms_per_sample": round(latency_per_sample, 2),
        "tokens_per_second": round(tokens_per_second, 1),
        "model_size_mb": round(model_size, 2),
        "accuracy": quality["accuracy"],
        "cer": quality["cer"],
        "num_samples": len(references),
    }

    print(f"  Time: {total_time:.3f}s")
    print(f"  Latency: {latency_per_sample:.2f} ms/sample")
    print(f"  Throughput: {tokens_per_second:.1f} tokens/s")
    print(f"  Model size: {model_size:.2f} MB")
    print(f"  Accuracy: {quality['accuracy']:.2f}%")
    print(f"  CER: {quality['cer']:.2f}%")

    return benchmark


def print_comparison(opennmt_bench: dict, ct2_bench: dict):
    """Print side-by-side benchmark comparison."""
    print(f"\n{'='*70}")
    print(f"{'BENCHMARK COMPARISON':^70}")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'OpenNMT-py':>18} {'CTranslate2':>18}")
    print(f"{'-'*70}")

    metrics = [
        ("Latency (ms/sample)", "latency_ms_per_sample"),
        ("Throughput (tokens/s)", "tokens_per_second"),
        ("Model Size (MB)", "model_size_mb"),
        ("Accuracy (%)", "accuracy"),
        ("CER (%)", "cer"),
    ]

    for label, key in metrics:
        onmt_val = opennmt_bench.get(key, "N/A") if opennmt_bench else "N/A"
        ct2_val = ct2_bench.get(key, "N/A")
        print(f"{label:<30} {str(onmt_val):>18} {str(ct2_val):>18}")

    # Speed gain
    if opennmt_bench and ct2_bench:
        speed_gain = opennmt_bench["latency_ms_per_sample"] / ct2_bench["latency_ms_per_sample"]
        size_reduction = (1 - ct2_bench["model_size_mb"] / opennmt_bench["model_size_mb"]) * 100
        print(f"{'-'*70}")
        print(f"{'Speed Gain':<30} {f'{speed_gain:.2f}x faster':>38}")
        print(f"{'Size Reduction':<30} {f'{size_reduction:.1f}%':>38}")

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark transliteration models")
    parser.add_argument("--ct2_model", type=str, default="models/ct2_model",
                        help="Path to CTranslate2 model directory")
    parser.add_argument("--onmt_model", type=str, default=None,
                        help="Path to OpenNMT-py model checkpoint (default: latest)")
    parser.add_argument("--src", type=str, default="data/processed/src-test.txt",
                        help="Source test file")
    parser.add_argument("--ref", type=str, default="data/processed/tgt-test.txt",
                        help="Reference target file")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to benchmark")
    parser.add_argument("--beam_size", type=int, default=5,
                        help="Beam size for decoding")
    args = parser.parse_args()

    # Load test data
    sources, references = load_test_data(args.src, args.ref, args.num_samples)
    print(f"Loaded {len(sources)} test samples")

    # Create trimmed source file for OpenNMT benchmarking
    trimmed_src = "results/benchmark_src.txt"
    Path("results").mkdir(parents=True, exist_ok=True)
    with open(trimmed_src, "w", encoding="utf-8") as f:
        for line in sources:
            f.write(line + "\n")

    # Benchmark CTranslate2
    ct2_bench = benchmark_ctranslate2(args.ct2_model, sources, references, args.beam_size)

    # Benchmark OpenNMT-py
    opennmt_bench = None
    if args.onmt_model:
        onmt_model = args.onmt_model
    else:
        model_dir = Path("models")
        checkpoints = sorted(model_dir.glob("transliteration_model_step_*.pt"))
        if checkpoints:
            onmt_model = str(checkpoints[-1])
        else:
            onmt_model = None

    if onmt_model:
        opennmt_bench = benchmark_opennmt(onmt_model, trimmed_src, references, args.beam_size)

    # Print comparison
    print_comparison(opennmt_bench, ct2_bench)

    # Save results
    results = {
        "ctranslate2": ct2_bench,
        "opennmt": opennmt_bench,
    }

    if opennmt_bench and ct2_bench:
        results["comparison"] = {
            "speed_gain": round(opennmt_bench["latency_ms_per_sample"] / ct2_bench["latency_ms_per_sample"], 2),
            "size_reduction_pct": round((1 - ct2_bench["model_size_mb"] / opennmt_bench["model_size_mb"]) * 100, 1),
            "accuracy_diff": round(ct2_bench["accuracy"] - opennmt_bench["accuracy"], 2),
            "cer_diff": round(ct2_bench["cer"] - opennmt_bench["cer"], 2),
        }

    results_file = Path("results/benchmark.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark results saved to: {results_file}")


if __name__ == "__main__":
    main()
