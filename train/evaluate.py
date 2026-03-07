"""
Evaluate the trained transliteration model.
Computes per-language Accuracy, Character Error Rate (CER), and Word Error Rate (WER).

Usage:
    python train/evaluate.py
    python train/evaluate.py --model models/transliteration_model_step_30000.pt
    python train/evaluate.py --model models/transliteration_model_step_30000.pt --beam_size 5
"""

import os
import sys
import json
import subprocess
import argparse
import editdistance
from pathlib import Path
from collections import defaultdict


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


def run_translation(model_path: str, src_file: str, output_file: str,
                    beam_size: int = 5, gpu: int = -1):
    """Run OpenNMT-py translation."""
    cmd = [
        sys.executable, "-m", "onmt.bin.translate",
        "-model", model_path,
        "-src", src_file,
        "-output", output_file,
        "-beam_size", str(beam_size),
        "-replace_unk",
        "-verbose",
    ]

    if gpu >= 0:
        cmd.extend(["-gpu", str(gpu)])

    print(f"\nTranslating: {src_file}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Translation error: {result.stderr}")
        sys.exit(1)

    return output_file


def chars_to_word(char_sequence: str) -> str:
    """Convert space-separated characters back to a word.

    Example: 'न म स ् त े' → 'नमस्ते'
    """
    return char_sequence.replace(" ", "").strip()


def get_language_prefix(src_line: str) -> str:
    """Extract language prefix from source line.

    Example: '<hin> n a m a s t e' → 'hin'
    """
    if src_line.startswith("<") and ">" in src_line:
        return src_line[1:src_line.index(">")]
    return "unknown"


def compute_metrics(predictions: list, references: list, sources: list) -> dict:
    """Compute Accuracy, CER, and WER per language and overall."""
    per_lang = defaultdict(lambda: {
        "total": 0, "correct": 0,
        "total_cer_dist": 0, "total_cer_len": 0,
        "total_wer_dist": 0, "total_wer_count": 0,
    })

    for pred, ref, src in zip(predictions, references, sources):
        lang = get_language_prefix(src)

        pred_word = chars_to_word(pred)
        ref_word = chars_to_word(ref)

        stats = per_lang[lang]
        stats["total"] += 1

        # Exact match accuracy
        if pred_word == ref_word:
            stats["correct"] += 1

        # Character Error Rate (CER)
        char_dist = editdistance.eval(list(pred_word), list(ref_word))
        stats["total_cer_dist"] += char_dist
        stats["total_cer_len"] += len(ref_word)

        # Word Error Rate (WER) - for single words, it's 0 or 1
        word_dist = 0 if pred_word == ref_word else 1
        stats["total_wer_dist"] += word_dist
        stats["total_wer_count"] += 1

    # Compute final metrics
    results = {}
    overall = {
        "total": 0, "correct": 0,
        "total_cer_dist": 0, "total_cer_len": 0,
        "total_wer_dist": 0, "total_wer_count": 0,
    }

    for lang, stats in sorted(per_lang.items()):
        accuracy = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        cer = stats["total_cer_dist"] / stats["total_cer_len"] * 100 if stats["total_cer_len"] > 0 else 0
        wer = stats["total_wer_dist"] / stats["total_wer_count"] * 100 if stats["total_wer_count"] > 0 else 0

        results[lang] = {
            "total_pairs": stats["total"],
            "accuracy": round(accuracy, 2),
            "cer": round(cer, 2),
            "wer": round(wer, 2),
        }

        # Accumulate for overall
        for key in overall:
            overall[key] += stats[key]

    # Overall metrics
    accuracy = overall["correct"] / overall["total"] * 100 if overall["total"] > 0 else 0
    cer = overall["total_cer_dist"] / overall["total_cer_len"] * 100 if overall["total_cer_len"] > 0 else 0
    wer = overall["total_wer_dist"] / overall["total_wer_count"] * 100 if overall["total_wer_count"] > 0 else 0

    results["overall"] = {
        "total_pairs": overall["total"],
        "accuracy": round(accuracy, 2),
        "cer": round(cer, 2),
        "wer": round(wer, 2),
    }

    return results


def print_results(results: dict):
    """Pretty-print evaluation results."""
    print(f"\n{'='*70}")
    print(f"{'EVALUATION RESULTS':^70}")
    print(f"{'='*70}")
    print(f"{'Language':<15} {'Pairs':>8} {'Accuracy%':>12} {'CER%':>10} {'WER%':>10}")
    print(f"{'-'*70}")

    lang_names = {"hin": "Hindi", "ben": "Bengali", "tam": "Tamil"}

    for lang, metrics in sorted(results.items()):
        if lang == "overall":
            continue
        name = lang_names.get(lang, lang)
        print(f"{name:<15} {metrics['total_pairs']:>8,} {metrics['accuracy']:>11.2f} {metrics['cer']:>9.2f} {metrics['wer']:>9.2f}")

    print(f"{'-'*70}")
    overall = results["overall"]
    print(f"{'OVERALL':<15} {overall['total_pairs']:>8,} {overall['accuracy']:>11.2f} {overall['cer']:>9.2f} {overall['wer']:>9.2f}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate transliteration model")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint (default: latest)")
    parser.add_argument("--src", type=str, default="data/processed/src-test.txt",
                        help="Source test file")
    parser.add_argument("--ref", type=str, default="data/processed/tgt-test.txt",
                        help="Reference target file")
    parser.add_argument("--output", type=str, default="results/predictions.txt",
                        help="Output predictions file")
    parser.add_argument("--beam_size", type=int, default=5,
                        help="Beam size for decoding")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU to use (-1 for CPU)")
    args = parser.parse_args()

    # Find model
    model_path = args.model or find_best_model()

    # Create results directory
    Path("results").mkdir(parents=True, exist_ok=True)

    # Run translation
    run_translation(model_path, args.src, args.output,
                    beam_size=args.beam_size, gpu=args.gpu)

    # Read files
    with open(args.src, "r", encoding="utf-8") as f:
        sources = [line.strip() for line in f]
    with open(args.ref, "r", encoding="utf-8") as f:
        references = [line.strip() for line in f]
    with open(args.output, "r", encoding="utf-8") as f:
        predictions = [line.strip() for line in f]

    # Compute metrics
    results = compute_metrics(predictions, references, sources)

    # Print results
    print_results(results)

    # Save results
    results_file = Path("results/evaluation.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {results_file}")

    # Save sample outputs
    samples_file = Path("results/sample_outputs.txt")
    with open(samples_file, "w", encoding="utf-8") as f:
        f.write("Sample Transliteration Outputs\n")
        f.write("=" * 60 + "\n\n")
        lang_samples = defaultdict(list)
        for src, ref, pred in zip(sources, references, predictions):
            lang = get_language_prefix(src)
            if len(lang_samples[lang]) < 10:
                lang_samples[lang].append((src, ref, pred))

        for lang, samples in sorted(lang_samples.items()):
            lang_names = {"hin": "Hindi", "ben": "Bengali", "tam": "Tamil"}
            f.write(f"\n--- {lang_names.get(lang, lang)} ---\n")
            for src, ref, pred in samples:
                src_word = chars_to_word(src.split("> ", 1)[-1]) if ">" in src else chars_to_word(src)
                ref_word = chars_to_word(ref)
                pred_word = chars_to_word(pred)
                match = "✓" if ref_word == pred_word else "✗"
                f.write(f"  {src_word} → {pred_word} (expected: {ref_word}) {match}\n")

    print(f"Sample outputs saved to: {samples_file}")


if __name__ == "__main__":
    main()
