"""
Preprocess Aksharantar JSONL data into character-level parallel text files
for OpenNMT-py training.

Each word is tokenized into space-separated characters.
Language prefix tokens (<hin>, <ben>, <tam>) are added to the source
to enable multilingual training with a single model.

Usage:
    python data/preprocess.py
    python data/preprocess.py --max_train 100000 --max_val 5000 --max_test 5000
"""

import os
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

# Language codes
LANGUAGES = ["hin", "ben", "tam"]

# Directories
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def tokenize_to_chars(word: str) -> str:
    """Convert a word into space-separated characters.

    Example: 'hello' → 'h e l l o'
             'नमस्ते' → 'न म स ् त े'
    """
    return " ".join(list(word.strip()))


def read_jsonl(filepath: Path) -> list:
    """Read a JSONL file and return list of records."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                records.append(record)
    return records


def extract_pairs(records: list, lang_code: str) -> list:
    """Extract (source, target) pairs from JSONL records.

    Source = language_prefix + space-separated English characters
    Target = space-separated native script characters

    The dataset has fields: 'native word' and 'english word'
    (may also use 'native_word' and 'english_word' depending on version)
    """
    pairs = []
    for record in records:
        # Handle both field name formats
        english = record.get("english word") or record.get("english_word") or record.get("en", "")
        native = record.get("native word") or record.get("native_word") or record.get("native", "")

        if not english or not native:
            continue

        english = english.strip()
        native = native.strip()

        # Skip empty or single-char entries
        if len(english) < 1 or len(native) < 1:
            continue

        # Create character-level tokens with language prefix
        src = f"<{lang_code}> {tokenize_to_chars(english)}"
        tgt = tokenize_to_chars(native)

        pairs.append((src, tgt))

    return pairs


def write_parallel_files(pairs: list, src_path: Path, tgt_path: Path):
    """Write source and target parallel text files."""
    with open(src_path, "w", encoding="utf-8") as src_f, \
         open(tgt_path, "w", encoding="utf-8") as tgt_f:
        for src, tgt in pairs:
            src_f.write(src + "\n")
            tgt_f.write(tgt + "\n")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Aksharantar data for OpenNMT-py")
    parser.add_argument("--max_train", type=int, default=100000,
                        help="Max training pairs per language (default: 100000)")
    parser.add_argument("--max_val", type=int, default=5000,
                        help="Max validation pairs per language (default: 5000)")
    parser.add_argument("--max_test", type=int, default=5000,
                        help="Max test pairs per language (default: 5000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Split limits
    max_per_split = {
        "train": args.max_train,
        "test": args.max_test,
    }

    # Collect all pairs by split
    all_pairs = defaultdict(list)
    stats = {}

    print("=" * 60)
    print("Preprocessing Aksharantar Data")
    print(f"Max per language — train: {args.max_train:,}, val: {args.max_val:,}, test: {args.max_test:,}")
    print("=" * 60)

    for lang_code in LANGUAGES:
        lang_dir = RAW_DIR / lang_code
        if not lang_dir.exists():
            print(f"\nWARNING: No data found for {lang_code} at {lang_dir}")
            continue

        print(f"\nProcessing {lang_code}...")
        lang_stats = {}

        # Try to find split files
        for split_name in ["train", "test"]:
            # Try different possible filenames
            possible_files = [
                lang_dir / f"{split_name}.jsonl",
                lang_dir / f"{split_name}.json",
            ]

            jsonl_file = None
            for pf in possible_files:
                if pf.exists():
                    jsonl_file = pf
                    break

            if jsonl_file is None:
                print(f"  WARNING: No {split_name} file found for {lang_code}")
                continue

            records = read_jsonl(jsonl_file)
            pairs = extract_pairs(records, lang_code)

            # Shuffle and limit
            random.shuffle(pairs)
            limit = max_per_split.get(split_name, args.max_val)
            if len(pairs) > limit:
                pairs = pairs[:limit]

            all_pairs[split_name].extend(pairs)
            lang_stats[split_name] = len(pairs)
            print(f"  {split_name}: {len(pairs):,} pairs (from {len(records):,} raw records)")

        # Handle validation split - check for 'val' or 'validation' or 'dev'
        val_file = None
        for val_name in ["val", "validation", "dev"]:
            for ext in [".jsonl", ".json"]:
                candidate = lang_dir / f"{val_name}{ext}"
                if candidate.exists():
                    val_file = candidate
                    break
            if val_file:
                break

        if val_file:
            records = read_jsonl(val_file)
            pairs = extract_pairs(records, lang_code)
            random.shuffle(pairs)
            if len(pairs) > args.max_val:
                pairs = pairs[:args.max_val]
            all_pairs["val"].extend(pairs)
            lang_stats["val"] = len(pairs)
            print(f"  val: {len(pairs):,} pairs")
        else:
            # Split from train if no val file exists
            train_pairs = all_pairs.get("train", [])
            # Take last max_val items as validation
            if len(train_pairs) > args.max_val:
                val_pairs = train_pairs[-args.max_val:]
                all_pairs["train"] = train_pairs[:-args.max_val]
                all_pairs["val"].extend(val_pairs)
                lang_stats["val"] = len(val_pairs)
                print(f"  val: {len(val_pairs):,} pairs (split from train)")

        stats[lang_code] = lang_stats

    # Shuffle combined multilingual data
    for split_name in all_pairs:
        random.shuffle(all_pairs[split_name])

    # Write parallel text files
    print(f"\n{'='*60}")
    print("Writing parallel text files...")
    print(f"{'='*60}")

    for split_name, pairs in all_pairs.items():
        src_file = PROCESSED_DIR / f"src-{split_name}.txt"
        tgt_file = PROCESSED_DIR / f"tgt-{split_name}.txt"
        write_parallel_files(pairs, src_file, tgt_file)
        print(f"  {split_name}: {len(pairs):,} pairs → {src_file}, {tgt_file}")

    # Show sample pairs
    print(f"\n{'='*60}")
    print("Sample pairs (first 5 from train):")
    print(f"{'='*60}")
    for src, tgt in all_pairs.get("train", [])[:5]:
        print(f"  SRC: {src}")
        print(f"  TGT: {tgt}")
        print()

    # Save preprocessing summary
    summary = {
        "languages": LANGUAGES,
        "max_per_language": {
            "train": args.max_train,
            "val": args.max_val,
            "test": args.max_test,
        },
        "total_pairs": {split: len(pairs) for split, pairs in all_pairs.items()},
        "per_language": stats,
    }

    summary_file = PROCESSED_DIR / "preprocess_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_file}")
    print("Done!")


if __name__ == "__main__":
    main()
