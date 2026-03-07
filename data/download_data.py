"""
Download Aksharantar transliteration dataset from HuggingFace Hub.
Extracts Hindi (hin), Bengali (ben), and Tamil (tam) subsets.

The dataset is stored as per-language zip files on HuggingFace,
each containing JSON files like {lang}_train.json, {lang}_test.json, etc.

Usage:
    python data/download_data.py
"""

import os
import json
import zipfile
import io
from pathlib import Path

import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# Languages to download
LANGUAGES = {
    "hin": "Hindi",
    "ben": "Bengali",
    "tam": "Tamil",
}

# Dataset info
REPO_ID = "ai4bharat/Aksharantar"
RAW_DIR = Path("data/raw")


def download_language(lang_code: str, lang_name: str) -> dict:
    """Download and extract a single language zip from the HF repo."""
    print(f"\n{'='*60}")
    print(f"Downloading {lang_name} ({lang_code})...")
    print(f"{'='*60}")

    lang_dir = RAW_DIR / lang_code
    lang_dir.mkdir(parents=True, exist_ok=True)

    # Download the zip file using huggingface_hub
    zip_filename = f"{lang_code}.zip"
    print(f"  Downloading {zip_filename} from {REPO_ID}...")

    zip_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=zip_filename,
        repo_type="dataset",
        local_dir=str(RAW_DIR / "downloads"),
    )

    print(f"  Downloaded to: {zip_path}")

    # Extract the zip
    print(f"  Extracting...")
    stats = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        json_files = [f for f in zf.namelist() if f.endswith(".json")]
        print(f"  Found {len(json_files)} JSON files: {json_files}")

        for json_file in json_files:
            # Determine split name from filename
            # Format: hin_train.json, hin_test.json, hin_valid.json, etc.
            basename = os.path.basename(json_file)
            # Remove language prefix and .json extension
            split_name = basename.replace(f"{lang_code}_", "").replace(".json", "")

            # Normalize split names
            if split_name in ["valid", "validation", "dev"]:
                split_name = "val"

            print(f"  Extracting {basename} → {split_name}...")

            # Read JSON content from zip
            with zf.open(json_file) as jf:
                raw_content = jf.read().decode("utf-8")

            # The JSON files may be a JSON array or JSONL
            # Try parsing as JSON array first
            try:
                records = json.loads(raw_content)
                if isinstance(records, dict):
                    # Some files have a wrapper dict
                    records = list(records.values())[0] if records else []
            except json.JSONDecodeError:
                # Try as JSONL (one JSON object per line)
                records = []
                for line in raw_content.strip().split("\n"):
                    if line.strip():
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            # Save as JSONL
            output_file = lang_dir / f"{split_name}.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for record in records:
                    json.dump(record, f, ensure_ascii=False)
                    f.write("\n")

            stats[split_name] = len(records)
            print(f"    {split_name}: {len(records):,} pairs → {output_file}")

    return stats


def main():
    """Download all language subsets."""
    print("=" * 60)
    print("Aksharantar Dataset Downloader")
    print("Languages: " + ", ".join(f"{v} ({k})" for k, v in LANGUAGES.items()))
    print("=" * 60)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for lang_code, lang_name in LANGUAGES.items():
        try:
            stats = download_language(lang_code, lang_name)
            all_stats[lang_code] = stats
        except Exception as e:
            print(f"  ERROR downloading {lang_name}: {e}")
            import traceback
            traceback.print_exc()
            all_stats[lang_code] = {"error": str(e)}

    # Save download summary
    summary_file = RAW_DIR / "download_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("Download Summary:")
    print(f"{'='*60}")
    for lang_code, stats in all_stats.items():
        lang_name = LANGUAGES[lang_code]
        if "error" in stats:
            print(f"  {lang_name}: FAILED - {stats['error']}")
        else:
            total = sum(stats.values())
            print(f"  {lang_name}: {total:,} total pairs")
            for split, count in stats.items():
                print(f"    {split}: {count:,}")

    print(f"\nSummary saved to: {summary_file}")
    print("Done!")


if __name__ == "__main__":
    main()
