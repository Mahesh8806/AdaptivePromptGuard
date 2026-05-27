# -*- coding: utf-8 -*-
"""
download_datasets.py

Downloads and saves all datasets needed for the APG project:
  1. JailBreakV-28K  -- jailbreak prompts  (label = 1)
  2. Stanford Alpaca -- benign prompts     (label = 0)

Usage:
    python scripts/download_datasets.py

Saves to:
    data/raw/jailbreakv_28k.jsonl
    data/raw/alpaca_data.jsonl
    data/raw/dataset_stats.json
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from tqdm import tqdm
from loguru import logger
from src.utils import setup_logger, save_json
from src.config import DATA_RAW


def download_jailbreakv() -> list[dict]:
    """
    Download JailBreakV-28K from HuggingFace.
    Source: https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k
    Paper:  https://arxiv.org/abs/2404.03027

    Returns list of dicts with keys: prompt, label, category
    """
    logger.info("Downloading JailBreakV-28K dataset...")
    logger.info("Source: JailbreakV-28K/JailBreakV-28k on HuggingFace")

    dataset = load_dataset("JailbreakV-28K/JailBreakV-28k", "JailBreakV_28K", split="JailBreakV_28K")

    logger.info(f"Raw dataset size: {len(dataset)} samples")
    logger.info(f"Columns available: {dataset.column_names}")

    samples = []
    for row in tqdm(dataset, desc="Processing JailBreakV"):
        # 'jailbreak_query' is the main text prompt field in this dataset
        prompt = row.get("jailbreak_query", "")
        if not prompt or len(str(prompt).strip()) < 10:
            continue

        category = row.get("format", "unknown")

        samples.append({
            "prompt":   str(prompt).strip(),
            "label":    1,              # 1 = jailbreak
            "category": str(category),
            "source":   "jailbreakv_28k",
        })

    logger.info(f"JailBreakV samples extracted: {len(samples)}")
    return samples


def download_alpaca() -> list[dict]:
    """
    Download Stanford Alpaca dataset from HuggingFace.
    Source: https://huggingface.co/datasets/tatsu-lab/alpaca
    Paper:  https://arxiv.org/abs/2212.10560

    We use the 'instruction' field as the prompt.
    Returns list of dicts with keys: prompt, label, category
    """
    logger.info("Downloading Stanford Alpaca dataset...")
    logger.info("Source: tatsu-lab/alpaca on HuggingFace")

    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    logger.info(f"Raw dataset size: {len(dataset)} samples")

    samples = []
    for row in tqdm(dataset, desc="Processing Alpaca"):
        instruction = row.get("instruction", "").strip()
        input_text  = row.get("input", "").strip()

        # Combine instruction + input if both present
        if input_text:
            prompt = f"{instruction}\n{input_text}"
        else:
            prompt = instruction

        if not prompt or len(prompt.strip()) < 5:
            continue

        samples.append({
            "prompt":   prompt.strip(),
            "label":    0,              # 0 = benign
            "category": "benign",
            "source":   "stanford_alpaca",
        })

    logger.info(f"Alpaca samples extracted: {len(samples)}")
    return samples


def save_jsonl(samples: list[dict], path: Path):
    """Save list of dicts as JSONL (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(samples)} samples -> {path}")


def print_stats(jailbreak: list, benign: list):
    """Print dataset statistics."""
    total = len(jailbreak) + len(benign)

    # Category breakdown for jailbreak samples
    categories = {}
    for s in jailbreak:
        cat = s.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print("\n" + "=" * 55)
    print("  Dataset Download Summary")
    print("=" * 55)
    print(f"  Jailbreak samples  : {len(jailbreak):>7,}  (label=1)")
    print(f"  Benign samples     : {len(benign):>7,}  (label=0)")
    print(f"  Total              : {total:>7,}")
    print(f"\n  Class balance      : {len(jailbreak)/total*100:.1f}% jailbreak")
    print(f"                       {len(benign)/total*100:.1f}% benign")
    print(f"\n  Jailbreak categories:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:10]:
        print(f"    {cat:<35} {count:>5}")
    print("=" * 55 + "\n")

    return {
        "jailbreak_count": len(jailbreak),
        "benign_count":    len(benign),
        "total":           total,
        "jailbreak_ratio": round(len(jailbreak) / total, 4),
        "categories":      categories,
    }


def main():
    setup_logger()
    logger.info("=" * 55)
    logger.info("  APG Dataset Downloader")
    logger.info("=" * 55)

    # Download
    jailbreak_samples = download_jailbreakv()
    benign_samples    = download_alpaca()

    # Save raw files
    save_jsonl(jailbreak_samples, DATA_RAW / "jailbreakv_28k.jsonl")
    save_jsonl(benign_samples,    DATA_RAW / "alpaca_data.jsonl")

    # Stats
    stats = print_stats(jailbreak_samples, benign_samples)
    save_json(stats, DATA_RAW / "dataset_stats.json")

    logger.info("All datasets downloaded and saved to data/raw/")
    logger.info("Next step: run scripts/prepare_dataset.py to split into train/val/test")


if __name__ == "__main__":
    main()
