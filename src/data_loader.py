"""
data_loader.py
Dataset loading and preprocessing utilities for APG.
Will be populated fully in Week 4 — skeleton created in Week 1.
"""

from pathlib import Path
from loguru import logger
from src.config import DATA_RAW, DATA_PROCESSED


def load_jailbreakv(split: str = "train") -> list[dict]:
    """
    Load JailBreakV-28K dataset from local raw data folder.
    Expected file: data/raw/jailbreakv_28k.jsonl

    Args:
        split: "train", "val", or "test"

    Returns:
        List of dicts with keys: 'prompt', 'label', 'category'
    """
    path = DATA_RAW / "jailbreakv_28k.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"JailBreakV dataset not found at {path}. "
            "Download it in Week 4 with scripts/download_datasets.py"
        )
    # Placeholder — full implementation in Week 4
    logger.info(f"Loading JailBreakV ({split}) from {path}")
    return []


def load_alpaca(split: str = "train") -> list[dict]:
    """
    Load Stanford Alpaca dataset (benign samples).
    Expected file: data/raw/alpaca_data.json

    Args:
        split: "train", "val", or "test"

    Returns:
        List of dicts with keys: 'prompt', 'label', 'category'
    """
    path = DATA_RAW / "alpaca_data.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Alpaca dataset not found at {path}. "
            "Download it in Week 4 with scripts/download_datasets.py"
        )
    logger.info(f"Loading Alpaca ({split}) from {path}")
    return []


def load_combined_dataset(split: str = "train") -> list[dict]:
    """
    Load and merge jailbreak + benign samples into one dataset.

    Returns:
        Combined list of dicts with 'prompt', 'label' (0=benign, 1=jailbreak)
    """
    jailbreak_samples = load_jailbreakv(split)
    benign_samples    = load_alpaca(split)
    combined          = jailbreak_samples + benign_samples
    logger.info(f"Combined dataset size ({split}): {len(combined)}")
    return combined
