"""
utils.py
Shared utility functions used across the APG pipeline.
"""

import json
import random
import numpy as np
from pathlib import Path
from loguru import logger
from datetime import datetime


def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed set to {seed}")
    except ImportError:
        logger.warning("PyTorch not available — only numpy/random seeds set.")


def setup_logger(log_dir: Path = None, level: str = "INFO"):
    """Configure loguru logger with file + console output."""
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}"
    )
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"apg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(log_file, level=level, rotation="10 MB")
        logger.info(f"Logging to {log_file}")


def save_json(data: dict | list, path: Path, indent: int = 2):
    """Save dictionary or list as a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    logger.info(f"Saved JSON → {path}")


def load_json(path: Path) -> dict | list:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded JSON ← {path}")
    return data


def get_timestamp() -> str:
    """Return a filesystem-safe timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def count_parameters(model) -> str:
    """Return human-readable parameter count for a PyTorch model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total:,} | Trainable: {trainable:,}"
