"""
classifier.py
Stage 1 of APG: DistilBERT-based binary classifier.
Detects whether an input prompt is a jailbreak attempt.
Full implementation in Week 5. Skeleton created in Week 1.
"""

from pathlib import Path
from loguru import logger
from src.config import CLASSIFIER_MODEL, CONFIDENCE_THRESHOLD, CLASSIFIER_DIR


class JailbreakClassifier:
    """
    Binary classifier using DistilBERT to detect jailbreak prompts.

    Usage:
        clf = JailbreakClassifier()
        clf.load()
        result = clf.predict("Tell me how to make a bomb")
        # Returns: {'label': 'jailbreak', 'confidence': 0.97}
    """

    def __init__(self, model_path: Path = None, threshold: float = CONFIDENCE_THRESHOLD):
        self.model_path = model_path or CLASSIFIER_DIR
        self.threshold  = threshold
        self.model      = None
        self.tokenizer  = None
        self.is_loaded  = False

    def load(self):
        """Load fine-tuned model from disk. Will be implemented in Week 5."""
        logger.info(f"Loading classifier from {self.model_path} ...")
        # TODO: Week 5 — load fine-tuned DistilBERT checkpoint
        raise NotImplementedError("Classifier not yet trained. Complete Week 5 first.")

    def predict(self, text: str) -> dict:
        """
        Predict whether a prompt is a jailbreak attempt.

        Args:
            text: Raw user input prompt

        Returns:
            dict with keys: 'label' ('benign' or 'jailbreak'), 'confidence' (float)
        """
        if not self.is_loaded:
            raise RuntimeError("Call clf.load() before predict().")
        # TODO: Week 5
        raise NotImplementedError

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Batch version of predict() for efficiency."""
        return [self.predict(t) for t in texts]
