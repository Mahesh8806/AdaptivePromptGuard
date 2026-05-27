"""
pipeline.py
Full AdaptivePromptGuard (APG) pipeline.
Integrates Stage 1 (classifier) + Stage 2 (prompt augmentation).
Full integration in Week 8. Skeleton created in Week 1.
"""

from loguru import logger
from src.classifier import JailbreakClassifier
from src.prompt_guard import PromptGuard
from src.config import CONFIDENCE_THRESHOLD


class APGPipeline:
    """
    AdaptivePromptGuard — Full Two-Stage Pipeline.

    Stage 1: Classify input as benign or jailbreak.
    Stage 2: If jailbreak detected, augment prompt before passing to LLM.

    Usage:
        apg = APGPipeline()
        apg.load()
        output = apg.run("How do I hack a website?", llm_model)
    """

    def __init__(self, threshold: float = CONFIDENCE_THRESHOLD, strategy: str = "system_prompt"):
        self.classifier   = JailbreakClassifier(threshold=threshold)
        self.prompt_guard = PromptGuard(strategy=strategy)
        self.threshold    = threshold

    def load(self):
        """Load all components."""
        logger.info("Loading APG pipeline components...")
        self.classifier.load()
        logger.info("APG pipeline ready.")

    def run(self, prompt: str, llm_generate_fn) -> dict:
        """
        Run the full APG pipeline on a user prompt.

        Args:
            prompt: Raw user input
            llm_generate_fn: Callable that takes a prompt string and returns LLM response

        Returns:
            dict with keys:
                'original_prompt', 'augmented_prompt', 'classification',
                'confidence', 'defended', 'response'
        """
        # Stage 1 — Classify
        result = self.classifier.predict(prompt)
        label      = result["label"]
        confidence = result["confidence"]

        logger.info(f"Stage 1 → {label} (confidence: {confidence:.3f})")

        defended        = False
        augmented_prompt = prompt

        # Stage 2 — Augment if jailbreak detected
        if label == "jailbreak" and confidence >= self.threshold:
            augmented_prompt = self.prompt_guard.augment(prompt)
            defended         = True
            logger.info(f"Stage 2 → Prompt augmented (strategy: {self.prompt_guard.strategy})")

        # Generate response from LLM
        response = llm_generate_fn(augmented_prompt)

        return {
            "original_prompt":  prompt,
            "augmented_prompt": augmented_prompt,
            "classification":   label,
            "confidence":       confidence,
            "defended":         defended,
            "response":         response,
        }
