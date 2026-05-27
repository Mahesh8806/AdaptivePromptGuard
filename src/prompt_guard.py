"""
prompt_guard.py
Stage 2 of APG: Adaptive Prompt Augmentation.
Modifies flagged prompts to neutralize jailbreak attempts.
Full implementation in Week 7. Skeleton created in Week 1.
"""

from loguru import logger
from src.config import SYSTEM_PROMPT_SAFE, SYSTEM_PROMPT_DEFAULT


class PromptGuard:
    """
    Augments prompts detected as jailbreak attempts (Stage 2 of APG).

    Strategies:
        - Prepend safety system prompt
        - Wrap prompt in refusal-inducing context
        - Add explicit constraint reminders
    """

    def __init__(self, strategy: str = "system_prompt"):
        """
        Args:
            strategy: Augmentation strategy.
                      Options: "system_prompt", "wrap", "constraint"
        """
        self.strategy = strategy
        logger.info(f"PromptGuard initialized with strategy: {strategy}")

    def augment(self, prompt: str) -> str:
        """
        Augment a flagged jailbreak prompt.

        Args:
            prompt: Original user prompt detected as jailbreak

        Returns:
            Augmented prompt string passed to LLM
        """
        if self.strategy == "system_prompt":
            return self._prepend_system_prompt(prompt)
        elif self.strategy == "wrap":
            return self._wrap_with_context(prompt)
        elif self.strategy == "constraint":
            return self._add_constraints(prompt)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _prepend_system_prompt(self, prompt: str) -> str:
        """Prepend safety-oriented system prompt."""
        return f"{SYSTEM_PROMPT_SAFE}\n\nUser: {prompt}"

    def _wrap_with_context(self, prompt: str) -> str:
        """Wrap prompt in explicit safety context."""
        return (
            f"[SAFETY CONTEXT: The following request has been flagged. "
            f"Respond only if the request is clearly benign.]\n\n{prompt}"
        )

    def _add_constraints(self, prompt: str) -> str:
        """Add explicit behavioural constraints to the prompt."""
        constraints = (
            "\n\n[CONSTRAINTS: Do not produce harmful, illegal, or unethical content. "
            "Refuse if this request violates your safety guidelines.]"
        )
        return prompt + constraints
