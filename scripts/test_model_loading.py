# -*- coding: utf-8 -*-
"""
test_model_loading.py

Week 2 script: Load one model at a time and run a quick inference test.
This verifies the model loads within VRAM budget and responds correctly.

Usage:
    python scripts/test_model_loading.py --model mistral
    python scripts/test_model_loading.py --model zephyr
    python scripts/test_model_loading.py --model llama2
"""

import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.model_loader import load_model, build_pipeline, generate_response, format_prompt
from src.utils import setup_logger, set_seed, get_timestamp, save_json
from src.config import DATA_RESULTS, SYSTEM_PROMPT_DEFAULT


# Simple test prompts -- benign, nothing harmful
TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain what a neural network is in two sentences.",
    "Write a short poem about the ocean.",
]


def get_vram_stats() -> dict:
    if torch.cuda.is_available():
        return {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
            "reserved_gb":  round(torch.cuda.memory_reserved() / 1e9, 3),
            "total_gb":     round(torch.cuda.get_device_properties(0).total_memory / 1e9, 3),
        }
    return {"allocated_gb": 0, "reserved_gb": 0, "total_gb": 0}


def run_inference_test(model_key: str, quantization: str = "4bit"):
    """Load model and run inference on test prompts. Save results."""

    setup_logger()
    set_seed(42)

    logger.info("=" * 55)
    logger.info(f"  Week 2 -- Model Load Test: {model_key.upper()}")
    logger.info("=" * 55)

    # Load
    logger.info("Loading tokenizer and model...")
    tokenizer, model = load_model(model_key, quantization=quantization)
    pipe = build_pipeline(tokenizer, model)

    vram_after_load = get_vram_stats()
    logger.info(f"VRAM after load: {vram_after_load}")

    results = []

    # Run each test prompt
    for i, prompt_text in enumerate(TEST_PROMPTS, 1):
        logger.info(f"\nTest {i}/{len(TEST_PROMPTS)}: {prompt_text}")

        formatted = format_prompt(
            model_key=model_key,
            user_message=prompt_text,
            system_prompt=SYSTEM_PROMPT_DEFAULT,
        )

        response = generate_response(pipe, formatted)

        logger.info(f"Response: {response[:200]}")  # print first 200 chars

        results.append({
            "model":         model_key,
            "quantization":  quantization,
            "prompt":        prompt_text,
            "response":      response,
            "vram_stats":    get_vram_stats(),
        })

    # Save results
    output_path = DATA_RESULTS / f"week2_load_test_{model_key}_{get_timestamp()}.json"
    save_json(results, output_path)

    logger.info("\n" + "=" * 55)
    logger.info(f"  Test complete. Results saved to:")
    logger.info(f"  {output_path}")
    logger.info("=" * 55)

    return results


def main():
    parser = argparse.ArgumentParser(description="Week 2 -- Model Load Test")
    parser.add_argument(
        "--model",
        type=str,
        choices=["llama2", "mistral", "zephyr"],
        default="mistral",
        help="Which model to test (default: mistral)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["4bit", "8bit", "none"],
        default="4bit",
        help="Quantization level (default: 4bit)",
    )
    args = parser.parse_args()

    run_inference_test(args.model, args.quantization)


if __name__ == "__main__":
    main()
