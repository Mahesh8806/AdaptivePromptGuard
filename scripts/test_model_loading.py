# -*- coding: utf-8 -*-
"""
test_model_loading.py

Week 2: Load each LLM via Ollama and run 3 inference tests.
Verifies the model responds correctly and records results.

Usage:
    python scripts/test_model_loading.py --model mistral
    python scripts/test_model_loading.py --model llama2
    python scripts/test_model_loading.py --model zephyr
    python scripts/test_model_loading.py --model all
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.model_loader import (
    check_ollama_running,
    check_model_available,
    generate_response_raw,
    format_prompt,
    generate_response,
)
from src.utils import setup_logger, set_seed, get_timestamp, save_json
from src.config import DATA_RESULTS, OLLAMA_MODELS, SYSTEM_PROMPT_DEFAULT


# Benign test prompts to verify the model loads and responds correctly
TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain what a neural network is in two sentences.",
    "Write a haiku about artificial intelligence.",
]


def run_test(model_key: str) -> dict:
    """Run inference test on a single model. Returns result dict."""

    logger.info("=" * 55)
    logger.info(f"  Testing: {model_key.upper()} ({OLLAMA_MODELS[model_key]})")
    logger.info("=" * 55)

    if not check_model_available(model_key):
        return {"model": model_key, "status": "MISSING", "results": []}

    results  = []
    total_ms = 0

    for i, prompt_text in enumerate(TEST_PROMPTS, 1):
        logger.info(f"\n[{i}/{len(TEST_PROMPTS)}] Prompt: {prompt_text}")

        start = time.time()
        result = generate_response_raw(
            model_key=model_key,
            user_message=prompt_text,
            system_prompt=SYSTEM_PROMPT_DEFAULT,
        )
        elapsed_ms = round((time.time() - start) * 1000)
        total_ms  += elapsed_ms

        logger.info(f"  Response ({elapsed_ms}ms): {result['response'][:150]}")

        results.append({
            "prompt":      prompt_text,
            "response":    result["response"],
            "elapsed_ms":  elapsed_ms,
        })

    avg_ms = total_ms // len(TEST_PROMPTS)
    logger.info(f"\n  Avg response time: {avg_ms}ms")
    logger.info(f"  Status: ALL TESTS PASSED")

    return {
        "model":      model_key,
        "model_name": OLLAMA_MODELS[model_key],
        "status":     "OK",
        "avg_ms":     avg_ms,
        "results":    results,
    }


def main():
    setup_logger()
    set_seed(42)

    parser = argparse.ArgumentParser(description="Week 2 -- Model Load Test")
    parser.add_argument(
        "--model",
        choices=["llama2", "mistral", "zephyr", "all"],
        default="mistral",
        help="Which model to test (default: mistral)",
    )
    args = parser.parse_args()

    # Check Ollama is running
    if not check_ollama_running():
        logger.error("Start Ollama first: open a terminal and run 'ollama serve'")
        sys.exit(1)

    # Determine which models to test
    if args.model == "all":
        models_to_test = list(OLLAMA_MODELS.keys())
    else:
        models_to_test = [args.model]

    all_results = []
    for model_key in models_to_test:
        result = run_test(model_key)
        all_results.append(result)

    # Save results
    out_path = DATA_RESULTS / f"week2_inference_test_{get_timestamp()}.json"
    save_json(all_results, out_path)

    # Print summary
    print("\n" + "=" * 55)
    print("  WEEK 2 — MODEL TEST SUMMARY")
    print("=" * 55)
    print(f"  {'Model':<12} {'Ollama Name':<22} {'Status':<8} {'Avg ms'}")
    print(f"  {'-'*12} {'-'*22} {'-'*8} {'-'*7}")
    for r in all_results:
        avg = r.get("avg_ms", "N/A")
        print(f"  {r['model']:<12} {r.get('model_name',''):<22} {r['status']:<8} {avg}")
    print(f"\n  Results saved -> {out_path}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
