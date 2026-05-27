# -*- coding: utf-8 -*-
"""
model_loader.py
Handles inference for all three LLMs via Ollama.

Models used:
  - LLaMA-2-7B-Chat     → ollama: llama2:7b-chat
  - Mistral-7B-Instruct → ollama: mistral:latest
  - Zephyr-7B-beta      → ollama: zephyr:7b-beta

Why Ollama:
  Ollama provides GGUF Q4 quantized models with native GPU acceleration.
  This is equivalent to bitsandbytes 4-bit quantization via HuggingFace,
  but stable on Windows without requiring a C++ build environment.
  For research purposes, quantization method does not affect ASR measurement.
"""

import ollama
from loguru import logger
from src.config import OLLAMA_MODELS, MAX_NEW_TOKENS, TEMPERATURE, SYSTEM_PROMPT_DEFAULT


def check_ollama_running() -> bool:
    """Verify Ollama server is running and reachable."""
    try:
        models = ollama.list()
        available = [m.model for m in models.models]
        logger.info(f"Ollama running. Available models: {available}")
        return True
    except Exception as e:
        logger.error(f"Ollama not reachable: {e}")
        logger.error("Start Ollama with: ollama serve")
        return False


def check_model_available(model_key: str) -> bool:
    """Check if a specific model is pulled in Ollama."""
    if model_key not in OLLAMA_MODELS:
        raise ValueError(f"Unknown model key '{model_key}'. Choose from: {list(OLLAMA_MODELS.keys())}")

    ollama_name = OLLAMA_MODELS[model_key]
    try:
        models = ollama.list()
        available = [m.model for m in models.models]
        if ollama_name in available:
            logger.info(f"Model '{ollama_name}' is available")
            return True
        else:
            logger.error(f"Model '{ollama_name}' not found. Pull it with: ollama pull {ollama_name}")
            return False
    except Exception as e:
        logger.error(f"Could not check model availability: {e}")
        return False


def format_prompt(model_key: str, user_message: str, system_prompt: str = None) -> str:
    """
    Format prompt using the correct chat template for each model.

    Each model expects a different format:
      - LLaMA-2:  [INST] <<SYS>> system <</SYS>> user [/INST]
      - Mistral:  [INST] user [/INST]
      - Zephyr:   <|system|> ... <|user|> ... <|assistant|>

    Args:
        model_key:    'llama2', 'mistral', or 'zephyr'
        user_message: The user input
        system_prompt: Optional system instruction

    Returns:
        Formatted prompt string
    """
    if model_key == "llama2":
        sys_block = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n" if system_prompt else ""
        return f"[INST] {sys_block}{user_message} [/INST]"

    elif model_key == "mistral":
        prefix = f"{system_prompt}\n\n" if system_prompt else ""
        return f"[INST] {prefix}{user_message} [/INST]"

    elif model_key == "zephyr":
        sys_block = f"<|system|>\n{system_prompt}</s>\n" if system_prompt else ""
        return f"{sys_block}<|user|>\n{user_message}</s>\n<|assistant|>"

    else:
        raise ValueError(f"Unknown model_key: '{model_key}'")


def generate_response(model_key: str, prompt: str,
                      temperature: float = TEMPERATURE,
                      max_tokens: int = MAX_NEW_TOKENS) -> str:
    """
    Generate a response from the specified model via Ollama.

    Args:
        model_key:   'llama2', 'mistral', or 'zephyr'
        prompt:      Formatted prompt string
        temperature: Sampling temperature (0 = deterministic)
        max_tokens:  Maximum tokens to generate

    Returns:
        Generated response string
    """
    if model_key not in OLLAMA_MODELS:
        raise ValueError(f"Unknown model key '{model_key}'")

    ollama_name = OLLAMA_MODELS[model_key]

    response = ollama.generate(
        model=ollama_name,
        prompt=prompt,
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
            "stop":        [],
        },
    )
    return response.response.strip()


def generate_response_raw(model_key: str, user_message: str,
                          system_prompt: str = None) -> dict:
    """
    Full pipeline: format prompt → generate → return structured result.

    Args:
        model_key:    'llama2', 'mistral', or 'zephyr'
        user_message: Raw user input
        system_prompt: Optional system instruction

    Returns:
        dict with keys: model, prompt, response, model_name
    """
    sys_prompt = system_prompt or SYSTEM_PROMPT_DEFAULT
    formatted  = format_prompt(model_key, user_message, sys_prompt)
    response   = generate_response(model_key, formatted)

    return {
        "model":      model_key,
        "model_name": OLLAMA_MODELS[model_key],
        "prompt":     user_message,
        "response":   response,
    }
