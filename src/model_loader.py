# -*- coding: utf-8 -*-
"""
model_loader.py
Handles loading of quantized LLMs for APG experiments.

Supports:
  - LLaMA-2-7B-Chat
  - Mistral-7B-Instruct-v0.2
  - Zephyr-7B-beta

Quantization: 4-bit (BitsAndBytes) for 8GB VRAM budget.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from loguru import logger
from src.config import MODELS, MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE


def get_bnb_config() -> BitsAndBytesConfig:
    """
    Returns 4-bit quantization config using BitsAndBytes (NF4).
    This cuts VRAM from ~14GB to ~4-5GB per 7B model.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,   # extra memory savings
        bnb_4bit_quant_type="nf4",        # NormalFloat4 -- best for LLMs
    )


def load_model(model_key: str, quantization: str = "4bit"):
    """
    Load a tokenizer and model by key name.

    Args:
        model_key: One of 'llama2', 'mistral', 'zephyr'
        quantization: '4bit', '8bit', or 'none'

    Returns:
        tokenizer, model
    """
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key '{model_key}'. Choose from: {list(MODELS.keys())}")

    model_id = MODELS[model_key]
    logger.info(f"Loading model: {model_key} ({model_id})")
    logger.info(f"Quantization: {quantization}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # Pad token fix (LLaMA-2 doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token")

    # Quantization config
    bnb_config = None
    if quantization == "4bit":
        bnb_config = get_bnb_config()
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",           # auto-assigns to GPU
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.eval()

    # Log VRAM usage
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1e9
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"VRAM used after loading: {vram_used:.2f} / {vram_total:.2f} GB")

    logger.info(f"Model '{model_key}' loaded successfully.")
    return tokenizer, model


def build_pipeline(tokenizer, model):
    """
    Wrap tokenizer + model in a HuggingFace text-generation pipeline.
    Easier to use for inference experiments.
    """
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=DO_SAMPLE,
        repetition_penalty=1.1,
        return_full_text=False,      # only return new tokens, not the prompt
    )


def generate_response(pipe, prompt: str) -> str:
    """
    Generate a response from the LLM pipeline.

    Args:
        pipe: HuggingFace pipeline object
        prompt: Formatted prompt string

    Returns:
        Generated text response (str)
    """
    output = pipe(prompt)
    return output[0]["generated_text"].strip()


def format_prompt(model_key: str, user_message: str, system_prompt: str = None) -> str:
    """
    Format a prompt using the correct chat template for each model.

    Each model expects a different format:
      - LLaMA-2: [INST] <<SYS>> ... <</SYS>> {user} [/INST]
      - Mistral:  [INST] {user} [/INST]
      - Zephyr:   <|system|>...<|user|>...<|assistant|>

    Args:
        model_key: 'llama2', 'mistral', or 'zephyr'
        user_message: The user's input
        system_prompt: Optional system instruction

    Returns:
        Formatted prompt string
    """
    if model_key == "llama2":
        sys_block = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n" if system_prompt else ""
        return f"[INST] {sys_block}{user_message} [/INST]"

    elif model_key == "mistral":
        # Mistral does not officially use system prompts in v0.2
        prefix = f"{system_prompt}\n\n" if system_prompt else ""
        return f"[INST] {prefix}{user_message} [/INST]"

    elif model_key == "zephyr":
        sys_block = f"<|system|>\n{system_prompt}</s>\n" if system_prompt else ""
        return f"{sys_block}<|user|>\n{user_message}</s>\n<|assistant|>"

    else:
        raise ValueError(f"Unknown model_key: {model_key}")
