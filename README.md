# AdaptivePromptGuard (APG)

> A two-stage jailbreak defense system for open-source instruction-tuned LLMs.

## Overview

**AdaptivePromptGuard (APG)** is a modular, lightweight defense pipeline that protects open-source LLMs against adversarial jailbreak attacks without modifying model weights.

| Stage | Component | Role |
|-------|-----------|------|
| Stage 1 | DistilBERT Classifier | Detect jailbreak prompts in real-time |
| Stage 2 | Prompt Augmentation | Neutralize detected threats before LLM inference |

## Models Evaluated

| Model | Size | Quantization |
|-------|------|-------------|
| LLaMA-2-7B-Chat | 7B | 4-bit (bitsandbytes) |
| Mistral-7B-Instruct-v0.2 | 7B | 4-bit (bitsandbytes) |
| Zephyr-7B-β | 7B | 4-bit (bitsandbytes) |

## Key Metrics

- **Attack Success Rate (ASR)** — baseline vs. defended
- **F1, Precision, Recall** — classifier performance
- **AUC-ROC** — classifier discrimination ability
- **ASR Reduction %** — APG effectiveness

## Project Structure

```
AdaptivePromptGuard/
├── data/               # Datasets (raw, processed, results)
├── models/             # LLM weights + classifier checkpoints
├── notebooks/          # Experiment notebooks
├── src/                # Core source code
│   ├── config.py       # Central configuration
│   ├── data_loader.py  # Dataset utilities
│   ├── classifier.py   # Stage 1 — DistilBERT
│   ├── prompt_guard.py # Stage 2 — Prompt augmentation
│   ├── pipeline.py     # Full APG integration
│   ├── evaluator.py    # Metrics computation
│   └── utils.py        # Shared utilities
├── scripts/            # Runnable experiment scripts
├── reports/            # Literature notes, weekly logs, figures
├── tests/              # Unit tests
└── requirements.txt
```

## Setup

```bash
# 1. Create virtual environment
python -m venv apg_env
source apg_env/bin/activate       # Linux/Mac
.\apg_env\Scripts\Activate.ps1   # Windows PowerShell

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify setup
python scripts/verify_setup.py
```

## Research Context

| Field | Detail |
|-------|--------|
| Degree | MSc Artificial Intelligence |
| Institution | National College of Ireland |
| Year | 2025 |
| Author | Mahesh Bunage |

## Citation

If you use this work, please cite:
```
Bunage, M. (2025). Reducing Jailbreak Vulnerability in Open-Source
Instruction-Tuned LLMs using Prompt Defense Mechanisms: A Comparative Study.
MSc Dissertation, National College of Ireland.
```
