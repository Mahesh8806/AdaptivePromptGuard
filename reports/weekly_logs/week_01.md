# Week 1 Log — Environment Setup & Literature Review

**Date:** May 2025
**Status:** 🔄 In Progress
**Week Theme:** Build the foundation

---

## Tasks Completed

### Environment
- [x] Project folder structure created locally (14 folders)
- [x] Python virtual environment (apg_env) created and activated
- [x] requirements.txt — all 14 packages installed and verified
- [x] verify_setup.py — passes all checks (14/14 packages, GPU detected)
- [x] GPU confirmed: NVIDIA RTX 4070 Laptop — 8.59GB VRAM — CUDA 12.6
- [x] PyTorch 2.12.0+cu126 — CUDA build installed

### Code Files Created
- [x] src/config.py       — central config (paths, model names, hyperparams)
- [x] src/utils.py        — seed, logger, save/load JSON, helpers
- [x] src/data_loader.py  — skeleton (implemented Week 4)
- [x] src/classifier.py   — skeleton (implemented Week 5)
- [x] src/prompt_guard.py — skeleton (implemented Week 7)
- [x] src/pipeline.py     — skeleton (implemented Week 8)
- [x] src/evaluator.py    — ASR, F1, AUC-ROC metric functions
- [x] src/model_loader.py — 4-bit quantized LLM loader
- [x] scripts/verify_setup.py — environment health check
- [x] tests/test_pipeline.py  — placeholder unit tests

### Project Files
- [x] requirements.txt   — all dependencies pinned
- [x] README.md          — project overview and setup instructions
- [x] .gitignore         — excludes venv, __pycache__, model weights, raw data
- [x] reports/literature_notes.md — 9 papers pre-loaded with templates

### Git
- [x] Git repository initialized
- [x] First commit made (18 files, 1541 insertions)
- [ ] GitHub remote repo created
- [ ] Pushed to GitHub

### Report
- [x] Introduction skeleton drafted (reports/introduction_skeleton.md)
      Sections: Background, Problem Statement, ROs, Contributions,
      Scope, Dissertation Structure

---

## Tasks Remaining

### Literature Review (Manual reading tasks)
- [ ] Wei et al. (2023) — Jailbroken — READ + annotate
- [ ] Zou et al. (2023) — GCG Attack — READ + annotate
- [ ] Touvron et al. (2023) — LLaMA 2 — READ + annotate
- [ ] Bai et al. (2022) — Constitutional AI — READ + annotate
- [ ] Luo et al. (2024) — JailBreakV-28K — READ + annotate
- [ ] Ouyang et al. (2022) — InstructGPT — READ + annotate
- [ ] Liu et al. (2023) — Prompt Injection — READ + annotate
- [ ] Sanh et al. (2019) — DistilBERT — READ + annotate
- [ ] Ganguli et al. (2022) — Red Teaming — READ + annotate

### GitHub
- [ ] Create GitHub repo: AdaptivePromptGuard
- [ ] git remote add origin <url>
- [ ] git push -u origin master

---

## Hardware Summary

| Item          | Detail                          |
|---------------|---------------------------------|
| GPU           | NVIDIA GeForce RTX 4070 Laptop  |
| VRAM          | 8.59 GB                         |
| CUDA          | 12.6                            |
| PyTorch       | 2.12.0+cu126                    |
| Python        | 3.14.0                          |
| OS            | Windows 11                      |
| Quantization  | 4-bit (NF4) — fits 7B models    |

---

## Key Learnings This Week

- 4-bit quantization (NF4) reduces 7B model VRAM from ~14GB to ~4-5GB
- Each model uses a different chat prompt format — must format correctly
- DistilBERT is the right classifier choice: 97% of BERT perf, 40% smaller
- Jailbreak success (ASR) is the core metric — everything is measured against it
- APG augments prompts, does NOT hard-block — more research value

---

## Next Week (Week 2)

- Login to HuggingFace CLI
- Request LLaMA-2 gated access on HuggingFace
- Download Mistral-7B-Instruct (start here — open access)
- Download Zephyr-7B-beta
- Run test_model_loading.py for each model
- Record VRAM usage and load times
- Fill in results table in notebook 01_model_loading_week2.ipynb
