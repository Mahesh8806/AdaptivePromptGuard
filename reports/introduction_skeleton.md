# Chapter 1: Introduction
**AdaptivePromptGuard — MSc Dissertation**
**Author:** Mahesh Bunage | National College of Ireland | 2025
**Status:** Draft — Week 1

---

## 1.1 Background and Motivation

Large Language Models (LLMs) have seen unprecedented adoption across industries
in recent years. Models such as GPT-4, LLaMA-2, Mistral, and Zephyr are now
deployed in customer service, education, healthcare, legal, and software
development contexts. Their ability to follow complex instructions and generate
human-quality text has made them attractive components of production systems.

However, this rapid deployment has exposed a critical vulnerability: **jailbreak
attacks**. A jailbreak attack is an adversarial prompt crafted by a user to
bypass a model's built-in safety alignment — causing the model to produce
harmful, illegal, or unethical content that it was explicitly trained to refuse.

Despite significant investment in safety alignment techniques such as
Reinforcement Learning from Human Feedback (RLHF) [Ouyang et al., 2022] and
Constitutional AI [Bai et al., 2022], state-of-the-art open-source models
remain vulnerable to a wide range of adversarial prompting strategies. Wei
et al. (2023) demonstrated that safety training failure stems from two root
causes: competing objectives between helpfulness and harmlessness, and
generalisation failures where safety behaviours learned during training do not
transfer to adversarial inputs.

Open-source models present a particular challenge. Unlike commercial APIs such
as GPT-4 and Claude, which benefit from continuous monitoring and proprietary
post-hoc filtering systems, open-source models like LLaMA-2, Mistral, and
Zephyr are deployed without such infrastructure. Users running these models
locally or integrating them into applications receive no protection beyond
whatever safety fine-tuning is baked into the model weights — fine-tuning that
has been repeatedly shown to be bypassable.

This gap motivates the present work.

---

## 1.2 Problem Statement

Despite advances in safety alignment, open-source instruction-tuned LLMs
remain susceptible to adversarial jailbreak attacks. Existing defences are
either weight-level (requiring expensive retraining), computationally
prohibitive (adversarial training), or insufficiently studied in the context of
open-source deployment.

There is a clear need for a lightweight, inference-time defence mechanism that
can be layered on top of any open-source LLM without modifying model weights,
without requiring GPU-intensive retraining, and without significantly degrading
response quality for benign users.

**Central Research Question:**
> To what extent can a two-stage prompt defence system — combining a fine-tuned
> binary classifier with adaptive prompt augmentation — reduce the Attack
> Success Rate of jailbreak prompts against open-source instruction-tuned LLMs?

---

## 1.3 Research Objectives

This study pursues four concrete, measurable research objectives:

**RO1:** Establish a quantitative baseline Attack Success Rate (ASR) for three
open-source instruction-tuned LLMs — LLaMA-2-7B-Chat, Mistral-7B-Instruct-v0.2,
and Zephyr-7B-β — against a standardised set of jailbreak prompts drawn from
the JailBreakV-28K benchmark.

**RO2:** Design, train, and evaluate a DistilBERT-based binary classifier
(APG Stage 1) capable of distinguishing jailbreak prompts from benign user
inputs, and optimise its decision threshold to balance precision and recall.

**RO3:** Develop an adaptive prompt augmentation module (APG Stage 2) that
modifies flagged prompts at inference time to reinforce model safety behaviour
without hard-blocking user input.

**RO4:** Measure the reduction in ASR achieved by the full APG pipeline
across all three models, and compare results against the undefended baseline.

---

## 1.4 Proposed Contributions

This dissertation makes the following contributions:

1. **AdaptivePromptGuard (APG):** A novel two-stage, inference-time jailbreak
   defence system for open-source LLMs, requiring no model retraining and
   no access to model weights.

2. **Empirical baseline:** Systematic measurement of jailbreak Attack Success
   Rates across three widely-used open-source 7B models under standardised
   experimental conditions.

3. **Comparative evaluation:** Rigorous before/after comparison of ASR, with
   statistical significance testing, across multiple attack categories defined
   by the JailBreakV-28K taxonomy.

4. **Reproducible codebase:** A fully open-source implementation released on
   GitHub, enabling researchers to replicate, extend, and benchmark against
   this work.

---

## 1.5 Scope and Limitations

**In scope:**
- Text-only jailbreak attacks (not multimodal)
- Three 7B parameter open-source instruction-tuned models
- Inference-time defence (no fine-tuning of LLMs)
- English-language prompts only

**Out of scope:**
- Closed-source models (GPT-4, Claude, Gemini)
- Models larger than 7B parameters
- Multimodal attacks (image + text)
- Real-time streaming inference

**Hardware constraints:**
- All experiments run locally on an NVIDIA RTX 4070 Laptop GPU (8GB VRAM)
- 4-bit quantization used throughout to fit models within VRAM budget
- This constraint is acknowledged and its potential effect on baseline ASR
  is discussed in the methodology chapter

---

## 1.6 Dissertation Structure

The remainder of this dissertation is organised as follows:

- **Chapter 2 — Literature Review:** Reviews prior work on LLM safety
  alignment, jailbreak attack taxonomies, and existing defence mechanisms.

- **Chapter 3 — Methodology:** Details the experimental design, dataset
  construction, APG architecture, training procedure, and evaluation protocol.

- **Chapter 4 — Results:** Presents quantitative findings including baseline
  ASR measurements, classifier performance metrics, and post-defence ASR
  comparisons.

- **Chapter 5 — Discussion:** Interprets results in light of the research
  objectives, compares findings to related work, and discusses implications
  for open-source LLM deployment.

- **Chapter 6 — Conclusion:** Summarises contributions, acknowledges
  limitations, and proposes directions for future work.

---

## References (to fill in)

- Ouyang et al. (2022) — InstructGPT / RLHF
- Bai et al. (2022) — Constitutional AI
- Wei et al. (2023) — Jailbroken
- Zou et al. (2023) — Universal Adversarial Attacks
- Touvron et al. (2023) — LLaMA 2
- Luo et al. (2024) — JailBreakV-28K
- Sanh et al. (2019) — DistilBERT

---
*This document is a working draft. Expand each section as research progresses.*
*Target length for final Chapter 1: 1,500 – 2,000 words.*
