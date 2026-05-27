# Literature Review Notes — AdaptivePromptGuard

**Author:** Mahesh Bunage  
**Last Updated:** May 2025  

---

## Paper Template

```
### [Paper Title]
- **Authors:**
- **Year:**
- **Venue:**
- **Link:**
- **Core Claim:**
- **Method:**
- **Key Findings:**
- **Limitations:**
- **Relevance to APG:**
- **My Critical Note:**
```

---

## 🔴 Tier 1 — Core Papers (Read in Full)

### Jailbroken: How Does LLM Safety Training Fail?
- **Authors:** Wei et al.
- **Year:** 2023
- **Venue:** NeurIPS 2023
- **Link:** https://arxiv.org/abs/2307.02483
- **Core Claim:** Safety training fails due to competing objectives and mismatched generalization.
- **Method:** Taxonomy of jailbreak strategies tested empirically on GPT-4, Claude, etc.
- **Key Findings:** Two root causes identified — (1) competing objectives between helpfulness and safety, (2) safety training that doesn't generalize to adversarial inputs.
- **Limitations:** Only evaluates closed-source models. No defense mechanism proposed.
- **Relevance to APG:** Motivates the need for an external defense layer (APG). Their attack taxonomy maps directly to our test dataset categories.
- **My Critical Note:** TODO — fill after reading.

---

### Universal and Transferable Adversarial Attacks on Aligned Language Models
- **Authors:** Zou et al.
- **Year:** 2023
- **Venue:** arXiv
- **Link:** https://arxiv.org/abs/2307.15043
- **Core Claim:** Gradient-based adversarial suffixes can jailbreak aligned LLMs reliably.
- **Method:** GCG (Greedy Coordinate Gradient) attack to generate transferable adversarial suffixes.
- **Key Findings:** Single adversarial suffix transfers across models including GPT-3.5, GPT-4, Claude.
- **Limitations:** Computationally expensive. Requires white-box access for generation.
- **Relevance to APG:** Represents a strong attack baseline. Our classifier must detect GCG-style token manipulation.
- **My Critical Note:** TODO — fill after reading.

---

### Llama 2: Open Foundation and Fine-Tuned Chat Models
- **Authors:** Touvron et al.
- **Year:** 2023
- **Venue:** arXiv / Meta AI
- **Link:** https://arxiv.org/abs/2307.09288
- **Core Claim:** LLaMA 2 uses RLHF + safety fine-tuning to create safe chat models.
- **Method:** Two-stage training: supervised fine-tuning followed by RLHF with safety reward models.
- **Key Findings:** Despite safety training, models remain vulnerable to certain adversarial prompts.
- **Limitations:** Safety measures are baked into weights — hard to update without retraining.
- **Relevance to APG:** This is one of our three test models. Understanding its safety architecture is essential.
- **My Critical Note:** TODO — fill after reading.

---

### Constitutional AI: Harmlessness from AI Feedback
- **Authors:** Bai et al.
- **Year:** 2022
- **Venue:** arXiv / Anthropic
- **Link:** https://arxiv.org/abs/2212.08073
- **Core Claim:** LLMs can be trained to be harmless using a set of constitutional principles + AI feedback.
- **Method:** Two phases: (1) supervised learning from AI critiques, (2) RL from AI feedback (RLAIF).
- **Key Findings:** Constitutional AI achieves harmlessness comparable to RLHF without human red-teaming.
- **Limitations:** Requires large-scale compute. Principles must be carefully designed.
- **Relevance to APG:** Contrasts with our approach — APG is an inference-time external defense, not a training-time solution.
- **My Critical Note:** TODO — fill after reading.

---

### JailBreakV-28K: A Benchmark for Assessing the Robustness of MultiModal Large Language Models Against Jailbreak Attacks
- **Authors:** Luo et al.
- **Year:** 2024
- **Venue:** arXiv
- **Link:** https://arxiv.org/abs/2404.03027
- **Core Claim:** Introduces a large-scale benchmark of 28K jailbreak samples across multiple modalities.
- **Method:** Collected and categorized jailbreak prompts; tested on 10+ open/closed source LLMs.
- **Key Findings:** Text-based jailbreaks remain highly effective even on safety-trained models.
- **Limitations:** Primarily multimodal; text-only subset used in our work.
- **Relevance to APG:** Primary source of jailbreak samples for our Stage 1 classifier training dataset.
- **My Critical Note:** TODO — fill after reading.

---

## 🟡 Tier 2 — Background Papers (Abstract + Method)

### Training language models to follow instructions with human feedback (InstructGPT)
- **Authors:** Ouyang et al.
- **Year:** 2022
- **Venue:** NeurIPS 2022
- **Link:** https://arxiv.org/abs/2203.02155
- **Core Claim:** RLHF significantly improves instruction following and safety.
- **Relevance to APG:** Foundational paper for understanding why safety training exists and its limits.
- **My Critical Note:** TODO

---

### Prompt Injection Attacks and Defenses in LLM-Integrated Applications
- **Authors:** Liu et al.
- **Year:** 2023
- **Venue:** arXiv
- **Link:** https://arxiv.org/abs/2310.12815
- **Core Claim:** Prompt injection is a major attack surface in LLM applications.
- **Relevance to APG:** Prompt injection is a category in our jailbreak taxonomy; relevant to Stage 2 augmentation design.
- **My Critical Note:** TODO

---

### DistilBERT: a distilled version of BERT, smaller, faster, cheaper and lighter
- **Authors:** Sanh et al.
- **Year:** 2019
- **Venue:** NeurIPS 2019 Workshop
- **Link:** https://arxiv.org/abs/1910.01108
- **Core Claim:** DistilBERT retains 97% of BERT's performance with 40% fewer parameters.
- **Relevance to APG:** This is our Stage 1 classifier backbone. Must understand its architecture.
- **My Critical Note:** TODO

---

### Red Teaming Language Models to Reduce Harms
- **Authors:** Ganguli et al.
- **Year:** 2022
- **Venue:** arXiv / Anthropic
- **Link:** https://arxiv.org/abs/2209.07858
- **Core Claim:** Systematic red teaming reveals consistent failure modes in LLM safety.
- **Relevance to APG:** Provides attack methodology context; our jailbreak test suite mirrors their red team approach.
- **My Critical Note:** TODO

---

## 📌 Reading Progress

| Paper | Read | Notes Done |
|-------|------|------------|
| Wei et al. (2023) — Jailbroken | ☐ | ☐ |
| Zou et al. (2023) — GCG Attack | ☐ | ☐ |
| Touvron et al. (2023) — LLaMA 2 | ☐ | ☐ |
| Bai et al. (2022) — Constitutional AI | ☐ | ☐ |
| Luo et al. (2024) — JailBreakV-28K | ☐ | ☐ |
| Ouyang et al. (2022) — InstructGPT | ☐ | ☐ |
| Liu et al. (2023) — Prompt Injection | ☐ | ☐ |
| Sanh et al. (2019) — DistilBERT | ☐ | ☐ |
| Ganguli et al. (2022) — Red Teaming | ☐ | ☐ |
