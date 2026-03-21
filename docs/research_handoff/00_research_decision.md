# Research Decision

## Bottom line

Your best next project is **not** "OpenRSD + some fashionable module".

Your best next project is:

**Hierarchy- and Context-Aware Open-Prompt Rotated Detection for Remote Sensing**

Short codename: **GeoNexus-RSD**

Why this is the best choice:

1. It stays close to the strongest baseline you already trust: [OpenRSD](./openrsd-ICCV2025_paper.pdf).
2. It attacks the exact weaknesses still visible in OpenRSD: mixed-prompt inconsistency, cross-dataset shift, and fine-grained/small rotated object ambiguity.
3. It is feasible under **4 x 4090** because the LLM part can be fully offline and lightweight.
4. It is more publishable than an "agent" paper and lower risk than making diffusion the core method.

## What I recommend you do first

Choose **object detection**, not semantic segmentation, as your primary paper track.

More specifically:

- Task: **open-prompt rotated object detection**
- Focus: **small and confusing targets under cross-dataset shift**
- Baseline: **OpenRSD**
- Main add-on: **LLM-built semantic hierarchy + scene-context prompt adaptation + hierarchy-consistent self-training**

## What I do not recommend as the main direction

### 1. Agent as the core contribution

I do not recommend this.

Reason:

- I did not find strong 2025 top-tier remote sensing detection/segmentation papers where "agentic reasoning" is the main source of benchmark gains.
- It is harder to turn into stable AP improvements.
- Reviewers will ask whether the agent is doing anything beyond expensive prompt engineering.

Agent can be used later as a tooling layer for data curation, but not as the paper's central method.

### 2. Diffusion as the core contribution

I also do not recommend this as the first route.

Reason:

- Diffusion is hot, but the evidence is much stronger for generation/foundation modeling than for **open-prompt rotated remote sensing detection**.
- Under 4 x 4090, diffusion-heavy training is higher risk.
- If diffusion is used, it should be a **small supporting module**, not the whole paper.

### 3. Semantic segmentation as the first paper

This is a valid backup direction, but not the best first move for you.

Reason:

- Recent open-vocabulary remote sensing segmentation papers are already strong and moving fast.
- Your current baseline and intuition are much more aligned with object detection.
- Detection lets you reuse more of OpenRSD's structure, datasets, and evaluation logic.

## Recommended paper title candidates

1. **GeoNexus-RSD: Hierarchy- and Context-Aware Open-Prompt Rotated Detection for Remote Sensing**
2. **OpenRSD++: Semantic Hierarchy and Scene Context for Cross-Dataset Rotated Remote Sensing Detection**
3. **Towards Fine-Grained Open-Prompt Rotated Detection in Remote Sensing via Hierarchy-Aware Context Adaptation**

## The one-sentence thesis

**Open-prompt remote sensing detection still fails on small, ambiguous, and cross-domain targets because the prompt space lacks hierarchy and scene grounding; adding lightweight hierarchy-aware and context-aware prompt control is the highest-probability way to beat OpenRSD under practical hardware limits.**
