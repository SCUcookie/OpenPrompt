# OpenRSD Gap Analysis

This file is based on the local reference paper: [OpenRSD](./openrsd-ICCV2025_paper.pdf).

## What OpenRSD already solved

OpenRSD is already a serious baseline, not a weak one.

Main strengths from the paper:

- It supports both **text prompts** and **image prompts**.
- It handles both **oriented bounding boxes** and **horizontal bounding boxes** in one framework.
- It combines two heads:
  - **Alignment head** for speed and vocabulary scalability
  - **Fusion head** for higher precision
- It uses a **multi-stage training pipeline**:
  - pretraining
  - fine-tuning
  - self-training
- It explicitly targets **cross-dataset generalization**, not only in-dataset AP.

## Hard numbers that matter

### Main OBB result

From Table 1:

- OpenRSD (text prompt) average OBB AP50: **69.3**
- RTMDet-R-L baseline average OBB AP50: **66.2**
- Gain over baseline: about **+3.1**

Notable per-dataset gains over RTMDet-R-L:

- DIOR-R: **73.7 vs 70.1**
- DOTA-v1.0: **76.9 vs 71.5**
- DOTA-v2.0: **70.1 vs 66.7**
- HRSC2016: **88.1 vs 79.9**

### Cross-dataset generalization

From Table 3:

- OpenRSD average cross-dataset AP50: **78.1**
- RTMDet-R-L average cross-dataset AP50: **75.9**
- Gain: **+2.2**

The most important line:

- On **SODA-A**, OpenRSD reaches **72.3** vs RTMDet-R-L **67.1**
- Gain: **+5.2**

This is directly relevant to your small-object instinct.

### Module behavior

From Tables 4 and 5:

- Fusion head beats alignment head after joint training:
  - Fusion average: **68.2**
  - Alignment average: **67.0**
- Class embeddings help stability and average performance.

### Prompt behavior

From Table 6:

- Text prompts are already very strong and stable.
- On DOTA-v2.0, text prompt stays around **70.1 to 70.2** from 1 to 20 prompts.
- Image prompt is weaker but improves with more prompts:
  - about **65.7 -> 68.0**

This means a paper that only says "better prompt generation" is too weak.

### Training behavior

From Table 7:

- Fine-tuning without pretraining/self-training average: **64.0**
- Adding pretraining: **65.5**
- Adding self-training with full-parameter update: **66.9**
- Using external Million-AID for self-training: **65.3**

This is very important:

- **More unlabeled data alone did not help.**
- So your next paper should not be "just add more data".

## What OpenRSD still does not solve well

### 1. Fine-grained ambiguity is still a weak point

The paper itself says the detector still struggles with:

- ships
- buildings
- mixed prompts across datasets

This is exactly the zone where small rotated targets become hard.

### 2. The self-training filter is still simple

OpenRSD uses:

- a category tree
- score thresholding
- SkyCLIP similarity filtering

This is useful, but still relatively shallow semantics.

That leaves room for:

- hierarchy-aware pseudo-label selection
- scene-aware negative suppression
- prompt granularity control

### 3. Prompt semantics are not deeply controlled

OpenRSD already uses GPT-4 to generate text prompts.

Therefore:

- "use an LLM to write better prompts" is not enough
- the next step must be **more structured** than prompt paraphrasing

### 4. Context is underused

OpenRSD uses cropped prompts and detection features, but it does not strongly model:

- global scene context
- regional context around ambiguous instances
- prompt adaptation conditioned on scene type

### 5. Small-object performance is promising but still not the final answer

The SODA-A gain is a good sign, but it also tells you where the real publishable headroom probably is:

- tiny object recall
- fine-grained category confusion
- cross-domain small-object robustness

## What your next paper should NOT do

Do not write a paper whose main story is one of these:

1. "We replace SkyCLIP with another encoder."
2. "We add more prompt templates."
3. "We add more unlabeled data."
4. "We add an agent loop."
5. "We use diffusion because diffusion is fashionable."

These are weak unless they are attached to a much stronger core idea.

## The actual opening left by OpenRSD

The cleanest opening is:

**OpenRSD has strong prompt-based rotated detection, but its prompt semantics are still flat, its context use is limited, and its pseudo-label refinement is not hierarchy-aware.**

That is the gap your paper should attack.
