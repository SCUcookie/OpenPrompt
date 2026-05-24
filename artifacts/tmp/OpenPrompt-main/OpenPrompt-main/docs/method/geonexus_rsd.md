# GeoNexus-RSD Method Notes

This repository implements a practical research scaffold for the planned method:

**GeoNexus-RSD = strong oriented detection baseline + hierarchy-aware prompt semantics + scene-context prompt adaptation + conservative pseudo-label purification**

## Baseline modules mirrored from the paper

- offline prompt construction
- class embedding refinement
- alignment head
- fusion head
- staged training hooks

## Added research modules

### 1. Hierarchy-aware prompt bank

Prompt embeddings are not kept flat. Instead, the repo builds a relation matrix from:

- parent sharing
- explicit confusing classes
- synonym-enriched prompt strings
- scene priors
- geometry cues

One simple smoothing form already exposed by the code is:

`E_hat = normalize((I + lambda * A_norm) E)`

where:

- `E` is the base prompt matrix
- `A_norm` is the normalized class-relation matrix
- `lambda` controls hierarchy injection strength

### 2. Scene-context prompt adapter

The model extracts a global scene feature `g` and reweights prompt embeddings:

`E_scene(c) = normalize(E_hat(c) * (1 + sigma(W_g g)) * (1 + beta * s_c))`

where `s_c` is the class-specific scene gate.

### 3. Alignment and fusion heads

The baseline keeps two score paths:

- alignment: cheap cosine matching
- fusion: richer prompt-query interaction

The final detector score is a weighted combination:

`z = (1 - r) * z_align + r * z_fuse`

The baseline uses a fixed mixture. A learned router `r` is optional and should
only be evaluated after hierarchy, context, and pseudo-label purification have
been shown to help on a credible baseline.

### 4. Hierarchy-consistent pseudo-labeling

Pseudo-label acceptance is not based on confidence only. The scaffold exposes a composite score:

`S = w1 * conf + w2 * sem + w3 * scene + w4 * hier - w5 * neg`

This gives a clean axis for experiments and ablations.

## Current credibility requirements

The local scaffold is useful for ablations, but its tiny backbone and hash text
embeddings are not enough for a serious journal claim. Before submission, the
main experiments should use:

- a strong oriented detector baseline, such as Oriented R-CNN, RoI Transformer,
  ReDet, Oriented RepPoints, or an MMRotate-style implementation
- CLIP, SkyCLIP, RemoteCLIP, or a comparable real text/image encoder instead of
  hash embeddings
- verified DOTA tiling, class mapping, rotated IoU/NMS, and official or widely
  accepted mAP evaluation
- real ablations for flat prompts, hierarchy, context adaptation, and
  pseudo-label purification

Routing and compression should remain secondary unless the core detection
story is already empirically strong.
