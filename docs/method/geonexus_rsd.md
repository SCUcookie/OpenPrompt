# GeoNexus-RSD Method Notes

This repository implements a practical research scaffold for the planned method:

**GeoNexus-RSD = OpenRSD-like dual-head detection + hierarchy-aware prompt semantics + scene-context prompt adaptation + hierarchy-consistent pseudo-labeling**

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

The baseline uses a fixed mixture; `GeoNexus-RSD` can use a learned router `r`.

### 4. Hierarchy-consistent pseudo-labeling

Pseudo-label acceptance is not based on confidence only. The scaffold exposes a composite score:

`S = w1 * conf + w2 * sem + w3 * scene + w4 * hier - w5 * neg`

This gives a clean axis for experiments and ablations.

## Why this repo chooses lightweight modules

The local planning package explicitly fixed a hard resource ceiling of `4 x 4090`. The rebuild therefore keeps:

- prompt encoders replaceable
- scene modeling lightweight
- hierarchy reasoning mostly offline
- routing and filtering cheap enough for ablations

