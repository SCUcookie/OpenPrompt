# Innovation Playbook

This file lists concrete upgrade routes that can push beyond the rebuilt baseline without drifting into vague trend chasing.

## Tier 1: Fast and defensible

### Hierarchy Laplacian regularization

Add a graph smoothness loss over prompt logits:

`L_hier = trace(P^T (D - A) P) / C`

Why it can help:

- stabilizes confusing fine-grained classes
- easy to ablate
- mathematically clean

### Scene-conditional temperature scaling

Instead of only gating prompt magnitudes, let scene features modulate logit temperature:

`tau(g) = softplus(w^T g) + eps`

`z_c = sim(q, e_c) / tau(g)`

Why it can help:

- ambiguous scenes often need sharper or softer class distributions
- low extra cost

### Confusing-class margin loss

Define hard negative pairs from the hierarchy file and enforce:

`L_margin = max(0, m + z_neg - z_pos)`

Why it can help:

- directly attacks ship/vehicle/tank confusions
- reviewer-friendly because the failure mode is visible

## Tier 2: Stronger structure upgrades

### Query routing by scale and uncertainty

Learn a router:

`r = sigmoid(W[q ; entropy(z_align) ; box_geom])`

Use it to decide how much fusion is needed.

Why it can help:

- tiny or rotated-hard objects benefit from richer reasoning
- easy objects keep fast alignment behavior

### Geometry-aware prompt modulation

Predict a geometry prior vector from query features and match it against class geometry attributes.

Example score:

`z_geom = z + alpha * cos(phi_geom(q), g_c)`

Why it can help:

- remote sensing categories often differ by aspect ratio, compactness, and orientation tolerance
- adds a more publishable inductive bias than generic prompt tuning

### Scene-aware hard-negative suppression

Use explicit scene priors to discount impossible labels:

`z'_c = z_c - gamma * 1[c in neg(scene)]`

Why it can help:

- removes high-confidence false positives in obvious contexts
- useful for cross-dataset shift

## Tier 3: Paper-shaping upgrades

### Mixed-granularity prompt evaluation

Benchmark exact class prompts, parent-class prompts, synonym prompts, and mixed prompt sets.

Why reviewers care:

- it shows the method solves a real open-prompt failure mode
- the story is broader than in-domain AP

### Small-object transfer protocol

Report transfer from large datasets to a small-object-heavy target such as `SODA-A`.

Why reviewers care:

- highlights a hard and practical setting
- aligns with the planning notes already preserved in this repo

### Efficiency table

Report:

- trainable parameter delta
- throughput delta
- GPU memory delta
- AP delta

Why reviewers care:

- avoids the "expensive prompt engineering" criticism

## Suggested ablation order

1. hierarchy bank only
2. scene adapter only
3. hierarchy + scene
4. add pseudo-label composite score
5. add routing
6. add one mathematical regularizer

## Ideas most likely to look strong in a paper

- hierarchy + scene + query routing as the core architecture story
- confusing-class margin or Laplacian regularization as the mathematical story
- mixed-granularity robustness + small-object transfer as the evaluation story

