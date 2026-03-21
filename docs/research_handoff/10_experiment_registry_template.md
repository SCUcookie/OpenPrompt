# Experiment Registry Template

Use this file as the canonical experiment log once implementation starts.

## Usage rule

One block per experiment.

Do not overwrite old records.
Append new blocks.

## Template

```text
Experiment ID:
Date:
Owner:

Goal:

Codebase path:
Branch / commit:

Datasets:

Model:

Baseline or variant:

Changed components:
- 

Frozen components:
- 

Training setup:
- GPUs:
- batch size:
- image size:
- epochs / iterations:
- optimizer:
- lr:

Evaluation setup:
- datasets:
- metric:
- prompt mode:

Primary results:
- 

Compared against:
- 

Observed gains:
- 

Observed failures:
- 

Interpretation:

Decision:
- continue
- revise
- drop

Next action:
```

## Mandatory first entries once work starts

1. baseline reproduction attempt
2. hierarchy-bank-only ablation
3. scene-context-only ablation
4. hierarchy-consistent pseudo-labeling ablation

## Reporting standard

Every experiment record should include:

1. what changed
2. what stayed fixed
3. what metric moved
4. whether the result is strong enough to justify more compute
