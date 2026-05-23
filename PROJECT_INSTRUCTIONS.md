# Project Instructions

This file is the persistent project memory. Keep it updated when the research
scope, repository structure, server workflow, or experiment protocol changes.
Paper-first rule: if the research direction, claim, experiment sequence, or
submission target changes, update the canonical manuscript and this file before
changing code, configs, or secondary docs.

## Research Direction

Project name: GeoNexus-RSD.

Primary goal: hierarchy- and context-aware vision-language prompting for
DOTA-style oriented remote sensing object detection.

Practical first target: IEEE JSTARS. Consider TGRS or ISPRS P&RS only if final
results are strong across at least two datasets. Consider GRSL, IGARSS, or a
workshop if results are modest or incomplete.

Main paper claim:

Hierarchy- and context-aware vision-language prompting improves fine-grained
oriented object detection and semi-supervised pseudo-label quality in remote
sensing imagery.

Core modules:

1. Hierarchical prompt bank.
2. Scene/context prompt adapter.
3. VLM-assisted pseudo-label purification.

Secondary only:

- Routing is optional after the three core modules are stable.
- Compression is a later-paper topic.
- Segmentation is not the primary task for this paper.

Do not claim open-vocabulary detection unless the final system uses real
vision-language embeddings and evaluates a real open-vocabulary or vocabulary
robustness setting.

## Current Code Reality

This repository is a research scaffold, not a competitive detector yet.

Current limitations:

- The local backbone is lightweight.
- The current text embedder is deterministic hash-based unless replaced.
- Official DOTA evaluation is not fully integrated.
- Real benchmark results are not available yet.

Paper-level claims require:

- A credible oriented detector baseline, preferably from MMRotate or an
  equivalent strong implementation.
- Real text/image embeddings such as CLIP, SkyCLIP, or RemoteCLIP.
- Verified tiling, class mapping, rotated IoU/NMS, and mAP.
- Complete ablations with real numbers.

## Experiment Sequence

Run experiments in this order:

1. S0: strong closed-set oriented detector baseline on DOTA v1.0 or DOTA v1.5, using whichever staged server asset is ready first.
2. S1: flat class-name prompt classifier.
3. S2: hierarchical prompt bank.
4. S3: hierarchy plus scene/context adapter.
5. S4: hierarchy plus context plus VLM-assisted pseudo-label purification.
6. S5: optional routing ablation.

Do not add S5 to the main story unless S2-S4 already show stable gains.

Required final analyses:

- Main comparison table.
- Core ablation table.
- Prompt robustness table.
- Pseudo-label quality table.
- Efficiency table.
- Qualitative detections.
- Accepted/rejected pseudo-label examples.
- Confusion matrix or fine-grained class-pair analysis.

No final submission may contain pending/planned result tables.

## Paper-First Workflow

Canonical manuscript source:

- `docs/geonexus_short_paper.tex`

Supporting drafts and presentation notes may exist, but they must not override
the canonical manuscript. Keep method wording aligned with the real claim:
hierarchical prompts, scene/context adaptation, and VLM-assisted pseudo-label
purification. When code exposes routing or compression hooks, document them as
optional ablations or future work unless measured results justify making them
central.

Before any paper-facing claim is added:

- identify which experiment record supports it
- link the config and command used to produce it
- record whether the run used DOTA v1.0 or DOTA v1.5, and do not mix those numbers with later DOTA v2 results
- record whether embeddings are hash fallback or real VLM embeddings
- record whether metrics are from scaffold evaluation or accepted DOTA-style
  evaluation

## Local And Server Workflow

Use GitHub as the shared code and result-metadata transport between:

- local machine: code editing, documentation, lightweight tests
- experiment server: training, evaluation, logs, heavy artifacts

Recommended loop:

1. Local: edit code/docs/configs.
2. Local: run unit tests and smoke tests.
3. Local: commit and push to GitHub.
4. Server: pull GitHub.
5. Server: link datasets/checkpoints outside Git.
6. Server: run experiments.
7. Server: save logs and small structured summaries in Git-tracked locations.
8. Server: keep large outputs/checkpoints outside Git.
9. Server: commit and push code/config/log-summary changes.
10. Local: pull and continue analysis or code improvement.

Never put datasets, model checkpoints, raw large logs, or generated training
directories in Git. Commit configs, scripts, documentation, small metrics JSON,
environment notes, and experiment summaries.

## Repository Boundary

Tracked in Git:

- `assets/hierarchies/`
- `assets/prompts/`
- `configs/`
- `docs/`
- `scripts/`
- `src/`
- `tests/`
- root metadata and instruction files
- canonical paper source only, not duplicate generated PDFs

Ignored or external:

- `DOTA/`
- `DOTAv2/`
- `images/`
- `labels/`
- `outputs/`
- `checkpoints/`
- `artifacts/generated/`
- `wandb/`
- generated PDFs and LaTeX auxiliary files

## Future-Agent Prompt

When starting a new coding session, give the agent this instruction:

Read `PROJECT_INSTRUCTIONS.md`, then inspect the current Git status. Preserve
unrelated user changes. Continue the GeoNexus-RSD baseline-first JSTARS path:
do not make unsupported performance claims, keep routing/compression secondary,
maintain the local/server GitHub workflow, start with DOTA v1.0 or DOTA v1.5
if those server assets are already staged, and update the canonical manuscript
before code/docs when the research direction changes.
