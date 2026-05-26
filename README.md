# openprompt

`openprompt` is a baseline-first research scaffold for:

**GeoNexus-RSD: Hierarchy- and Context-Aware Prompt Learning for Oriented Remote
Sensing Object Detection**

The practical first paper target is JSTARS. TGRS or ISPRS P&RS should only be
considered if final results are strong across at least two datasets.

## Active Direction

Main claim:

Hierarchy- and context-aware vision-language prompting improves fine-grained
oriented object detection and semi-supervised pseudo-label quality in remote
sensing imagery.

Core modules:

- hierarchical prompt bank
- scene/context prompt adapter
- VLM-assisted pseudo-label purification

Secondary only:

- routing is optional after the core modules are stable
- compression is a later-paper topic
- segmentation is not the primary task for this paper

Persistent project context is tracked in [PROJECT_INSTRUCTIONS.md](PROJECT_INSTRUCTIONS.md).
Future coding agents should also read [AGENTS.md](AGENTS.md).

## What This Repo Is

This repo contains:

- Python package under `src/openprompt_rs`
- config-driven training and evaluation entrypoints
- DOTA-style dataset loader scaffold
- synthetic smoke-test dataset
- prompt taxonomy and prompt-template assets
- hierarchy/context/pseudo-label modules for ablations
- setup, reproducibility, and experiment-record documentation

This repo does not claim to be the official OpenRSD implementation, and it does
not yet contain paper-ready benchmark results.

## Current Limitations

- The local detector is lightweight and mainly useful for plumbing.
- The default text embedder is a deterministic hash fallback for smoke tests;
  real CLIP/OpenCLIP/RemoteCLIP text embeddings are selectable when the active
  environment provides the required package and checkpoint.
- Datasets and checkpoints are intentionally not tracked.
- Official DOTA evaluation still needs to be integrated or documented before
  paper-level claims.

Paper-level experiments require a credible oriented detector baseline and real
CLIP/SkyCLIP/RemoteCLIP-style embeddings.

## Repository Structure

```text
openprompt/
├── assets/                  # taxonomy and prompt assets
├── configs/                 # model, dataset, experiment configs
├── docs/
│   ├── experiments/         # small tracked experiment summaries
│   ├── logs/                # short curated log excerpts only
│   ├── method/              # method notes and ablation ideas
│   ├── reproducibility/     # environment and reproduction notes
│   └── setup/               # dataset, VLM, and server workflow setup
├── scripts/                 # train/eval/prompt-bank entrypoints
├── src/openprompt_rs/       # source package
├── tests/                   # smoke and unit tests
├── AGENTS.md                # pointer for future coding agents
└── PROJECT_INSTRUCTIONS.md  # persistent project memory
```

## Setup Pointers

- Dataset setup: [docs/setup/datasets.md](docs/setup/datasets.md)
- Prompt/VLM pipeline: [docs/setup/prompt_vlm_pipeline.md](docs/setup/prompt_vlm_pipeline.md)
- GitHub/server workflow: [docs/setup/github_server_workflow.md](docs/setup/github_server_workflow.md)
- Next steps: [docs/setup/next_steps.md](docs/setup/next_steps.md)
- Experiment records: [docs/experiments/README.md](docs/experiments/README.md)

## Quick Start

Install locally:

```bash
python -m pip install -e .
```

Run tests:

```bash
python -m pytest
```

If the package is not installed:

```bash
PYTHONPATH=src python -m pytest
```

Run a smoke test:

```bash
PYTHONPATH=src python scripts/smoke_test.py \
  --config configs/experiments/geonexus_synthetic.yaml
```

Build a prompt-bank artifact for inspection:

```bash
PYTHONPATH=src python scripts/build_prompt_bank.py \
  --taxonomy assets/hierarchies/remote_sensing_taxonomy.json \
  --templates assets/prompts/prompt_templates.json \
  --output artifacts/generated/prompt_bank_remote_sensing.pt \
  --embedding-dim 256
```

Smoke-test real RemoteCLIP text embeddings before S1:

```bash
PYTHONPATH=src python scripts/smoke_vlm_embeddings.py \
  --embedding-backend remoteclip \
  --embedding-model-name ViT-B-32 \
  --embedding-checkpoint /path/to/RemoteCLIP-ViT-B-32.pt
```

## Recommended Experiment Order

1. Verify dataset loading and tiling.
2. Establish a credible closed-set oriented detector baseline.
3. Add flat class-name prompts.
4. Add hierarchical prompt bank.
5. Add scene/context prompt adapter.
6. Add VLM-assisted pseudo-label purification.
7. Add optional routing only as an ablation.

Do not submit a paper with pending/planned result tables.

## Git Policy

Track:

- source code
- configs
- prompt assets
- setup docs
- small experiment summaries
- reproducibility notes

Do not track:

- datasets
- checkpoints
- generated prompt-bank tensors
- raw output directories
- long logs
- LaTeX auxiliary files

## Citation Anchor

The local scaffold is inspired by the OpenRSD research direction:

- Huang et al., `OpenRSD: Towards Open-prompts for Object Detection in Remote
  Sensing Images`, ICCV 2025.
