# Dataset Setup

This repo should be complete in code, configs, prompts, tests, and
documentation, but incomplete in large assets. Datasets and checkpoints stay
outside Git.

## Supported Dataset Modes

Current scaffold:

- `synthetic`: smoke tests and CI-style checks
- `dota`: DOTA-style oriented detection labels

The DOTA loader supports:

```text
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
```

and converted normalized labels:

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

## Expected DOTA Layout

The tracked configs expect this repo-local path, normally as a symlink:

```text
DOTAv2/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
```

Use `scripts/link_local_assets.py` to link the actual dataset path:

```bash
python scripts/link_local_assets.py --dotav2-root /path/to/DOTAv2
```

If server outputs should also appear at `outputs/`:

```bash
python scripts/link_local_assets.py \
  --dotav2-root /path/to/DOTAv2 \
  --outputs-dir /path/to/openprompt_outputs
```

## Git Policy

Do not commit:

- `DOTAv2/`
- `images/`
- `labels/`
- `outputs/`
- `checkpoints/`
- generated prompt-bank tensors
- large raw logs

Commit:

- dataset config templates
- conversion scripts
- small dataset statistics
- class mapping notes
- official evaluation setup notes

## Paper-Fit Dataset Plan

Primary dataset:

- DOTA v2 for the main JSTARS path.

Optional validation datasets:

- FAIR1M
- DIOR-R
- HRSC2016 for ship-focused sanity checks

Do not expand to a second dataset until the DOTA baseline is credible.

## Required Dataset Checks

Before training a paper run, record:

- image count per split
- label count per split
- class distribution
- tile size and stride
- whether empty tiles are included
- coordinate format
- class mapping
- evaluation script or metric implementation

Store small summaries in `docs/experiments/` or `docs/reproducibility/`.
