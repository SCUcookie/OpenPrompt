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
DOTA/
  train/
    images/
    labelTxt-v1.0/
    labelTxt-v1.5/
  val/
    images/
    labelTxt-v1.0/
    labelTxt-v1.5/
  test/
    images/
```

Use `scripts/link_local_assets.py` to link the actual dataset path:

```bash
python scripts/link_local_assets.py --dota-root /path/to/DOTA
```

If server outputs should also appear at `outputs/`:

```bash
python scripts/link_local_assets.py \
  --dota-root /path/to/DOTA \
  --outputs-dir /path/to/openprompt_outputs
```

## Git Policy

Do not commit:

- `DOTA/`
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

- DOTA v1.0 or DOTA v1.5 for the first server baseline path.
- DOTA v2 only after the initial baseline is credible and the asset is actually staged.

Optional validation datasets:

- FAIR1M
- DIOR-R
- HRSC2016 for ship-focused sanity checks

Do not expand to a second dataset until the DOTA v1.0/v1.5 baseline is credible.

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
