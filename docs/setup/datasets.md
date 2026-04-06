# Dataset Setup

## Supported dataset modes in the current scaffold

- `synthetic`: runs immediately for smoke tests
- `dota`: parses common DOTA-style `image_dir + label_dir`

## DOTA-style expectation

The current loader supports both:

- original DOTA annotation lines
- converted `class_id + normalized polygon` lines

The local layout used by the tracked DOTA configs is:

```text
DOTAv2/
├── images/
│   ├── train/
│   ├── test/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Original-format labels can still be kept separately if you need them for
re-conversion or class recovery.

Supported label formats:

```text
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
```

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

## What you still need to fill in

- For custom datasets, edit `configs/datasets/dota_template.yaml`,
  `configs/experiments/dota_baseline_template.yaml`, and
  `configs/experiments/dota_geonexus_template.yaml`.
- For the tracked DOTA-v2 configs, keep a local `DOTAv2/` directory in the repo
  root or symlink it from an external dataset location.

Recommended local-asset workflow:

```bash
cd OpenPrompt
python scripts/link_local_assets.py --dotav2-root /path/to/DOTAv2
```

If you want training outputs outside the repo as well:

```bash
cd OpenPrompt
python scripts/link_local_assets.py \
  --dotav2-root /path/to/DOTAv2 \
  --outputs-dir /path/to/OpenPrompt_outputs
```

## Recommended next dataset additions

- `DIOR-R`
- `FAIR1M-2.0`
- `HRSC2016`
- `SODA-A`

These should be added as separate loaders or converted into a unified manifest format once the actual local dataset paths are known.
