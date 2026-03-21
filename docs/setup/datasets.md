# Dataset Setup

## Supported dataset modes in the current scaffold

- `synthetic`: runs immediately for smoke tests
- `dota`: parses common DOTA-style `image_dir + label_dir`

## DOTA-style expectation

The current loader expects:

```text
dataset_root/
├── images/
│   ├── train/
│   └── val/
└── labelTxt/
    ├── train/
    └── val/
```

Each label line should follow the common DOTA pattern:

```text
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
```

## What you still need to fill in

Edit:

- `configs/datasets/dota_template.yaml`
- `configs/experiments/dota_baseline_template.yaml`
- `configs/experiments/dota_geonexus_template.yaml`

Replace placeholder paths with absolute local paths.

## Recommended next dataset additions

- `DIOR-R`
- `FAIR1M-2.0`
- `HRSC2016`
- `SODA-A`

These should be added as separate loaders or converted into a unified manifest format once the actual local dataset paths are known.

