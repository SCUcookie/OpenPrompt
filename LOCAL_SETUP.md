# Local Setup

This repo is now organized so source code can be pushed cleanly while datasets,
checkpoints, and logs stay local.

## 1. Keep local assets out of Git

The following paths are intentionally ignored:

- `DOTAv2/`
- `labels/`
- `outputs/`

`DOTAv2/` is the active converted dataset root used by the tracked DOTA-v2
configs. `labels/` can be kept as a local stash for original-format
annotations if you want to regenerate labels later.

## 2. Link a local DOTA-v2 dataset

If your dataset lives outside the repo, create a repo-local symlink:

```bash
cd OpenPrompt
python scripts/link_local_assets.py --dotav2-root /path/to/DOTAv2
```

That makes these tracked configs work without editing them:

- `configs/datasets/dota_v2_train.yaml`
- `configs/datasets/dota_v2_val.yaml`

## 3. Optionally keep outputs outside the repo

```bash
cd OpenPrompt
python scripts/link_local_assets.py \
  --dotav2-root /path/to/DOTAv2 \
  --outputs-dir /path/to/OpenPrompt_outputs
```

## 4. Keep the official OpenRSD repo separate

Do not vendor `OpenRSD` inside `OpenPrompt`.

Use the sibling repo instead:

- `../OpenRSD`

That repo already has its own local bootstrap workflow and `CKpoint_pkl`
asset handling.
