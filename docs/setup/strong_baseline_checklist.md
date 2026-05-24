# Strong Baseline Checklist

Use this path in parallel with the anchor-repair smoke test. The goal is to
secure a standard oriented detector baseline before any prompt/VLM claim grows
beyond the current scaffold.

## Decision Points

1. Confirm whether MMRotate can be installed cleanly in `zwl_oneformer_ViT_P`.
   If dependency conflicts are likely, prefer a separate clean environment.
2. Confirm DOTA v1.5 train/val paths, tiling format, class names, ignored
   labels, and the normalized/absolute OBB conversion expected by the selected
   framework.
3. Select one standard oriented detector baseline first, preferably Oriented
   R-CNN or RoI Transformer.
4. Record the exact detector config, dataset split, pretrained checkpoint,
   training command, validation command, and output checkpoint path.
5. Keep large checkpoints and raw logs outside Git; commit only configs,
   environment notes, and small metric summaries.

## Selected Baseline

- First candidate: Oriented R-CNN (LE90, ResNet-50-FPN, 1x-style schedule).
- Adaptation target: swap the DOTA v1.0 dataset base for
  `OpenRSD/mmrotate_configs/_base_/datasets/dotav15.py` and keep the same
  1024x1024-style train/val pipelines used by the reference tree.
- Backup candidate: RoI Transformer (LE90, ResNet-50-FPN, 1x-style schedule).

## Working Notes

- Current probe in `zwl_oneformer_ViT_P`: `mmrotate` is not importable yet,
  so a separate detector environment may be cleaner than forcing the current
  scaffold env to host MMRotate.
- Candidate starting configs in the OpenRSD reference tree:
  - Oriented R-CNN: `OpenRSD/mmrotate_configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py`
  - RoI Transformer: `OpenRSD/mmrotate_configs/roi_trans/roi-trans-le90_r50_fpn_1x_dota.py`
- DOTA v1.5 dataset base in the reference tree:
  - `OpenRSD/mmrotate_configs/_base_/datasets/dotav15.py`
- Current reference-tree finding: there is no direct `oriented_rcnn` or
  `roi_trans` DOTA v1.5 config, so the strong baseline will need to adapt the
  DOTA v1.0 config to the `dotav15.py` dataset base or start from another
compatible MMRotate config and swap the dataset base carefully.
- The DOTA v1.0 configs are 15-class setups; a DOTA v1.5 baseline must be
  switched to 16 classes and the v1.5 label mapping before training.
- DOTA v1.5 class order in the reference tree is:
  plane, baseball-diamond, bridge, ground-track-field, small-vehicle,
  large-vehicle, ship, tennis-court, basketball-court, storage-tank,
  soccer-ball-field, roundabout, harbor, swimming-pool, helicopter,
  container-crane.
- MMRotate's DOTA dataset parser reads 8-point polygon annotations plus a
  difficulty column; `diff_thr` controls whether a ground-truth box is marked
  ignored. With the default threshold used in the reference tree, all current
  DOTA-style GT labels should be kept unless a specific split requires otherwise.
- The reference-tree DOTA v1.5 pipeline loads polygon boxes and converts them to
  rotated boxes through the dataset pipeline (`LoadAnnotations` with `qbox`,
  then `ConvertBoxType` to `rbox`), so the strong-baseline path should record
  the absolute-box to rbox conversion explicitly.
- Prefer the same DOTA v1.5 split already used for the scaffold baseline so the
  comparison stays controlled.
- Local DOTA v1.5 config references in this repo are
  `configs/datasets/dota_v15_train.yaml` and `configs/datasets/dota_v15_val.yaml`.
- The reference-tree DOTA v1.5 base uses `data_root='data/split_ss_dota1_5/'`
  with `ann_file='trainval/annfiles/'` and `data_prefix='trainval/images/'`.
- If the framework expects absolute OBBs instead of normalized boxes, record the
  conversion rule explicitly before training.
- If MMRotate is not cleanly installable in the current env, use a dedicated
  detector environment instead of bending the scaffold env further.

## Evidence To Collect

- detector name and config path
- dataset version and split
- pretrained weights source
- training command
- validation command
- checkpoint path
- mAP / recall summary
- any deviations from the scaffold baseline settings