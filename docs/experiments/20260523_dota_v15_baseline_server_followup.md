# Experiment: dota_v15_baseline_server_followup

Date: 2026-05-23

Last updated: 2026-05-24

Status: completed

Git commit: `fa6ca42`

Machine: nuosen (server)

GPU: 1 x NVIDIA GeForce RTX 4090

Dataset: `/data2/2023/lcs/xyun/DOTA` using the staged DOTA v1.5 train/val split

Config: `configs/experiments/dota_v15_baseline_repro.yaml`

Detached screen session: `openprompt_dota_v15_baseline`

External log path: `outputs/openprompt_dota_v15_baseline/train.log`

Checkpoint path: `outputs/dota_v15_baseline_repro/epoch_012.pt`

## Purpose

Run the matched DOTA v1.5 reduced tiled scaffold baseline after the DOTA v1.0
pipeline sanity check produced a nonzero but extremely weak validation result.
This run must stay separate from DOTA v1.0 in notes, tables, and manuscript
wording.

## Monitor

```bash
screen -r openprompt_dota_v15_baseline
```

The run completed all 12 epochs.

## Evaluation Command

Completed on 2026-05-24 11:26:

```bash
source /data1/anaconda3/etc/profile.d/conda.sh && \
conda activate zwl_oneformer_ViT_P && \
cd /data5/2025/ldh/OpenPrompt && \
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/experiments/dota_v15_baseline_repro.yaml \
  --checkpoint outputs/dota_v15_baseline_repro/epoch_012.pt \
  --split val \
  --metric-set both \
  --score-threshold 0.05 \
  --nms-iou-threshold 0.3 \
  --max-detections 100 \
  | tee outputs/dota_v15_baseline_repro/eval_epoch_012_val.json
```

Evaluation artifact path: `outputs/dota_v15_baseline_repro/eval_epoch_012_val.json`

## Diagnostic Command

Run this after evaluation, especially if `map50` is near zero:

```bash
source /data1/anaconda3/etc/profile.d/conda.sh && \
conda activate zwl_oneformer_ViT_P && \
cd /data5/2025/ldh/OpenPrompt && \
PYTHONPATH=src python scripts/diagnose_baseline.py \
  --config configs/experiments/dota_v15_baseline_repro.yaml \
  --checkpoint outputs/dota_v15_baseline_repro/epoch_012.pt \
  --split val \
  --score-thresholds 0.05,0.01,0.001 \
  --nms-iou-threshold 0.3 \
  --max-detections 100 \
  --output outputs/dota_v15_baseline_repro/diagnostics_epoch_012_val.json \
  | tee outputs/dota_v15_baseline_repro/diagnostics_epoch_012_val.log
```

Use `--max-batches 50` for a quick first check before the full validation pass.

## Fields To Record

- `detection_metrics.num_eval_images`
- `detection_metrics.map50`
- `detection_metrics.mean_precision`
- `detection_metrics.mean_recall`
- per-class AP and recall failures
- dataset image/tile/object/class counts from `diagnostics_epoch_012_val.json`
- raw prediction score quantiles and predicted-class counts
- per-threshold detection counts, score quantiles, and best-IoU summaries
- whether the result is only a scaffold sanity check
- whether the metric is from this repo's scaffold rotated mAP@50, not official
  DOTA server evaluation

## Decision Rule

If DOTA v1.5 is also near zero, pause hierarchy/context/pseudo-label novelty
experiments and diagnose the baseline scaffold first.

Diagnosis order:

1. Verify train/val image count, tile count, label count, and class distribution.
2. Inspect decoded prediction score distribution before thresholding.
3. Run threshold sweeps at `0.05`, `0.01`, and `0.001`.
4. Verify predicted box coordinates after tiling and normalization reversal.
5. Confirm DOTA v1.0/v1.5 class mapping and ignored/difficulty label handling.
6. Visualize predictions over a small set of validation tiles.

If DOTA v1.5 is meaningfully better, treat it as the current scaffold baseline
and proceed to controlled prompt experiments without changing the detector or
config between ablations.

## Result

Validation completed on 4055 images with:

- `map50=1.0926445202230628e-05`
- `mean_precision=0.0006667361585641629`
- `mean_recall=0.0011823561703749874`

Nonzero class values appeared for harbor, plane, ship, small-vehicle, and
tennis-court, but all values remain extremely weak.

This is a completed scaffold sanity check, not a paper-quality detector result.
Because v1.5 is also near zero, the next action is scaffold repair before any
hierarchy, scene-context, or pseudo-label novelty experiments.

## Preliminary Diagnosis

Quick diagnostic run with `--max-batches 50` showed:

- validation tiles are highly imbalanced, with `small-vehicle` dominating the GT class counts
- raw detection scores are already above the tested thresholds, so `0.05`, `0.01`, and `0.001` do not change the decoded count
- predictions are heavily class-collapsed toward `small-vehicle`, `harbor`, `plane`, and `ship`
- predicted boxes are center-biased on the inspected tile, with mean centers near the tile midpoint and very low same-class IoU against GT

Spot-checking one validation tile confirmed the failure mode:

- GT objects are spread across the tile
- top predictions cluster near the center
- top predicted classes are mostly `small-vehicle` with a few `plane` and `ship` boxes

This points to a weak scaffold localizer rather than a thresholding issue.

Implementation note: `QueryGenerator` returns `query_centers`, but the current
`AlignmentHead` and `FusionHead` regress boxes only from pooled query tokens.
That means the scaffold has no explicit spatial anchor in the box head, which
may contribute to the center bias seen in the sample spot check.

## Next Step

Inspect the box-regression path in `src/openprompt_rs/models/heads.py` and
`src/openprompt_rs/models/detector.py`, then decide whether the quickest repair
is to feed `query_centers` into box regression or to move to a stronger baseline
implementation before any prompt/context ablations.

If the anchor-repair smoke test fails to reduce center bias or improve
best-IoU/recall meaningfully, shift effort to the stronger detector path
instead of extending the scaffold further.

Update 2026-05-24:
The anchor-repair smoke test completed and is archived in
`docs/experiments/20260524_dota_v15_anchor_repair_quick_test.md`. The 1-epoch
run wrote `outputs/dota_v15_anchor_repair/epoch_001.pt` and finished with
training-only metrics `loss=0.07363908355801901`, `loss_cls=0.001671954903589549`,
`loss_box=0.035983564312892485`, `positive_cls_acc=0.5529336195676059`, and
`positive_box_l1=0.10294117139314753`.
