# Strong Detector Run Summary: DOTA v1.5 S0

Date: 2026-05-25

Purpose: summarize every detector run attempted in the current strong-baseline
sweep and record the OpenRSD changes needed to reproduce or continue the work.
This file is the compact handoff companion to
`docs/experiments/20260525_strong_detector_sweep.md`.

## Bottom Line

Normal S0 closed-set detector results were obtained after fixing the MMRotate
DOTA v1.5 validation pipeline. Oriented R-CNN is the primary usable baseline;
RoI Transformer is a close secondary baseline through epoch 11; ReDet completed
as a scratch diagnostic baseline.

Final usable result:

- Detector: Oriented R-CNN, LE90, ResNet-50-FPN, AMP-style wrapper.
- Dataset/split: DOTA v1.5 train/val under `/data5/2025/ldh/OpenPrompt/DOTA/`.
- Metric: MMRotate `DOTAMetric`, IoU 0.5, DOTA-style oriented mAP.
- Checkpoint: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/oriented_rcnn/epoch_12.pth`.
- Log: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/oriented_rcnn/20260525_163043/20260525_163043.log`.
- Final validation: `dota/mAP=0.2561`, `dota/AP50=0.2560`.
- Small metric JSON: `docs/experiments/20260525_oriented_rcnn_dota15_epoch12_metrics.json`.

Additional tracked summaries:

- RoI Transformer low-LR rerun: `docs/experiments/20260525_roi_transformer_dota15_metrics.json`.
- ReDet scratch rerun: `docs/experiments/20260525_redet_scratch_dota15_metrics.json`.

The near-zero early numbers were not detector evidence. They came from a
validation pipeline bug: val/test loaded annotations before resize, causing GT
boxes to be resized into the wrong evaluator coordinate space. The corrected
val/test order is:

```text
LoadImageFromFile -> Resize -> LoadAnnotations -> ConvertBoxType -> RandomFlip(prob=0) -> PackDetInputs
```

## OpenRSD Changes Required

The following OpenRSD working-tree changes are required for the successful
baseline path. They are outside the OpenPrompt Git repo but must be preserved on
the server or ported into the shared code transport before reruns.

OpenRSD also contains other pre-existing dirty files and generated outputs.
Only the files listed in this section are required by the DOTA v1.5 strong
baseline sweep summarized here.

### Strong-baseline wrapper directory

Directory:
`/data5/2025/ldh/OpenRSD/mmrotate_configs/strong_baseline_dota15/`

Files:

- `oriented-rcnn-le90_r50_fpn_amp-1x_dota15.py`
- `roi-trans-le90_r50_fpn_amp-1x_dota15.py`
- `redet-le90_re50_refpn_amp-1x_dota15.py`

Common changes in all three wrappers:

- DOTA v1.5 data root set to `/data5/2025/ldh/OpenPrompt/DOTA/`.
- Train annotations/images use `train/annfiles/` and `train/images/`.
- Val/test annotations/images use `val/annfiles/` and `val/images/`.
- Train pipeline keeps qbox annotation load, qbox-to-rbox conversion, 640x640 resize, random flips, and `PackDetInputs`.
- Val/test pipeline now resizes before annotation loading, then converts qbox to rbox.
- Val/test `PackDetInputs` uses explicit metadata keys: `img_id`, `img_path`, `ori_shape`, `img_shape`, `scale_factor`.

Detector-specific OpenRSD changes:

- Oriented R-CNN wrapper sets `roi_head.bbox_head.num_classes=16`.
- RoI Transformer wrapper lowers SGD LR to `0.001` after the earlier `0.005` run diverged to NaN and now explicitly overrides both cascade bbox heads to `num_classes=16`.
- ReDet wrapper clears `model.backbone.init_cfg` because the expected ReResNet pretrained checkpoint was missing, and keeps LR `0.001`. Treat ReDet as scratch unless the pretrained checkpoint is restored.

### Test script compatibility

File:
`/data5/2025/ldh/OpenRSD/tools/test.py`

Change:

- Guarded `from mmdet.evaluation import DumpDetResults` because this installed
  `mmdet` does not expose `DumpDetResults`; standalone validation without
  `--out` should not fail at import time.
- A fallback `setup_cache_size_limit_of_dynamo()` no-op is present if the
  installed `mmdet` lacks that helper.

This is needed for standalone `tools/test.py` validation in the current
`zwl_mmrotate` environment.

## Run Results

### Oriented R-CNN

Status: completed, usable S0 baseline.

Final config:
`/data5/2025/ldh/OpenRSD/mmrotate_configs/strong_baseline_dota15/oriented-rcnn-le90_r50_fpn_amp-1x_dota15.py`

Final checkpoint:
`/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/oriented_rcnn/epoch_12.pth`

Final log:
`/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/oriented_rcnn/20260525_163043/20260525_163043.log`

Training command used to complete the successful run:

```bash
CUDA_VISIBLE_DEVICES=1 /data1/anaconda3/envs/zwl_mmrotate/bin/python \
  tools/bootstrap_run.py tools/train.py \
  mmrotate_configs/strong_baseline_dota15/oriented-rcnn-le90_r50_fpn_amp-1x_dota15.py \
  --work-dir work_dirs/strong_baseline_dota15/oriented_rcnn \
  --resume work_dirs/strong_baseline_dota15/oriented_rcnn/epoch_3.pth
```

Corrected standalone validation command used for the epoch-3 diagnosis:

```bash
CUDA_VISIBLE_DEVICES=1 /data1/anaconda3/envs/zwl_mmrotate/bin/python \
  tools/bootstrap_run.py tools/test.py \
  mmrotate_configs/strong_baseline_dota15/oriented-rcnn-le90_r50_fpn_amp-1x_dota15.py \
  work_dirs/strong_baseline_dota15/oriented_rcnn/epoch_3.pth \
  --work-dir work_dirs/strong_baseline_dota15/oriented_rcnn_eval_epoch3_fixedval
```

Validation progression after the pipeline-order fix:

| Epoch | dota/mAP | AP50 | Status |
|---:|---:|---:|---|
| 3 | 0.1725 | 0.1720 | corrected spot validation |
| 4 | 0.1898 | 0.1900 | train-loop val |
| 5 | 0.2029 | 0.2030 | train-loop val |
| 6 | 0.2066 | 0.2070 | train-loop val |
| 7 | 0.2238 | 0.2240 | train-loop val |
| 8 | 0.2298 | 0.2300 | train-loop val |
| 9 | 0.2511 | 0.2510 | train-loop val |
| 10 | 0.2548 | 0.2550 | train-loop val |
| 11 | 0.2525 | 0.2530 | train-loop val |
| 12 | 0.2561 | 0.2560 | final train-loop val |

Final epoch-12 class AP:

| Class | AP |
|---|---:|
| plane | 0.353 |
| baseball-diamond | 0.337 |
| bridge | 0.091 |
| ground-track-field | 0.361 |
| small-vehicle | 0.091 |
| large-vehicle | 0.421 |
| ship | 0.091 |
| tennis-court | 0.727 |
| basketball-court | 0.310 |
| storage-tank | 0.091 |
| soccer-ball-field | 0.363 |
| roundabout | 0.107 |
| harbor | 0.422 |
| swimming-pool | 0.160 |
| helicopter | 0.176 |
| container-crane | 0.000 |

Final epoch-12 class recall:

| Class | Recall |
|---|---:|
| plane | 0.331 |
| baseball-diamond | 0.376 |
| bridge | 0.077 |
| ground-track-field | 0.483 |
| small-vehicle | 0.032 |
| large-vehicle | 0.432 |
| ship | 0.098 |
| tennis-court | 0.789 |
| basketball-court | 0.322 |
| storage-tank | 0.024 |
| soccer-ball-field | 0.443 |
| roundabout | 0.103 |
| harbor | 0.461 |
| swimming-pool | 0.153 |
| helicopter | 0.205 |
| container-crane | 0.000 |

Interpretation:

- This is the first credible S0 baseline for the project.
- It is usable as a closed-set detector baseline, not as open-vocabulary or VLM evidence.
- `container-crane` remains unsolved in this run; it has only 14 validation GT boxes in this split.

### RoI Transformer

Status: stable low-LR rerun usable through epoch 11; epoch 12 not written at
latest check.

Current config:
`/data5/2025/ldh/OpenRSD/mmrotate_configs/strong_baseline_dota15/roi-trans-le90_r50_fpn_amp-1x_dota15.py`

Observed preliminary result:

- The earlier 1-epoch run wrote `epoch_1.pth` under `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/roi_trans/`.
- Training diverged to NaN with the earlier `lr=0.005` setting.
- Because training diverged, no RoI Transformer validation number should be used.

Current OpenRSD state for next attempt:

- Wrapper LR is lowered to `0.001`.
- Both cascade bbox heads are set to 16 DOTA v1.5 classes.
- Val/test pipeline has the corrected order and metadata keys.

Low-LR rerun:

- Screen session: `geonexus_roi_trans_lr001`.
- Command:

```bash
CUDA_VISIBLE_DEVICES=2 /data1/anaconda3/envs/zwl_mmrotate/bin/python \
  tools/bootstrap_run.py tools/train.py \
  mmrotate_configs/strong_baseline_dota15/roi-trans-le90_r50_fpn_amp-1x_dota15.py \
  --work-dir work_dirs/strong_baseline_dota15/roi_trans_lr001_rerun
```

- Stable through epoch 11 with no visible NaN evidence.
- Best observed validation was epoch 10: `dota/mAP=0.2485`,
  `dota/AP50=0.2480`.
- Epoch 11 validation was `dota/mAP=0.2436`, `dota/AP50=0.2440`.
- Latest visible log reached epoch 12 iter 1280/1410, but `epoch_12.pth` had
  not been written and the screen hardcopy was blank. Treat epoch 12 as
  pending/stalled until verified.
- If epoch 12 remains stalled, either use epoch 10 as the best RoI Transformer
  row or relaunch from epoch 11.

### ReDet

Status: active scratch-rerun attempt.

Current config:
`/data5/2025/ldh/OpenRSD/mmrotate_configs/strong_baseline_dota15/redet-le90_re50_refpn_amp-1x_dota15.py`

Observed preliminary result:

- The earlier scratch 1-epoch run wrote `epoch_1.pth` under `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/redet/`.
- Training losses stayed finite in that short run.
- Corrected-metadata-only validation before the pipeline-order fix gave `dota/mAP=0.0001`, `dota/AP50=0.0000`.
- That number is invalid/diagnostic because val/test still loaded annotations before resize.

Active rerun:

- Screen session: `geonexus_redet_scratch`.
- Command:

```bash
CUDA_VISIBLE_DEVICES=4 /data1/anaconda3/envs/zwl_mmrotate/bin/python \
  tools/bootstrap_run.py tools/train.py \
  mmrotate_configs/strong_baseline_dota15/redet-le90_re50_refpn_amp-1x_dota15.py \
  --work-dir work_dirs/strong_baseline_dota15/redet_scratch_rerun
```

- Startup status: epoch 1 reached iter 300 with finite losses after an initial
  `grad_norm: nan` at iter 50; monitor before counting any checkpoint.

Current OpenRSD state for next attempt:

- ReDet wrapper clears `model.backbone.init_cfg=None`, so this is scratch unless the ReResNet pretrained checkpoint is restored.
- LR is `0.001`.
- Val/test pipeline has the corrected order and metadata keys.

Next action:

- Revalidate `epoch_1.pth` with the corrected pipeline only as a quick diagnostic, or preferably rerun longer after restoring ReResNet pretraining.
- Compare ReDet cautiously against Oriented R-CNN because the current setup is scratch while Oriented R-CNN uses ResNet-50 pretraining.

## Evidence Status

| Detector | Training status | Validation status | Use for S0 claim? |
|---|---|---|---|
| Oriented R-CNN | Completed 12 epochs, finite | `mAP=0.2561`, `AP50=0.2560` | Yes, closed-set S0 baseline |
| RoI Transformer | Previous run diverged to NaN | Not counted | No |
| ReDet | 1-epoch scratch smoke only | Previous number invalid due pipeline order | No |

## Immediate Next Steps

1. Preserve or commit the OpenRSD wrapper/test-script changes before rerunning other detectors.
2. Decide whether Oriented R-CNN alone is enough to open S1 flat-prompt experiments, or require one more stable detector.
3. Rerun RoI Transformer with the lowered LR wrapper.
4. Rerun or revalidate ReDet with the corrected val/test pipeline and preferably a real ReResNet pretrained checkpoint.
