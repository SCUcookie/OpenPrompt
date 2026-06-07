# DIOR-R ORCNN NaN Diagnostic And RoI Transformer Launch - 2026-06-07

This note records the failed low-LR DIOR-R Oriented R-CNN diagnostic and the
replacement DIOR-R RoI Transformer S0 launch.

## Invalid DIOR-R ORCNN Diagnostic

- Workdir:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dior_r_orcnn_r50_nan_diag_lr2p5e4_2e_20260607`
- Config:
  `G02_Baselines_Data2_DIOR_R_M5_ORCNN_R50_nan_diag_lr2p5e4_2e_20260607.py`
- Launch log: `launch_20260607_gpu5.log`
- Checkpoints: `epoch_1.pth`, `epoch_2.pth`

First NaN:

- Time: `2026-06-07 10:28:33 +0800`
- Iteration: `Epoch(train) [1][650/5862]`
- LR: `2.5000e-04`
- Signature: `grad_norm: nan`, `loss: nan`

Final validation:

- Time: `2026-06-07 11:07:21 +0800`
- Metric: `dota/mAP=0.0000`, `dota/AP50=0.0000`
- Class behavior: all classes had `dets=0`

Conclusion: reducing LR alone did not fix DIOR-R Oriented R-CNN. Do not cite
this run as DIOR-R baseline evidence.

## DIOR-R RoI Transformer S0

Run identity:

- Workdir:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dior_r_roi_trans_r50_20260607`
- Config:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dior_r_roi_trans_r50_20260607/G02_Baselines_Data2_DIOR_R_M2_RoITrans_20260607.py`
- Base config:
  `/data5/2025/ldh/OpenRSD/M_configs/G02_Baselines/Data2_DIOR_R/G02_Baselines_Data2_DIOR_R_M2_RoITrans.py`
- Dataset: train `DIOR_R_dota/train_val`, validation/test `DIOR_R_dota/test`
- Model: original DIOR-R RoI Transformer, `num_classes=20`
- Optimizer LR: original `0.0025`
- Schedule: `12` epochs, validation every epoch
- Logging/checkpointing: logger interval `25`, checkpoint interval `1`

Launch target:

- GPU: 5
- Screen: `s0_dior_r_roi_trans_r50_20260607_gpu5`
- Launch log:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dior_r_roi_trans_r50_20260607/launch_20260607_gpu5.log`

Command:

```bash
cd /data5/2025/ldh/OpenRSD
CUDA_VISIBLE_DEVICES=5 MPLCONFIGDIR=/tmp/matplotlib_dior_roi_trans \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py \
  tools/train.py \
  work_dirs/s0_dior_r_roi_trans_r50_20260607/G02_Baselines_Data2_DIOR_R_M2_RoITrans_20260607.py \
  --work-dir work_dirs/s0_dior_r_roi_trans_r50_20260607
```

Startup acceptance: screen remains detached, GPU 5 memory/utilization rises,
the log reaches at least `Epoch(train) [1][200/... ]`, and the launch log does
not contain `Traceback`, CUDA OOM, `libpng`, `CRC`, `NoneType`, `ValueError`,
`loss: nan`, or `grad_norm: nan`.

First validation acceptance: epoch 1 validation completes, detections are
nonzero for at least some classes, and `dota/mAP` and `dota/AP50` are not both
`0.0000`.

## Launch Status

Preflight passed on GPU 5 with three idle polls at `14 MiB`, `0%`. The dry
parse with `tools/bootstrap_run.py tools/misc/print_config.py` succeeded and
confirmed DIOR-R `train_val`/`test`, `num_classes=20`, LR `0.0025`, validation
every epoch, checkpoint interval `1`, and logger interval `25`.

Launched at `2026-06-07 11:20 +0800` in detached screen
`s0_dior_r_roi_trans_r50_20260607_gpu5`.

Startup acceptance passed at `2026-06-07 11:21:15 +0800`: the log reached
`Epoch(train) [1][200/5862]`, GPU 5 was active at about `5409 MiB` and `43%`,
and early training had finite `grad_norm` and `loss` with no listed failure
signature.

Acceptance later failed before epoch-1 validation. First RoI Transformer NaN:

- Time: `2026-06-07 11:30:25 +0800`
- Iteration: `Epoch(train) [1][3375/5862]`
- LR: `2.5000e-03`
- Signature: `grad_norm: nan`, `loss: nan`, `loss_rpn_cls: nan`,
  `loss_rpn_bbox: nan`, `s0.loss_cls: nan`, `s0.loss_bbox: nan`,
  `s1.loss_cls: nan`, `s1.loss_bbox: nan`

The screen was stopped after the NaN signature. GPU 5 returned to `14 MiB`,
`0%`; no checkpoint was written before the stop.

Conclusion: DIOR-R NaN is no longer ORCNN-specific. Treat it as a DIOR-R
detector/data/box-coder path issue and diagnose inputs/box conversion/loss
targets before launching another DIOR-R detector.
