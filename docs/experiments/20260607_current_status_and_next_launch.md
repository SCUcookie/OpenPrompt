# Current Status And DOTA2 GeoNexus S1 Launch - 2026-06-07

This note records the DOTA2-first paper-path state after the 2026-06-06 pivot.

## Completed Or Invalid Runs

- DOTA2 R3Det-KFIoU valid-PNG bs1 completed epoch 12 with
  `dota/mAP=0.5633`, `dota/AP50=0.5630`.
  Checkpoint:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_r3det_kfiou_validpng_bs1_20260603/epoch_12.pth`.
  Metric log:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_r3det_kfiou_validpng_bs1_20260603/20260605_100954/20260605_100954.log`.
- DIOR-R Oriented R-CNN R50 completed epoch 12 but is invalid evidence.
  Training hit `loss: nan` from epoch 2 onward, and epoch 4/8/12 validation
  stayed `0.0000/0.0000`. Preserve the checkpoint/logs, but do not cite this
  as a DIOR-R baseline.
- DIOR-R Oriented R-CNN R50 low-LR diagnostic completed epoch 2 but is also
  invalid evidence. It first hit NaN at `2026-06-07 10:28:33 +0800`,
  `Epoch(train) [1][650/5862]`, `lr=2.5000e-04`, with `grad_norm: nan` and
  `loss: nan`; final validation at `2026-06-07 11:07:21 +0800` was
  `dota/mAP=0.0000`, `dota/AP50=0.0000`, with all classes at `dets=0`.
  Low LR alone did not fix DIOR-R ORCNN, so do not cite it as DIOR-R baseline
  evidence.
- DOTA2 RTMDet-L valid-PNG bs1 completed epoch 12 at
  `2026-06-07 15:04:34 +0800` with `dota/mAP=0.2779`,
  `dota/AP50=0.2780`. Final checkpoint:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_rtmdet_l_validpng_bs1_20260603/epoch_12.pth`.
  The final metric degraded from epoch 8 `0.3521/0.3520`; do not prioritize
  RTMDet-L further.

## Live GPU State

Checked on 2026-06-07 before launch:

- GPU 6: RTMDet-L completed; GPU 6 was idle before preparing the low-LR S1
  replicate.
- GPU 1: active DOTA2 GeoNexus S1 screen
  `geonexus_dota2_roi_trans_s1_validpng_20260607_gpu1`.
- GPU 4: occupied by another Python job at about `3775 MiB`.
- GPUs 0/2/3/5 appeared idle before the DIOR-R RoI Transformer launch. After
  the stopped DIOR-R RoI Transformer NaN attempt, GPU 5 returned to `14 MiB`,
  `0%`.

## DOTA2 Taxonomy And Prompt Artifact

Created DOTA2 taxonomy:

- `/data5/2025/ldh/New/assets/hierarchies/dota2_remote_sensing_taxonomy.json`

Class order matches the DOTA2 config:

`airport`, `baseball-diamond`, `basketball-court`, `bridge`,
`container-crane`, `ground-track-field`, `harbor`, `helicopter`, `helipad`,
`large-vehicle`, `plane`, `roundabout`, `ship`, `small-vehicle`,
`soccer-ball-field`, `storage-tank`, `swimming-pool`, `tennis-court`.

Generated RemoteCLIP prompt artifact:

- `/data5/2025/ldh/New/artifacts/generated/remoteclip_vit_b32_dota2_prompt_embeddings.pt`

Validation: `class_names` length 18, `embeddings` shape `[18, 512]`, all
finite, and L2 norms within floating-point tolerance of 1.0.

## Launched DOTA2 GeoNexus S1

Run:

- Screen: `geonexus_dota2_roi_trans_s1_validpng_20260607_gpu1`
- Workdir:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607`
- Config:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607/roi-trans-le90_r50_fpn_remoteclip-s1-validpng-20260607_dota2.py`
- Launch log:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607/launch_20260607_gpu1.log`
- Base detector checkpoint:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/epoch_12.pth`
- Dataset: `DOTA2_1024_500`, train
  `train/annfiles_validpng_20260602`, validation `ss_val/annfiles`.

Launch command:

```bash
cd /data5/2025/ldh/OpenRSD
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_geonexus_dota2_s1 \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py \
  tools/train.py \
  work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607/roi-trans-le90_r50_fpn_remoteclip-s1-validpng-20260607_dota2.py \
  --work-dir work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607
```

Preflight:

- Prompt artifact validation passed.
- `tools/bootstrap_run.py tools/misc/print_config.py ... --cfg-options
  train_cfg.max_epochs=1` parsed the config and confirmed both cascade heads
  are `PromptShared2FCBBoxHead`, `num_classes=18`, using the DOTA2 RemoteCLIP
  artifact.

Startup acceptance target: the screen stays detached, GPU 1 memory/utilization
rises, the launch log reaches `Epoch(train) [1][200/... ]`, and no
`Traceback`, CUDA OOM, `libpng`, `CRC`, `NoneType`, `ValueError`, or prompt
class-count mismatch appears.

Startup acceptance passed at `2026-06-07 10:16:36 +0800`: the log reached
`Epoch(train) [1][200/39007]`, GPU 1 was active at about `14207 MiB` and
`45%`, and the failure-signature scan found no listed errors.

Compare the first validation against the DOTA2 RoI Transformer S0 baseline
`dota/mAP=0.6088`, `dota/AP50=0.6090`. Do not launch S2, S3, S4, FAIR1M, or
additional secondary baselines until S1 reaches validation cleanly and yields a
credible metric or diagnostic.

## DIOR-R RoI Transformer S0 Attempt

Because DIOR-R Oriented R-CNN has now failed twice with NaN and zero
validation evidence, DIOR-R RoI Transformer S0 was launched on GPU 5.

- Screen: `s0_dior_r_roi_trans_r50_20260607_gpu5`
- Workdir:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dior_r_roi_trans_r50_20260607`
- Config:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dior_r_roi_trans_r50_20260607/G02_Baselines_Data2_DIOR_R_M2_RoITrans_20260607.py`
- Launch log:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dior_r_roi_trans_r50_20260607/launch_20260607_gpu5.log`

The child config inherits
`M_configs/G02_Baselines/Data2_DIOR_R/G02_Baselines_Data2_DIOR_R_M2_RoITrans.py`,
keeps the original DIOR-R dataset, model, `num_classes=20`, and optimizer LR
`0.0025`, and overrides only run identity, `resume=False`, `load_from=None`,
`train_cfg.max_epochs=12`, `train_cfg.val_interval=1`, checkpoint interval
`1`, and logger interval `25`.

Preflight passed with three GPU-5 idle polls at `14 MiB`, `0%`, and
`tools/bootstrap_run.py tools/misc/print_config.py` parsed the config.
Startup acceptance passed at `2026-06-07 11:21:15 +0800`: detached screen
`s0_dior_r_roi_trans_r50_20260607_gpu5`, GPU 5 active at about `5409 MiB` and
`43%`, log reached `Epoch(train) [1][200/5862]`, and early `grad_norm`/`loss`
were finite with no listed failure signature.

Acceptance later failed before epoch-1 validation. RoI Transformer first hit
NaN at `2026-06-07 11:30:25 +0800`, `Epoch(train) [1][3375/5862]`,
`lr=2.5000e-03`, with `grad_norm: nan`, `loss: nan`, and NaN RPN/cascade
losses. The screen was stopped; GPU 5 returned to `14 MiB`, `0%`, and no
checkpoint was written. Treat the DIOR-R failure as a detector/data/box-coder
path issue, not an ORCNN-specific instability. Do not launch another DIOR-R
detector unchanged.

## Prepared Fill-GPU Runs

### DOTA2 GeoNexus S1 Low-LR Replicate

- GPU: 6
- Screen: `geonexus_dota2_roi_trans_s1_validpng_lr1e4_20260607_gpu6`
- Workdir:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_lr1e4_20260607`
- Config:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_lr1e4_20260607/roi-trans-le90_r50_fpn_remoteclip-s1-validpng-lr1e4-20260607_dota2.py`
- Change from active S1: optimizer LR `1e-4` and workdir/log identity only.
- Initialization:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/epoch_12.pth`

Command:

```bash
cd /data5/2025/ldh/OpenRSD
CUDA_VISIBLE_DEVICES=6 MPLCONFIGDIR=/tmp/geonexus_dota2_s1_lr1e4_20260607 \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py \
  tools/train.py \
  work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_lr1e4_20260607/roi-trans-le90_r50_fpn_remoteclip-s1-validpng-lr1e4-20260607_dota2.py \
  --work-dir work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_lr1e4_20260607
```

### DIOR-R Rotated RetinaNet One-Stage NaN Probe

- GPU: 5
- Screen: `s0_dior_r_rotated_retinanet_nan_probe_2e_20260607_gpu5`
- Workdir:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dior_r_rotated_retinanet_r50_nan_probe_2e_20260607`
- Config:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dior_r_rotated_retinanet_r50_nan_probe_2e_20260607/G02_Baselines_Data2_DIOR_R_M1_RtnNetOBB_nan_probe_2e_20260607.py`
- Dataset: local `DIOR_R_dota/train_val` and `DIOR_R_dota/test`
- Schedule: `max_epochs=2`, `val_interval=1`, checkpoint every epoch
- Optimizer LR: `1e-4`

Command:

```bash
cd /data5/2025/ldh/OpenRSD
CUDA_VISIBLE_DEVICES=5 MPLCONFIGDIR=/tmp/dior_r_retinanet_nan_probe_2e_20260607 \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py \
  tools/train.py \
  work_dirs/s0_dior_r_rotated_retinanet_r50_nan_probe_2e_20260607/G02_Baselines_Data2_DIOR_R_M1_RtnNetOBB_nan_probe_2e_20260607.py \
  --work-dir work_dirs/s0_dior_r_rotated_retinanet_r50_nan_probe_2e_20260607
```

Startup acceptance for both runs: reach `Epoch(train) [1][200/...]` with no
`Traceback`, CUDA OOM, `libpng`, `CRC`, `NoneType`, `ValueError`,
class-count mismatch, `loss: nan`, or `grad_norm: nan`. If the DIOR-R
RetinaNet probe hits NaN, stop it and record the first NaN timestamp, epoch,
iteration, LR, and loss fields before any additional DIOR-R training.
