# 2026-06-03 GeoNexus S1 Rerun Launch

## Scope

- Stage: S1 RemoteCLIP prompt-head rerun for DOTA v1.5.
- S0 detector runs from 2026-06-03 were left untouched.
- The completed DOTA2 valid-PNG RoI Transformer result remains the current S0 DOTA2 evidence only; it was not mixed into this 16-class DOTA v1.5 S1 path.

## Real VLM Smoke

- Command: `smoke_vlm_embeddings.py --embedding-backend remoteclip --embedding-model-name ViT-B-32 --embedding-checkpoint /data5/2025/ldh/OpenRSD/checkpoints/remoteclip/RemoteCLIP-ViT-B-32.pt --embedding-device cuda`
- Environment: `/home/zwl/.conda/envs/geonexus_vlm/bin/python`
- Result: passed with `classes=16` and `embedding_shape=[16, 512]`.

## S1 Launch

- Screen: `3118395.geonexus_s1_rerun_20260603_gpu1`
- GPU: 1
- Config: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_rerun_20260603/roi-trans-le90_r50_fpn_remoteclip-s1-rerun-20260603_dota15.py`
- Work dir: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_rerun_20260603`
- Log: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_rerun_20260603/launch_20260603_gpu1.log`
- Initializer: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/roi_trans_lr001_3x/epoch_36.pth`
- Startup threshold: reached `Epoch(train)` through at least epoch 1 iter 30 at 2026-06-03 18:00:56 +0800.

## Checks

- No matches in the launch log for `Traceback`, `CUDA out of memory`, `libpng`, `CRC`, or `NoneType` at 2026-06-03 18:01:17 +0800.
- GPU 1 showed PID `3118592` using about 10.7 GiB after startup.

## Failure Update

- Status checked after failure at 2026-06-03 19:09 +0800.
- `screen -ls` showed only S0 DOTA2 screens; no S1 screen remained active.
- Launch log ended with `EXIT 1 2026-06-03T18:02:19+08:00`.
- Failure class: `CUDA out of memory`.
- Failure point: epoch 1 iter 190, during RPN target assignment / overlap
  calculation.
- No epoch checkpoint was produced in the S1 rerun work dir, so the next safe
  retry must restart from epoch 0 unless a future retry creates
  `last_checkpoint`.
- Do not launch S2 from this failed run.

## Monitoring And Retry Policy

- Every monitoring pass must check `screen -ls`, `nvidia-smi`, and the active
  run log before reporting status.
- If the run failed, read the traceback and classify the failure before any
  relaunch.
- For `CUDA out of memory`, wait for an allowed physical GPU with
  `memory.used <= 4000 MiB` and `util <= 10%` for three consecutive polls, then
  relaunch there.
- For `libpng`, `CRC`, `NoneType`, or other data-read errors, identify the bad
  file or sample first; do not relaunch unchanged unless the bad input is fixed
  or excluded.
- For import/config errors, fix the environment or config first; do not relaunch
  unchanged.
- For an unknown traceback, record the traceback and relaunch once on a clean
  GPU. If the same traceback repeats, stop and require a fix.
- Cap automatic retries at 3 for this experiment. Each retry must use a fresh
  log name with retry index and physical GPU, and this handoff note must record
  the failure reason plus restart command.
- If `last_checkpoint` exists for a future failure, resume from it; otherwise
  restart from epoch 0.

## Safe Restart Command Template

Current retry should restart from scratch because no epoch checkpoint exists:

```bash
cd /data5/2025/ldh/OpenRSD
CUDA_VISIBLE_DEVICES=<physical_gpu> MPLCONFIGDIR=/tmp/matplotlib_geonexus_s1_rerun_retry<retry_index> \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py \
  /data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_rerun_20260603/roi-trans-le90_r50_fpn_remoteclip-s1-rerun-20260603_dota15.py \
  --work-dir /data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_rerun_20260603
```

Run it in a detached screen and write stdout/stderr to a new log named like:

`launch_retry<retry_index>_20260603_gpu<physical_gpu>.log`

## Retry 1 Launch

- Gate: physical GPU 1 passed three consecutive OOM-retry polls:
  - 19:10 +0800: `3499 MiB`, `0%`
  - 19:12 +0800: `3499 MiB`, `5%`
  - 19:14 +0800: `3499 MiB`, `0%`
- Screen: `3300640.geonexus_s1_rerun_retry1_20260603_gpu1`
- GPU: 1
- Log:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_rerun_20260603/launch_retry1_20260603_gpu1.log`
- Inner mmengine log:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_rerun_20260603/20260603_191534/20260603_191534.log`
- PID observed after startup: `3300816`
- Restart command:

```bash
screen -dmS geonexus_s1_rerun_retry1_20260603_gpu1 bash -lc 'cd /data5/2025/ldh/OpenRSD && echo START_RETRY1_GPU1 $(date -Is) > work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_rerun_20260603/launch_retry1_20260603_gpu1.log && CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_geonexus_s1_rerun_retry1 /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py /data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_rerun_20260603/roi-trans-le90_r50_fpn_remoteclip-s1-rerun-20260603_dota15.py --work-dir /data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_rerun_20260603 >> work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_rerun_20260603/launch_retry1_20260603_gpu1.log 2>&1; status=$?; echo EXIT_RETRY1_GPU1 $status $(date -Is) >> work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_rerun_20260603/launch_retry1_20260603_gpu1.log'
```

- Startup verification: retry reached `Epoch(train) [1][30/1410]` at
  `2026-06-03 19:18:07 +0800`, then continued through iter 190.
- Retry 1 failed at `2026-06-03 19:19:11 +0800` with the same failure class:
  `CUDA out of memory` during dense assignment. The log reports a failed
  allocation of `2.05 GiB` with only `878 MiB` free.
- No S1 screen remained active after retry 1. Do not launch S2.

## Config Mitigation After Retry 1

The repeated CUDA OOM matched the dense RPN/RCNN assignment path already fixed
in the corrected S2/S3 configs. The S1 rerun runtime config was updated to add
`gpu_assign_thr=256` to:

- `model.train_cfg.rpn.assigner`
- `model.train_cfg.rcnn[0].assigner`
- `model.train_cfg.rcnn[1].assigner`

Edited config:

`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_rerun_20260603/roi-trans-le90_r50_fpn_remoteclip-s1-rerun-20260603_dota15.py`

Retry 2 must still wait for the CUDA-OOM gate: an allowed physical GPU with
`memory.used <= 4000 MiB` and `util <= 10%` for three consecutive polls.

## Next Step

- Keep S2 sequenced behind this S1 rerun. Do not launch a new S2 until the S1
  rerun produces the checkpoint intended for S2 initialization.
