# DIOR-R Oriented R-CNN Baseline Launch

Date: 2026-06-06

## Purpose

Use the newly freed GPU 1 for the DOTA2 + cross-dataset pivot by launching the
required DIOR-R S0 closed-set baseline. This is the first paper-path DIOR-R run
after the DOTA v1.5 archive pivot.

## Run

- Dataset: `DIOR_R_dota`
- Train split: `train_val/labelTxt`, `train_val/images`
- Validation/test split: `test/labelTxt`, `test/images`
- Detector: Oriented R-CNN R50
- Source config:
  `/data5/2025/ldh/OpenRSD/M_configs/G02_Baselines/Data2_DIOR_R/G02_Baselines_Data2_DIOR_R_M5_ORCNN_R50.py`
- Runtime config:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dior_r_orcnn_r50_20260606/G02_Baselines_Data2_DIOR_R_M5_ORCNN_R50.py`
- Workdir:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dior_r_orcnn_r50_20260606`
- Launch log:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dior_r_orcnn_r50_20260606/launch_20260606_gpu1.log`
- Screen:
  `s0_dior_orcnn_r50_20260606_gpu1`
- GPU:
  physical GPU 1

## Command

```bash
cd /data5/2025/ldh/OpenRSD
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_dior_orcnn_20260606 \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py \
  tools/train.py \
  M_configs/G02_Baselines/Data2_DIOR_R/G02_Baselines_Data2_DIOR_R_M5_ORCNN_R50.py \
  --work-dir work_dirs/s0_dior_r_orcnn_r50_20260606
```

## Startup Acceptance

Accepted at `2026-06-06 15:30 +0800`:

- `screen -ls` showed `s0_dior_orcnn_r50_20260606_gpu1` detached and active.
- GPU 1 was using about `5043 MiB` with nonzero utilization.
- Log reached `Epoch(train) [1][100/5862]`.
- No `Traceback`, CUDA OOM, `libpng`, `CRC`, `NoneType`, or early-exit
  signature was observed.

## Next Check

Monitor for the first validation at epoch 4. Record final DOTA-style mAP/AP50
with the exact checkpoint and metric source before launching a DIOR-R GeoNexus
variant.
