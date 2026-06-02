# 2026-06-02 S0 DOTA2 Oriented R-CNN R50 Valid-PNG Launch

## Scope

This record is only for the S0 DOTA2 Oriented R-CNN R50 baseline launched with
the valid-PNG filtered training annotations. It is not S1/S2/S3/S4 evidence.

## Config

- Source config: `/data5/2025/ldh/OpenRSD/M_configs/G02_Baselines/Data1_DOTA2/G02_Baselines_Data1_DOTA2_M5_ORCNN_R50.py`.
- Runtime config: `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_orcnn_r50_validpng_20260602/G02_Baselines_Data1_DOTA2_M5_ORCNN_R50_validpng_20260602.py`.
- Work dir: `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_orcnn_r50_validpng_20260602`.
- Train annotations: `train/annfiles_validpng_20260602/`.
- Validation annotations: `ss_val/annfiles/`.
- `max_epochs = 12`.
- `val_interval = 4`.
- `ckpt_interval = 4`.
- `load_from = None`.
- `resume = False`.

The runtime config was copied from the source config and changed only to use
the filtered training annotation directory.

## Launch

- Launch time: `2026-06-02 14:53 +0800`.
- Screen: `s0_dota2_orcnn_r50_validpng_20260602_gpu1`.
- Screen PID: `1598562`.
- Training PID: `1598732`.
- GPU: `1`.
- Launch log: `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_orcnn_r50_validpng_20260602/launch_20260602.log`.
- MMEngine log: `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_orcnn_r50_validpng_20260602/20260602_145313/20260602_145313.log`.

Command:

```bash
cd /data5/2025/ldh/OpenRSD
screen -dmS s0_dota2_orcnn_r50_validpng_20260602_gpu1 bash -lc '
  cd /data5/2025/ldh/OpenRSD
  CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_s0_dota2_orcnn_validpng_20260602 \
    /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py \
    work_dirs/s0_dota2_1024_500_orcnn_r50_validpng_20260602/G02_Baselines_Data1_DOTA2_M5_ORCNN_R50_validpng_20260602.py \
    --work-dir work_dirs/s0_dota2_1024_500_orcnn_r50_validpng_20260602 \
    >> work_dirs/s0_dota2_1024_500_orcnn_r50_validpng_20260602/launch_20260602.log 2>&1
'
```

## Verification Status

Launch verification passed:

- The screen exists as `1598562.s0_dota2_orcnn_r50_validpng_20260602_gpu1`.
- PID `1598732` is active under the launch shell.
- GPU 1 shows PID `1598732` as `/data1/anaconda3/envs/zwl_mmrotate/bin/python`.
- The launch log entered MMEngine startup, hook registration, and the filtered
  annotation preparation pass over `170831` annotations.

Latest observed progress at `2026-06-02 14:59 +0800`:

```text
3%|▎         | 4295/170831 [02:02<1:51:24, 24.91it/s]
```

No `libpng`, `NoneType`, `CRC`, or `Traceback` crash signature was present in
the launch log through this check.

Training-iteration verification is still pending. Do not cite this as a
completed ORCNN run or as recovered iteration-level evidence until the log
enters `Epoch(train)` and reaches at least epoch 1 iteration `[1600/39007]` or
the equivalent denominator without PNG-related crash signatures.
