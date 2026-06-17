# 2026-06-15 DIOR-R GeoNexus S3 Scene-Adapter Replicas Complete

## Scope

This note archives the completed DIOR-R S3 scene-adapter replicas launched on
2026-06-14 from the strongest DIOR-R S2 hierarchy checkpoint, rep4 epoch 12.

Dataset/protocol: sanitized `DIOR_R_dota/train_val` to `DIOR_R_dota/test`,
MMRotate `DOTAMetric` mAP at IoU 0.5.

## Completion Status

All three S3 replicas completed cleanly. No active DIOR-R S3 training screens
remain. The only remaining project screen is `s0_result_log_monitor_20260603`.
All GPUs reported 14 MiB used and 0% utilization after completion.

Scoped failure scans over all three S3 workdirs found no matches for
`Traceback`, CUDA OOM/out-of-memory, `libpng`, `CRC`, `NoneType`, `ValueError`,
`KeyboardInterrupt`, `loss: nan/inf`, or `grad_norm: nan/inf`.

## Metrics

| Replica | Seed | Epoch 4 | Epoch 8 | Epoch 12 final | Best |
| --- | ---: | ---: | ---: | ---: | --- |
| rep0 | 13407 | 0.6940 | 0.6992 | 0.6793 | epoch 8, 0.6992 |
| rep1 | 14407 | 0.6903 | 0.6990 | 0.6871 | epoch 8, 0.6990 |
| rep2 | 15407 | 0.6928 | 0.6956 | 0.6912 | epoch 8, 0.6956 |

Aggregate:

- S3 best mean: `0.6979` (`0.6979383032`).
- S3 final mean: `0.6859` (`0.6858827869`).
- Best single replica: rep0 epoch 8, `0.6991876364`.

## Sources

- Source checkpoint:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep4_20260614/epoch_12.pth`
- Source S2 metric: rep4 epoch 12, `dota/mAP=0.6914003491`.
- rep0 scalar file:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_s2e12_rep0_20260614/20260614_161203/vis_data/scalars.json`
- rep1 scalar file:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_s2e12_rep1_20260614/20260614_161203/vis_data/scalars.json`
- rep2 scalar file:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_s2e12_rep2_20260614/20260614_161203/vis_data/scalars.json`

## Decision

DIOR-R S3 is strong best-checkpoint scene-adapter evidence over S2. The final
checkpoint evidence is only roughly tied with S2, so paper text should report
best and final means separately and avoid presenting S3 final checkpoints as a
clear gain. Keep pseudo-labeling, FAIR1M, and S4 paused until a separate route
decision.
