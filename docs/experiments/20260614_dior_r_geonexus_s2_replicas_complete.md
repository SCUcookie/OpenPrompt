# 2026-06-14 DIOR-R GeoNexus S2 Replicas Complete

## Scope

This note archives the completed DIOR-R S2 hierarchy-regularizer replicas that
were launched from DIOR-R S1 rep0 epoch 12 on 2026-06-13 and the three
confirmation replicas launched on 2026-06-14.

Dataset/protocol: sanitized `DIOR_R_dota/train_val` to `DIOR_R_dota/test`,
MMRotate `DOTAMetric` mAP at IoU 0.5.

## Completion Status

No active DIOR-R S2 training screens remain. The only remaining project screen
is `s0_result_log_monitor_20260603`.

Scoped failure scans over all six S2 workdirs found no matches for
`Traceback`, CUDA OOM/out-of-memory, `libpng`, `CRC`, `NoneType`, `ValueError`,
`KeyboardInterrupt`, `loss: nan/inf`, or `grad_norm: nan/inf`.

## Metrics

| Replica | Seed | Epoch 4 | Epoch 8 | Epoch 12 final | Best |
| --- | ---: | ---: | ---: | ---: | --- |
| rep0 | 7407 | 0.6802 | 0.6878 | 0.6858 | epoch 8, 0.6878 |
| rep1 | 8407 | 0.6811 | 0.6905 | 0.6833 | epoch 8, 0.6905 |
| rep2 | 9407 | 0.6833 | 0.6866 | 0.6868 | epoch 12, 0.6868 |
| rep3 | 10407 | 0.6805 | 0.6873 | 0.6808 | epoch 8, 0.6873 |
| rep4 | 11407 | 0.6826 | 0.6884 | 0.6914 | epoch 12, 0.6914 |
| rep5 | 12407 | 0.6787 | 0.6882 | 0.6857 | epoch 8, 0.6882 |

Aggregate:

- S2 best mean: `0.6887` (`0.6886706153`).
- S2 final mean: `0.6856` (`0.6856140991`).
- Best single replica: rep4 epoch 12, `0.6914003491`.

## Sources

- rep0 scalar file:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep0_20260613/20260613_154141/vis_data/scalars.json`
- rep1 scalar file:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep1_20260613/20260613_154141/vis_data/scalars.json`
- rep2 scalar file:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep2_20260613/20260613_154141/vis_data/scalars.json`
- rep3 scalar file:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep3_20260614/20260614_111933/vis_data/scalars.json`
- rep4 scalar file:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep4_20260614/20260614_111933/vis_data/scalars.json`
- rep5 scalar file:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep5_20260614/20260614_111934/vis_data/scalars.json`

## Decision

DIOR-R now has a clean S0 -> S1 -> S2 progression. S2 is positive over S0 and
S1 on both best-checkpoint mean and final-checkpoint mean across six replicas.
Use the strongest S2 checkpoint, rep4 epoch 12, as the source for the S3
scene-adapter replicas. Keep DOTA2 S2 classified separately as
early-checkpoint-positive but final-unstable.
