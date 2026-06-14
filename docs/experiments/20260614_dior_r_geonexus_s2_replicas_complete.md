# 2026-06-14 DIOR-R GeoNexus S2 Replicas Complete

## Scope

This note archives the completed DIOR-R S2 hierarchy-regularizer replicas that
were launched from DIOR-R S1 rep0 epoch 12 on 2026-06-13.

Dataset/protocol: sanitized `DIOR_R_dota/train_val` to `DIOR_R_dota/test`,
MMRotate `DOTAMetric` mAP at IoU 0.5.

## Completion Status

No active DIOR-R S2 training screens remain. The only remaining project screen
is `s0_result_log_monitor_20260603`.

Scoped failure scans over all three S2 workdirs found no matches for
`Traceback`, CUDA OOM/out-of-memory, `libpng`, `CRC`, `NoneType`, `ValueError`,
`KeyboardInterrupt`, `loss: nan/inf`, or `grad_norm: nan/inf`.

## Metrics

| Replica | Seed | Epoch 4 | Epoch 8 | Epoch 12 final | Best |
| --- | ---: | ---: | ---: | ---: | --- |
| rep0 | 7407 | 0.6802 | 0.6878 | 0.6858 | epoch 8, 0.6878 |
| rep1 | 8407 | 0.6811 | 0.6905 | 0.6833 | epoch 8, 0.6905 |
| rep2 | 9407 | 0.6833 | 0.6866 | 0.6868 | epoch 12, 0.6868 |

Aggregate:

- S2 best mean: `0.6884`.
- S2 final mean: `0.6853`.
- Best single replica: rep1 epoch 8, `0.6905`.

## Sources

- rep0 scalar file:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep0_20260613/20260613_154141/vis_data/scalars.json`
- rep1 scalar file:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep1_20260613/20260613_154141/vis_data/scalars.json`
- rep2 scalar file:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep2_20260613/20260613_154141/vis_data/scalars.json`

## Decision

DIOR-R now has a clean S0 -> S1 -> S2 progression. S2 is positive over S0 and
S1 on both best-checkpoint mean and final-checkpoint mean. Keep DOTA2 S2
classified separately as early-checkpoint-positive but final-unstable.
