# 2026-06-16 DOTA2 S3 Scene-Adapter Loss-0-Best Replicas Complete

## Scope

This note archives the completed DOTA2 S3 scene-adapter replicas launched on
2026-06-15 from each replica's strongest DOTA2 S2 loss-0 early checkpoint.

Dataset/protocol: valid-PNG `DOTA2_1024_500/ss_val`, MMRotate `DOTAMetric`
mAP at IoU 0.5.

## Completion Status

All three S3 replicas completed cleanly. No active DOTA2 S3 training screens
remain. The only remaining project screen is `s0_result_log_monitor_20260603`.

Scoped failure scans over all three S3 workdirs found no matches for
`Traceback`, OOM/out-of-memory, `libpng`, `CRC`, `NoneType`, `ValueError`,
`KeyboardInterrupt`, or `loss/grad_norm nan/inf`.

## Metrics

| Replica | Seed | Source checkpoint | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 final | Best |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| rep3407 | 93407 | S2 rep3407 epoch 1 | 0.6191 | 0.6214 | 0.6144 | 0.6132 | epoch 2, 0.6214 |
| rep4407 | 94407 | S2 rep4407 epoch 3 | 0.6176 | 0.6173 | 0.6159 | 0.6154 | epoch 1, 0.6176 |
| rep5407 | 95407 | S2 rep5407 epoch 1 | 0.6208 | 0.6204 | 0.6113 | 0.6166 | epoch 1, 0.6208 |

Aggregate:

- S3 best mean: `0.6199` (`0.6199271768`).
- S3 final mean: `0.6151` (`0.6150780916`).
- Best single replica: rep3407 epoch 2, `0.6214315891`.

## Comparisons

- DOTA2 S1 final: `0.6177`.
- DOTA2 S2 loss-0 best mean: `0.620606`.
- DOTA2 S2 loss-0 final mean: `0.616655`.
- DOTA2 S3 best mean is below S2 loss-0 best mean.
- DOTA2 S3 final mean is below S2 loss-0 final mean.

## Sources

- rep3407 scalar file:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep3407_20260615/20260615_150453/vis_data/scalars.json`
- rep4407 scalar file:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep4407_20260615/20260615_150453/vis_data/scalars.json`
- rep5407 scalar file:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep5407_20260615/20260615_150453/vis_data/scalars.json`

## Decision

DOTA2 S3 is clean but not stronger than S2 loss-0. Treat this as
exploratory/negative-to-neutral DOTA2 S3 evidence. Do not launch S4,
pseudo-labeling, FAIR1M, or route-changing experiments from this result.
