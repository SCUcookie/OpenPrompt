# 2026-06-15 DIOR-R S3 Epoch-8 LR5e-5 Stability Complete

## Scope

This note archives the completed DIOR-R S3 stability follow-up launched from
each S3 scene-adapter replica's epoch-8 checkpoint with LR `5e-5` for four
epochs.

Dataset/protocol: sanitized `DIOR_R_dota/train_val` to `DIOR_R_dota/test`,
MMRotate `DOTAMetric` mAP at IoU 0.5.

## Completion Status

All three stability replicas completed cleanly.

Scoped failure scans over the three stability workdirs found no matches for
`Traceback`, OOM, `libpng`, `CRC`, `NoneType`, `ValueError`,
`KeyboardInterrupt`, `loss/grad_norm nan/inf`.

## Metrics

| Replica | Source | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 final | Best |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| rep0 | S3 epoch-8 LR5e-5 | 0.693571 | 0.697102 | 0.691667 | 0.691846 | epoch 2, 0.697102 |
| rep1 | S3 epoch-8 LR5e-5 | 0.687566 | 0.686475 | 0.687118 | 0.687152 | epoch 1, 0.687566 |
| rep2 | S3 epoch-8 LR5e-5 | 0.690737 | 0.685418 | 0.687939 | 0.691912 | epoch 4, 0.691912 |

Aggregate:

- Stability best mean: `0.692193`.
- Stability final mean: `0.690303`.

## Decision

The epoch-8 LR5e-5 stability continuation is lower than the original DIOR-R S3
best mean `0.6979`, but improves final stability over the original S3 final
mean `0.6859` and S2 final mean `0.6856`. Report best and final means
separately; do not use this as evidence for S4, pseudo-labeling, or FAIR1M.
