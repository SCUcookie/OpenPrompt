# 2026-06-16 DIOR-R S3 Stability E4 LR2.5e-5 Complete

## Scope

This note archives the completed DIOR-R S3 final-stability follow-up launched
from each LR5e-5 stability replica's epoch-4 checkpoint with LR `2.5e-5` for
four more epochs.

Dataset/protocol: sanitized `DIOR_R_dota/train_val` to `DIOR_R_dota/test`,
MMRotate `DOTAMetric` mAP at IoU 0.5.

## Completion Status

All three annealed stability replicas completed cleanly with exit code 0. No
active training screens remain; the only remaining project screen is
`s0_result_log_monitor_20260603`.

Scoped failure scans over the three workdirs found no matches for `Traceback`,
OOM/out-of-memory, `libpng`, `CRC`, `NoneType`, `ValueError`,
`KeyboardInterrupt`, or `loss/grad_norm nan/inf`.

## Metrics

| Replica | Source checkpoint | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 final | Best |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| rep0 | LR5e-5 stability rep0 epoch 4 | 0.691012 | 0.689341 | 0.695021 | 0.692381 | epoch 3, 0.695021 |
| rep1 | LR5e-5 stability rep1 epoch 4 | 0.686839 | 0.686984 | 0.687204 | 0.687131 | epoch 3, 0.687204 |
| rep2 | LR5e-5 stability rep2 epoch 4 | 0.687636 | 0.688336 | 0.690285 | 0.688153 | epoch 3, 0.690285 |

Aggregate:

- Annealed stability best mean: `0.6908363303`.
- Annealed stability final mean: `0.6892216007`.
- Best single replica: rep0 epoch 3, `0.6950206161`.

## Comparisons

- Previous LR5e-5 stability best mean: `0.692193`.
- Previous LR5e-5 stability final mean: `0.690303`.
- Original DIOR-R S3 best mean: `0.6979`.
- Original DIOR-R S3 final mean: `0.6859`.
- DIOR-R S2 final mean: `0.6856`.

## Decision

By the 2026-06-16 decision rules, this is a neutral final-stability result:
the final mean `0.689222` is above DIOR-R S2 final mean `0.6856`, but below
the useful threshold `0.6903`. Archive the result, keep best and final metrics
separate, and do not extend automatically. Do not launch S4, pseudo-labeling,
FAIR1M, routing, or DOTA2 follow-ups from this result without a separate route
decision.
