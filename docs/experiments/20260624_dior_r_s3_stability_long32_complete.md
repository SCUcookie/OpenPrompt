# 2026-06-24 DIOR-R S3 Stability Long32 Complete

## Scope

This note archives the completed DIOR-R S3 long-stability continuation launched
on `2026-06-23` from the 2026-06-16 annealed stability epoch-4 checkpoints.

Launch provenance:
`New/docs/experiments/20260623_dior_r_s3_stability_long32_launch.md`.

Dataset/protocol: sanitized `DIOR_R_dota/train_val` to `DIOR_R_dota/test`,
MMRotate `DOTAMetric` mAP at IoU 0.5.

S4, pseudo-labeling, FAIR1M, DOTA2 follow-up training, and route changes remain
outside this archive.

## Completion Status

All three long32 replicas completed through epoch 32 on `2026-06-23 CST`.
The only remaining `screen` session observed during archive preflight was
`s0_result_log_monitor_20260603`.

Scoped failure scans across the three launch logs and runtime logs found no
matches for `Traceback`, CUDA OOM, `out-of-memory`, `out of memory`, `libpng`,
`CRC`, `NoneType`, `ValueError`, `KeyboardInterrupt`, `loss: nan`,
`loss: inf`, `grad_norm: nan`, or `grad_norm: inf`.

## Metrics

| Replica | Final epoch 32 `dota/mAP` | Final epoch 32 `dota/AP50` | Best rounded mAP | Best epoch |
| --- | ---: | ---: | ---: | ---: |
| rep0 | 0.6960 | 0.6960 | 0.6990 | 28 |
| rep1 | 0.6897 | 0.6900 | 0.6921 | 30 |
| rep2 | 0.6941 | 0.6940 | 0.6984 | 25 |

Aggregate:

- Final mean rounded mAP: `0.6933`.
- Best mean rounded mAP: `0.6965`.

## Decision

Treat long32 as useful final-stability evidence, not a new best-checkpoint
claim. The best mean `0.6965` remains below the earlier original DIOR-R S3 best
mean `0.6979`.

Approved follow-up: launch DIOR-R S3 long60 continuation from each long32
`epoch_32.pth`. Keep S4, pseudo-labeling, FAIR1M, DOTA2 follow-up training,
and route changes paused unless separately approved.
