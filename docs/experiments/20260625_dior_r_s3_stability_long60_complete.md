# 2026-06-25 DIOR-R S3 Stability Long60 Complete

## Scope

Archive the completed DIOR-R S3 long60 continuation launched on `2026-06-24`
from the completed long32 `epoch_32.pth` checkpoints.

Launch provenance:
`New/docs/experiments/20260624_dior_r_s3_stability_long60_launch.md`.

Dataset/protocol: sanitized `DIOR_R_dota/train_val` to `DIOR_R_dota/test`,
MMRotate `DOTAMetric` mAP at IoU 0.5.

S4, pseudo-labeling, FAIR1M, DOTA2 follow-up training, and route changes remain
outside this archive.

## Completion Status

All three long60 replicas completed through epoch 60 on `2026-06-25 CST`.
The only remaining `screen` session observed during archive preflight was
`s0_result_log_monitor_20260603`.

Scoped failure scans across the three launch logs and runtime logs found no
matches for `Traceback`, CUDA OOM, `out-of-memory`, `out of memory`, `libpng`,
`CRC`, `NoneType`, `ValueError`, `KeyboardInterrupt`, `loss: nan`,
`loss: inf`, `grad_norm: nan`, or `grad_norm: inf`.

## Metrics

| Replica | Final epoch 60 `dota/mAP` | Final epoch 60 `dota/AP50` | Best exact mAP | Best epoch |
| --- | ---: | ---: | ---: | ---: |
| rep0 | 0.694532 | 0.6950 | 0.698892 | 51 |
| rep1 | 0.688295 | 0.6880 | 0.692448 | 33 |
| rep2 | 0.696215 | 0.6960 | 0.698467 | 58 |

Aggregate:

- Final mean exact mAP: `0.693014`.
- Best mean exact mAP: `0.696602`.

## Decision

Treat long60 as useful final-stability evidence, not a new best-checkpoint
claim. The best mean remains below the earlier original DIOR-R S3 best mean
`0.6979`, and the final mean is roughly tied with the long32 final-stability
evidence.

Approved follow-up: launch DIOR-R S3 long88 continuation from each long60
`epoch_60.pth`. Keep all three replicas for controlled aggregate stability,
including rep1 despite its lower final epoch. Keep S4, pseudo-labeling, FAIR1M,
DOTA2 follow-up training, and route changes paused unless separately approved.
