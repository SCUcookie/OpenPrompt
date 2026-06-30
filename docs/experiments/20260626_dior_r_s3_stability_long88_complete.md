# 2026-06-26 DIOR-R S3 Stability Long88 Complete

## Scope

Archive the completed DIOR-R S3 long88 continuation launched on `2026-06-25`
from the completed long60 `epoch_60.pth` checkpoints.

Launch provenance:
`New/docs/experiments/20260625_dior_r_s3_stability_long88_launch.md`.

Dataset/protocol: sanitized `DIOR_R_dota/train_val` to `DIOR_R_dota/test`,
MMRotate `DOTAMetric` mAP at IoU 0.5.

S4, pseudo-labeling, FAIR1M, DOTA2 follow-up training, route-changing
experiments, and new GPU launches remain outside this archive.

## Completion Status

All three long88 replicas completed through epoch 88 on `2026-06-26 CST`.
Each workdir contains `epoch_88.pth`, and each `last_checkpoint` points to the
matching `epoch_88.pth`.

The only remaining `screen` session observed during archive verification was
`s0_result_log_monitor_20260603`. The original long88 launch PIDs `1652371`,
`1652558`, and `1652680` were no longer present. GPUs 0-5 were idle; GPU 6 was
occupied by another user/process.

Scoped failure scans across the three launch logs and runtime logs found no
matches for `Traceback`, CUDA OOM, `out-of-memory`, `out of memory`, `libpng`,
`CRC`, `NoneType`, `ValueError`, `KeyboardInterrupt`, `loss: nan`,
`loss: inf`, `grad_norm: nan`, or `grad_norm: inf`.

## Metrics

| Replica | Final epoch 88 `dota/mAP` | Final epoch 88 `dota/AP50` | Best exact mAP | Best epoch |
| --- | ---: | ---: | ---: | ---: |
| rep0 | 0.6874858141 | 0.6870 | 0.6985080242 | 66 |
| rep1 | 0.6918782592 | 0.6920 | 0.6920918226 | 86 |
| rep2 | 0.6982299089 | 0.6980 | 0.6985765696 | 86 |

Aggregate:

- Final mean exact mAP: `0.6925313274`.
- Best mean exact mAP: `0.6963921388`.

## Decision

Treat long88 as useful stability evidence, not an improvement over long60.
The long88 best mean `0.6963921388` is below the long60 best mean `0.696602`,
and the final mean is also below long60 final mean `0.693014`. Pause further
DIOR-R S3 continuation.

Keep S4, pseudo-labeling, FAIR1M, DOTA2 follow-up training, route-changing
experiments, and new long-continuation launches paused until a separate route
decision approves them.
