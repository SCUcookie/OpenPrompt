# 2026-06-29 DIOR-R S4 Low-LR Complete

## Scope

Archive the completed DIOR-R S4 low-LR stabilization run launched on
`2026-06-28` from each S4 short-pack epoch-1 checkpoint.

Launch provenance:
`New/docs/experiments/20260628_dior_r_s4_pseudolabel_low_lr_from_e1_launch.md`.

Dataset/protocol: pseudo-label train root
`data/DIOR_R_dota_s4_pseudo_agreement_20260627/`, sanitized
`DIOR_R_dota/test`, MMRotate `DOTAMetric` mAP at IoU 0.5.

## Completion Status

All three accepted bootstrap runs completed through epoch 8 on
`2026-06-28 CST`. Each workdir contains `epoch_8.pth`, and each
`last_checkpoint` points to the matching final checkpoint.

The initial direct `tools/train.py` attempts failed because
`geonexus_mmrotate` was not on `sys.path`; those preserved
`launch_20260628_gpu*.log` files contain the expected import `Traceback`.
Accepted training used `tools/bootstrap_run.py tools/train.py`, with launch
logs `launch_20260628_gpu*_bootstrap.log` and runtime logs under
`20260628_101540/`.

Archive verification on `2026-06-29 09:06 CST` found only
`s0_result_log_monitor_20260603` remaining in `screen`. GPU 6 was occupied by
another user/process, and GPUs 0, 2, and 3 were idle for paper-eval follow-up.

Scoped failure scans across the accepted bootstrap launch logs and runtime
logs found no matches for `Traceback`, CUDA OOM, `out-of-memory`,
`out of memory`, `libpng`, `CRC`, `NoneType`, `ValueError`,
`KeyboardInterrupt`, `loss: nan`, `loss: inf`, `grad_norm: nan`, or
`grad_norm: inf`.

## Metrics

| Replica | Best epoch | Best `dota/mAP` | Best `dota/AP50` | Final epoch 8 `dota/mAP` | Final epoch 8 `dota/AP50` |
| --- | ---: | ---: | ---: | ---: | ---: |
| rep23407 | 2 | 0.6935 | 0.6930 | 0.6892 | 0.6890 |
| rep24407 | 6 | 0.6966 | 0.6970 | 0.6963 | 0.6960 |
| rep25407 | 2 | 0.6967 | 0.6970 | 0.6923 | 0.6920 |

Aggregate:

- Best mean mAP: `0.6956`.
- Final mean mAP: `0.6926`.

## Decision

Classify this as weak stabilization only, not paper-facing S4 superiority.

The final mean improves over the S4 short-pack final mean `0.691337` by about
`+0.0013`, but remains below the S3 long60 final mean `0.693014`. The best
mean remains below the S4 short-pack best mean `0.696903` and below the
original S3 best threshold `0.6979`.

Pause further S4 training unless separately approved. Use `2026-06-29` for
paper-facing evaluation artifacts on the best low-LR checkpoint from each
replica.
