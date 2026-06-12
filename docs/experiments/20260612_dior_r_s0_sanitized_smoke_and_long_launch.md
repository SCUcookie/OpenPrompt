# DIOR-R S0 Sanitized Smoke And Long-Run Launch - 2026-06-12

This note records the sanitized DIOR-R S0 smoke evidence and the follow-on
long diagnostic S0 launch. These runs use the sanitized DIOR-R root:

`/data5/2025/ldh/OpenRSD/data/DIOR_R_dota_sanitized_invalidsize_20260612/`

The long-run jobs are diagnostic DIOR-R S0 evidence, not final benchmark
claims.

## Completed Sanitized Smokes

| Model | GPU | Work dir | Finished | Metric |
| --- | --- | --- | --- | --- |
| ORCNN R50 | 0 | `dior_r_s0_orcnn_sanitized_smoke_20260612_gpu0` | `2026-06-12 17:21:56 CST` | `dota/mAP=0.3823`, `dota/AP50=0.3820` |
| RoI Transformer R50 repeat | 1 | `dior_r_s0_roi_trans_sanitized_smoke_rep_20260612_gpu1` | `2026-06-12 17:08:24 CST` | `dota/mAP=0.3676`, `dota/AP50=0.3680` |
| Rotated RetinaNet R50 | 2 | `dior_r_s0_retinanet_sanitized_smoke_20260612_gpu2` | `2026-06-12 16:55:51 CST` | `dota/mAP=0.0946`, `dota/AP50=0.0950` |
| RoI Transformer R50 earlier smoke | 3 | `dior_r_s0_roi_trans_sanitized_smoke_20260612` | previously recorded | `dota/mAP=0.4011`, `dota/AP50=0.4010` |

Train-step diagnostics already completed finite:

- ORCNN: `1000/1000` checked batches.
- RoI Transformer: `4000/4000` checked batches.
- Rotated RetinaNet: `1500/1500` checked batches.

## Long-Run Launch Plan

| GPU | Model | Work dir | Config | Schedule | Estimate |
| --- | --- | --- | --- | --- | --- |
| 0 | ORCNN R50 | `dior_r_s0_orcnn_sanitized_long_20260612_gpu0` | `work_dirs/dior_r_s0_orcnn_sanitized_long_20260612_gpu0/dior_r_s0_orcnn_sanitized_long_20260612_gpu0.py` | `max_epochs=12`, `val_interval=4` | about `5h30m` |
| 1 | RoI Transformer R50 | `dior_r_s0_roi_trans_sanitized_long_20260612_gpu1` | `work_dirs/dior_r_s0_roi_trans_sanitized_long_20260612_gpu1/dior_r_s0_roi_trans_sanitized_long_20260612_gpu1.py` | `max_epochs=16`, `val_interval=4` | about `5h15m` |
| 2 | Rotated RetinaNet R50 | `dior_r_s0_retinanet_sanitized_long_20260612_gpu2` | `work_dirs/dior_r_s0_retinanet_sanitized_long_20260612_gpu2/dior_r_s0_retinanet_sanitized_long_20260612_gpu2.py` | `max_epochs=32`, `val_interval=4` | about `5h35m` |

GPU 4 remains untouched because it is occupied by another user.

If launched around `18:00 CST` on `2026-06-12`, expected finish times are:

- RoI Transformer: about `23:15 CST`, `2026-06-12`.
- ORCNN: about `23:30 CST`, `2026-06-12`.
- RetinaNet: about `23:35 CST`, `2026-06-12`.

## Monitoring Protocol

Accept startup only after each job logs at least the first `50` training
batches. Scan logs for `Traceback`, CUDA OOM, `libpng`, `CRC`, `NoneType`,
`ValueError`, true `nan`, and true `inf`. If any job fails or catches
non-finite behavior, stop that model family and record the log path without
relaunching unchanged.

Final metrics, checkpoint paths, logs, and any failures should be appended
after all three jobs finish.

## Launch Record

Launched at `2026-06-12 18:12 CST` in detached screens:

| GPU | Screen | Python PID at startup | Startup acceptance |
| --- | --- | --- | --- |
| 0 | `dior_r_s0_orcnn_sanitized_long_20260612_gpu0` | `208913` | reached `Epoch(train) [1][50/5862]` with finite losses at `18:12:20 CST` |
| 1 | `dior_r_s0_roi_trans_sanitized_long_20260612_gpu1` | `209667` | reached `Epoch(train) [1][50/5862]` with finite losses at `18:12:24 CST` |
| 2 | `dior_r_s0_retinanet_sanitized_long_20260612_gpu2` | `210781` | reached `Epoch(train) [1][50/5862]` with finite losses at `18:12:32 CST` |

Startup `nvidia-smi` at `18:13:07 CST` confirmed active Python training
processes on GPUs 0, 1, and 2. GPU 4 remained occupied by the pre-existing
separate Python process and was not touched.

Initial failure scan over the three launch logs found no `Traceback`, CUDA
OOM, `libpng`, `CRC`, `NoneType`, `ValueError`, `KeyboardInterrupt`, or true
`nan`/`inf` signature.

## Interim Metrics

| Time | Run | Status |
| --- | --- | --- |
| `2026-06-12 18:54:11 CST` | Rotated RetinaNet long GPU2 | epoch 4 validation complete: `dota/mAP=0.4271`, `dota/AP50=0.4270`; training resumed into epoch 5 |

## Overnight Extension

At user request on `2026-06-12 19:33 CST`, the configs were extended so the
runs should continue past `2026-06-13 08:00 CST` after the original processes
finish:

| Run | Original cap | Extended cap | Continuation screen | Continuation log |
| --- | --- | --- | --- | --- |
| ORCNN GPU0 | 12 epochs | 36 epochs | `dior_r_s0_orcnn_extend_after_original_20260612` | `work_dirs/dior_r_s0_orcnn_sanitized_long_20260612_gpu0/continuation_watcher_20260612.log` |
| RoI Transformer GPU1 | 16 epochs | 52 epochs | `dior_r_s0_roi_trans_extend_after_original_20260612` | `work_dirs/dior_r_s0_roi_trans_sanitized_long_20260612_gpu1/continuation_watcher_20260612.log` |
| Rotated RetinaNet GPU2 | 32 epochs | 96 epochs | `dior_r_s0_retinanet_extend_after_original_20260612` | `work_dirs/dior_r_s0_retinanet_sanitized_long_20260612_gpu2/continuation_watcher_20260612.log` |

The running Python jobs do not re-read config files, so the extension is
implemented as a non-disruptive continuation: each watcher waits for the
original PID to exit normally, then relaunches the same work dir with
`--resume auto` from the latest checkpoint and the higher `max_epochs`.

Verification at `2026-06-12 19:33 CST`: only the original Python training PIDs
were using GPUs 0/1/2 (`208913`, `209667`, `210781`); the continuation screens
were waiting and had not started duplicate training jobs. GPU 4 remained
occupied by the pre-existing separate process.
