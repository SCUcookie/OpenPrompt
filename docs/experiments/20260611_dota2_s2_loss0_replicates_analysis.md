# DOTA2 S2 Loss-0 Replicate Analysis - 2026-06-11

Machine: `nuosen`

Repos:

| Repo | Commit |
| --- | --- |
| `New` | `5cab791` |
| `OpenRSD` | `12d3fd8` |

Evaluation protocol: `DOTA2_1024_500/ss_val`, MMRotate `DOTAMetric` mAP at IoU 0.5. Baseline comparator is DOTA2 S1 `dota/mAP=0.6177`, `dota/AP50=0.6180`.

## Completed Loss-0 Runs

All runs load S1 epoch 12:

`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607/epoch_12.pth`

Controlled S2 settings: `hierarchy_loss_weight=0.0`, `learnable_prompt_offsets=True`, optimizer LR `5e-5`, validation interval `1`, checkpoint interval `1`.

| Run | Config | Runtime log | Scalars | Checkpoints |
| --- | --- | --- | --- | --- |
| first loss-0 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_20260610/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-ablate-loss0-s1e12-20260610_dota2.py` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_20260610/20260610_100253/20260610_100253.log` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_20260610/20260610_100253/vis_data/scalars.json` | `epoch_1.pth` to `epoch_4.pth`, `last_checkpoint` |
| rep3407 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-ablate-loss0-s1e12-rep3407-20260610_dota2.py` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610/20260610_191026/20260610_191026.log` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610/20260610_191026/vis_data/scalars.json` | `epoch_1.pth` to `epoch_4.pth`, `last_checkpoint` |
| rep4407 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep4407_20260610/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-ablate-loss0-s1e12-rep4407-20260610_dota2.py` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep4407_20260610/20260610_210021/20260610_210021.log` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep4407_20260610/20260610_210021/vis_data/scalars.json` | `epoch_1.pth` to `epoch_4.pth`, `last_checkpoint` |
| rep5407 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep5407_20260610/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-ablate-loss0-s1e12-rep5407-20260610_dota2.py` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep5407_20260610/20260610_210021/20260610_210021.log` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep5407_20260610/20260610_210021/vis_data/scalars.json` | `epoch_1.pth` to `epoch_4.pth`, `last_checkpoint` |

## Metrics

| Run | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Best | Final | Read |
| --- | --- | --- | --- | --- | --- | --- | --- |
| first loss-0 | `0.617559 / 0.6180` | `0.620374 / 0.6200` | `0.618740 / 0.6190` | `0.617885 / 0.6180` | epoch 2 `0.620374 / 0.6200` | epoch 4 `0.617885 / 0.6180` | best and final both above S1 |
| rep3407 | `0.621121 / 0.6210` | `0.617161 / 0.6170` | `0.612046 / 0.6120` | `0.620299 / 0.6200` | epoch 1 `0.621121 / 0.6210` | epoch 4 `0.620299 / 0.6200` | strongest completed final |
| rep4407 | `0.615142 / 0.6150` | `0.615141 / 0.6150` | `0.620637 / 0.6210` | `0.614786 / 0.6150` | epoch 3 `0.620637 / 0.6210` | epoch 4 `0.614786 / 0.6150` | useful best, unstable final |
| rep5407 | `0.621215 / 0.6210` | `0.620206 / 0.6200` | `0.616239 / 0.6160` | `0.614990 / 0.6150` | epoch 1 `0.621215 / 0.6210` | epoch 4 `0.614990 / 0.6150` | useful best, unstable final |

Combined analysis:

- All four best checkpoints beat S1.
- Best-checkpoint mean: `0.620837`, `+0.003137` over S1.
- Replication-only best mean: `0.620991`, `+0.003291` over S1.
- Final-checkpoint mean: `0.616990`, `-0.000710` below S1.

Conclusion: loss-0 short fine-tuning has repeatable early-epoch value, but epoch-4 final selection is unstable.

## Evidence Status

Paper-facing candidate: best checkpoints only, pending protocol decision.

Diagnostic finding: final epoch selection is unstable and should not be cited as S2 evidence unless follow-up stabilization removes the regression.

## 2026-06-12 Stabilization Completion

The three controlled 3-epoch stabilization runs launched on 2026-06-11 completed cleanly.

| Run | Epoch 1 | Epoch 2 | Epoch 3 final | Best | Final | Read |
| --- | --- | --- | --- | --- | --- | --- |
| rep6407 | `0.620483 / 0.6200` | `0.618960 / 0.6190` | `0.618167 / 0.6180` | epoch 1 `0.620483 / 0.6200` | epoch 3 `0.618167 / 0.6180` | early positive, final only marginally above S1 |
| rep7407 | `0.620785 / 0.6210` | `0.614483 / 0.6140` | `0.618315 / 0.6180` | epoch 1 `0.620785 / 0.6210` | epoch 3 `0.618315 / 0.6180` | early positive, final only marginally above S1 |
| rep8407 | `0.616526 / 0.6170` | `0.619625 / 0.6200` | `0.612147 / 0.6120` | epoch 2 `0.619625 / 0.6200` | epoch 3 `0.612147 / 0.6120` | useful best, unstable final |

Aggregate against S1 comparator `0.6177 / 0.6180`:

- Original 4 loss-0 runs: best mean `0.620837`, `+0.003137` over S1; final mean `0.616990`, `-0.000710` below S1.
- New 3 stabilization runs: best mean `0.620298`, `+0.002598` over S1; final mean `0.616209`, `-0.001491` below S1.
- All 7 runs: best mean `0.620606`, `+0.002906` over S1; final mean `0.616655`, `-0.001045` below S1.

Conclusion: loss-0 S2 is a repeatable early-checkpoint candidate, but remains an unstable final-checkpoint S2. Do not cite the final epoch as S2 evidence.

## 2026-06-12 DIOR-R Sanitized-Label Gate

Sanitized DIOR-R label directories were prepared under:

- `/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/train_val/labelTxt_sanitized_invalidsize_20260612`
- `/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/test/labelTxt_sanitized_invalidsize_20260612`
- Scanner-compatible root: `/data5/2025/ldh/OpenRSD/data/DIOR_R_dota_sanitized_invalidsize_20260612`

The prior 2026-06-09 full-scan artifact recorded two zero-area/invalid-size records per split, but the current raw label files no longer contain those invalid records under the same finite-coordinate, positive-area, positive-edge, and known-class checks. The sanitized directories therefore mirror the current raw labels. Raw `labelTxt` directories were not modified.

Fresh sanitized scan artifacts:

- JSON: `/data5/2025/ldh/New/artifacts/dior_r_diagnostics_20260612_sanitized_invalidsize_geometry.json`
- Markdown: `/data5/2025/ldh/New/artifacts/dior_r_diagnostics_20260612_sanitized_invalidsize_geometry.md`

Sanitized scan result: `11725` train_val label files / `68070` objects and `11738` test label files / `124443` objects; both splits have `num_bad_label_files=0` and `invalid_rbox_size=0`. Remaining out-of-bounds counts are unchanged diagnostic context, not removed by this invalid-size filter.

## 2026-06-12 DIOR-R Train-Step Diagnostics

`OpenRSD/tools/diagnose_first_nonfinite_loss.py` now supports `--mode train-step`, which runs the real `model.train_step(data_batch, optim_wrapper)` path, prints progress records for startup acceptance, and writes the first non-finite loss or exception with batch image/label/bbox context.

Runtime configs were copied under `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_trainstep_diag_20260612/` and point to `data/DIOR_R_dota_sanitized_invalidsize_20260612/`.

| Model | GPU | Limit | Config | JSON | Result |
| --- | --- | --- | --- | --- | --- |
| ORCNN R50 | 0 | `1000` train-step batches | `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_trainstep_diag_20260612/orcnn/dior_r_orcnn_sanitized_trainstep_diag_20260612.py` | `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_trainstep_diag_20260612/orcnn/trainstep_diag_20260612.json` | `finite_within_limit`, `checked_batches=1000` |
| RoI Transformer | 1 | `4000` train-step batches | `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_trainstep_diag_20260612/roi_trans/dior_r_roi_trans_sanitized_trainstep_diag_20260612.py` | `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_trainstep_diag_20260612/roi_trans/trainstep_diag_20260612.json` | `finite_within_limit`, `checked_batches=4000` |
| Rotated RetinaNet | 2 | `1500` train-step batches | `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_trainstep_diag_20260612/retinanet/dior_r_retinanet_sanitized_trainstep_diag_20260612.py` | `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_trainstep_diag_20260612/retinanet/trainstep_diag_20260612.json` | `finite_within_limit`, `checked_batches=1500` |

Failure-signature scans after completion found no `Traceback`, CUDA OOM, `out of memory`, `libpng`, `CRC`, `NoneType`, `ValueError`, true `nan`, or true `inf` in the three diagnostic logs.

Because all three diagnostics finished finite within limits, one diagnostic-only sanitized DIOR-R RoI Transformer S0 smoke was launched on GPU 3:

- Screen: `dior_r_roi_trans_s0_sanitized_smoke_20260612_gpu3`
- PID: `3363864`
- Workdir: `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_roi_trans_sanitized_smoke_20260612`
- Config: `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_roi_trans_sanitized_smoke_20260612/dior_r_roi_trans_sanitized_s0_smoke_1e_20260612.py`
- Launch log: `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_roi_trans_sanitized_smoke_20260612/launch_20260612_gpu3.log`
- Runtime log: `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_roi_trans_sanitized_smoke_20260612/20260612_153506/20260612_153506.log`

Startup acceptance passed at `Epoch(train) [1][  50/5862]` with clean loss values and no immediate failure signatures. At the documentation update, the run was still active on GPU 3.

## 2026-06-11 Stabilization Plan

Launch three controlled 3-epoch DOTA2 stabilization jobs on GPUs 0, 1, and 2, copied from `rep3407` and changed only for seed, workdir/name, `max_epochs=3`, and matching scheduler horizon.

| Run | GPU | Seed | Workdir | Config |
| --- | --- | --- | --- | --- |
| rep6407 | 0 | `6407` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep6407_20260611` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep6407_20260611/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-ablate-loss0-s1e12-rep6407-20260611_dota2.py` |
| rep7407 | 1 | `7407` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep7407_20260611` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep7407_20260611/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-ablate-loss0-s1e12-rep7407-20260611_dota2.py` |
| rep8407 | 2 | `8407` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep8407_20260611` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep8407_20260611/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-ablate-loss0-s1e12-rep8407-20260611_dota2.py` |

Acceptance rule: accept each launch only after `Epoch(train) [1][  200/39007]` and a clean scan for `Traceback`, OOM, `libpng`, `CRC`, `NoneType`, `ValueError`, `KeyboardInterrupt`, true `nan`, and true `inf`.

Decision rule after completion: if best-checkpoint mean stays above S1 by at least `+0.002 mAP`, keep loss-0 early-checkpoint selection as the DOTA2 S2 candidate. If finals remain unstable, do not cite final epoch as S2 evidence. Do not open S3/S4 today unless these runs finish cleanly and the best-checkpoint signal remains stable.

## 2026-06-11 Launch Status

Preflight at `2026-06-11 10:26:39 CST` showed GPUs 0, 1, and 2 idle at `14 MiB` and `0%` utilization. Jobs were launched in detached `screen` sessions at `2026-06-11 10:27:23 CST`.

| Run | GPU | PID | Screen | Runtime log | Startup acceptance |
| --- | --- | --- | --- | --- | --- |
| rep6407 | 0 | `1870974` | `geonexus_dota2_s2_loss0_rep6407_20260611_gpu0` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep6407_20260611/20260611_102732/20260611_102732.log` | accepted at `2026-06-11 10:32:53 CST`, `Epoch(train) [1][  200/39007]` |
| rep7407 | 1 | `1870996` | `geonexus_dota2_s2_loss0_rep7407_20260611_gpu1` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep7407_20260611/20260611_102732/20260611_102732.log` | accepted at `2026-06-11 10:32:52 CST`, `Epoch(train) [1][  200/39007]` |
| rep8407 | 2 | `1870993` | `geonexus_dota2_s2_loss0_rep8407_20260611_gpu2` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep8407_20260611/20260611_102732/20260611_102732.log` | accepted at `2026-06-11 10:32:53 CST`, `Epoch(train) [1][  200/39007]` |

Failure-signature scans through startup acceptance found no `Traceback`, OOM, `libpng`, `CRC`, `NoneType`, `ValueError`, `KeyboardInterrupt`, true `nan`, or true `inf`.

Current state at the last check: all three Python processes were still running and GPU-active. Metrics are pending completion of `epoch_1.pth`, `epoch_2.pth`, `epoch_3.pth`, `last_checkpoint`, and `vis_data/scalars.json`.

## 2026-06-11 Live Monitoring

Poll at `2026-06-11 14:56:52 CST`: all three stabilization training jobs were still running and GPU-active. No replacement launch was needed. `nvidia-smi --query-compute-apps` showed the three training PIDs plus one unrelated `VLLM::EngineCore` process; only the training PIDs count toward the three-process requirement.

| Run | GPU | PID | Process state | Elapsed | GPU memory | Latest metrics |
| --- | --- | --- | --- | --- | --- | --- |
| rep6407 | 0 | `1870974` | `Rsl+` | `04:29:05` | `15458 MiB` | epoch 1 `0.620483 / 0.6200`, epoch 2 `0.618960 / 0.6190` |
| rep7407 | 1 | `1870996` | `Rsl+` | `04:29:04` | `16936 MiB` | epoch 1 `0.620785 / 0.6210`, epoch 2 `0.614483 / 0.6140` |
| rep8407 | 2 | `1870993` | `Rsl+` | `04:29:05` | `12586 MiB` | epoch 1 `0.616526 / 0.6170`, epoch 2 `0.619625 / 0.6200` |

Failure-signature scan through this poll found no `Traceback`, OOM, `out of memory`, `libpng`, `CRC`, `NoneType`, `ValueError`, `KeyboardInterrupt`, true `nan`, or true `inf`. No `epoch_3.pth` or `last_checkpoint` completion marker was present yet in the scanned outputs.

Poll at `2026-06-11 15:09:13 CST`: all three stabilization training jobs remained running and GPU-active. No replacement launch was needed.

| Run | GPU | PID | Process state | Elapsed | GPU memory | Latest metrics |
| --- | --- | --- | --- | --- | --- | --- |
| rep6407 | 0 | `1870974` | `Rsl+` | `04:41:51` | `15458 MiB` | epoch 1 `0.620483 / 0.6200`, epoch 2 `0.618960 / 0.6190` |
| rep7407 | 1 | `1870996` | `Rsl+` | `04:41:50` | `16936 MiB` | epoch 1 `0.620785 / 0.6210`, epoch 2 `0.614483 / 0.6140` |
| rep8407 | 2 | `1870993` | `Ssl+` | `04:41:51` | `12586 MiB` | epoch 1 `0.616526 / 0.6170`, epoch 2 `0.619625 / 0.6200` |

Failure-signature scan through this poll remained clean. No `epoch_3.pth`, `last_checkpoint`, or epoch 3 metric was present yet in the scanned outputs.

Poll at `2026-06-11 15:37:45 CST`: all three stabilization training jobs remained running and GPU-active. No replacement launch was needed. Direct log tails showed active epoch 3 training:

- rep6407: latest observed `Epoch(train) [3][23350/39007]`, ETA about `0:44:09`.
- rep7407: latest observed `Epoch(train) [3][24150/39007]`, ETA about `0:41:36`.
- rep8407: latest observed `Epoch(train) [3][23500/39007]`, ETA about `0:43:41`.

Failure-signature scan through this poll remained clean. No `epoch_3.pth`, `last_checkpoint`, or epoch 3 validation metric was present yet in the scanned outputs.

Poll at `2026-06-11 16:16:29 CST`: all three stabilization training jobs remained running and GPU-active. No replacement launch was needed. `nvidia-smi --query-compute-apps` also showed an unrelated extra `python` PID `2205521` and `VLLM::EngineCore`; these were not counted toward the three tracked training jobs. Direct log tails showed late epoch 3 training:

- rep6407: latest observed `Epoch(train) [3][36500/39007]`, ETA about `0:07:06`.
- rep7407: latest observed `Epoch(train) [3][37000/39007]`, ETA about `0:05:39`.
- rep8407: latest observed `Epoch(train) [3][36550/39007]`, ETA about `0:06:57`.

Failure-signature scan through this poll remained clean. No `epoch_3.pth`, `last_checkpoint`, or epoch 3 validation metric was present yet in the scanned outputs.

Poll at `2026-06-11 16:02:15 CST`: all three stabilization training jobs remained running and GPU-active. No replacement launch was needed. Direct log tails showed active epoch 3 training:

- rep6407: latest observed `Epoch(train) [3][31650/39007]`, ETA about `0:20:48`.
- rep7407: latest observed `Epoch(train) [3][32200/39007]`, ETA about `0:19:09`.
- rep8407: latest observed `Epoch(train) [3][31700/39007]`, ETA about `0:20:39`.

Failure-signature scan through this poll remained clean. No `epoch_3.pth`, `last_checkpoint`, or epoch 3 validation metric was present yet in the scanned outputs.

Poll at `2026-06-11 15:49:54 CST`: all three stabilization training jobs remained running and GPU-active. No replacement launch was needed. Direct log tails showed active epoch 3 training:

- rep6407: latest observed `Epoch(train) [3][27400/39007]`, ETA about `0:32:48`.
- rep7407: latest observed `Epoch(train) [3][28100/39007]`, ETA about `0:30:38`.
- rep8407: latest observed `Epoch(train) [3][27500/39007]`, ETA about `0:32:29`.

Failure-signature scan through this poll remained clean. No `epoch_3.pth`, `last_checkpoint`, or epoch 3 validation metric was present yet in the scanned outputs.

Poll at `2026-06-11 15:20:44 CST`: all three stabilization training jobs remained running and GPU-active. No replacement launch was needed.

| Run | GPU | PID | Process state | Elapsed | GPU memory | Latest metrics |
| --- | --- | --- | --- | --- | --- | --- |
| rep6407 | 0 | `1870974` | `Rsl+` | `04:53:21` | `15458 MiB` | epoch 1 `0.620483 / 0.6200`, epoch 2 `0.618960 / 0.6190` |
| rep7407 | 1 | `1870996` | `Rsl+` | `04:53:20` | `16936 MiB` | epoch 1 `0.620785 / 0.6210`, epoch 2 `0.614483 / 0.6140` |
| rep8407 | 2 | `1870993` | `Rsl+` | `04:53:21` | `12586 MiB` | epoch 1 `0.616526 / 0.6170`, epoch 2 `0.619625 / 0.6200` |

Failure-signature scan through this poll remained clean. No `epoch_3.pth`, `last_checkpoint`, or epoch 3 metric was present yet in the scanned outputs.
