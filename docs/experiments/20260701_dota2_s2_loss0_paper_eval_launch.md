# 2026-07-01 DOTA2 S2 Loss-0 Paper Evaluation Launch

## Scope

Evaluation-only DOTA2 follow-up for the paper-positive S2 loss-0 best
checkpoints. No DOTA2 S3, S4, pseudo-labeling, FAIR1M, route-change training,
or new training was launched.

Claim boundary: these runs preserve best-checkpoint S2 loss-0 evidence for the
paper-facing artifact bundle. Do not cite final checkpoints as stable S2
evidence from this launch.

## Preflight

Checked at `2026-07-01 09:52 CST`.

Command: `screen -ls`

```text
There is a screen on:
        3470174.s0_result_log_monitor_20260603     (06/03/26 19:55:37)     (Detached)
1 Socket in /run/screen/S-zwl.
```

Command:
`nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`

```text
0, 14 MiB, 0 %
1, 14 MiB, 0 %
2, 20727 MiB, 0 %
3, 14 MiB, 0 %
4, 14 MiB, 0 %
5, 14 MiB, 0 %
6, 22615 MiB, 32 %
```

GPUs 0, 1, and 3 were idle and selected. GPUs 4 and 5 remained available as
remap capacity. GPUs 2 and 6 were occupied.

All requested config and checkpoint paths existed before launch.

## Launches

All jobs were launched from `/data5/2025/ldh/OpenRSD` at about
`2026-07-01 09:53 CST` with:

```bash
tools/bootstrap_run.py tools/test.py <config> <checkpoint> --work-dir <workdir> --out <workdir>/preds.pkl
```

| GPU | Screen | Workdir | Config | Checkpoint | Expected metric | Launch log |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `paper_eval_dota2_s2_loss0_rep3407_e1_20260701_gpu0` | `work_dirs/paper_eval_20260701/dota2_s2_loss0_rep3407_e1` | `work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-ablate-loss0-s1e12-rep3407-20260610_dota2.py` | `work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610/epoch_1.pth` | about `0.621121 / 0.6210` | `work_dirs/paper_eval_20260701/dota2_s2_loss0_rep3407_e1/launch_20260701_gpu0.log` |
| 1 | `paper_eval_dota2_s2_loss0_rep4407_e3_20260701_gpu1` | `work_dirs/paper_eval_20260701/dota2_s2_loss0_rep4407_e3` | `work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep4407_20260610/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-ablate-loss0-s1e12-rep4407-20260610_dota2.py` | `work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep4407_20260610/epoch_3.pth` | about `0.620637 / 0.6210` | `work_dirs/paper_eval_20260701/dota2_s2_loss0_rep4407_e3/launch_20260701_gpu1.log` |
| 3 | `paper_eval_dota2_s2_loss0_rep5407_e1_20260701_gpu3` | `work_dirs/paper_eval_20260701/dota2_s2_loss0_rep5407_e1` | `work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep5407_20260610/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-ablate-loss0-s1e12-rep5407-20260610_dota2.py` | `work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep5407_20260610/epoch_1.pth` | about `0.621215 / 0.6210` | `work_dirs/paper_eval_20260701/dota2_s2_loss0_rep5407_e1/launch_20260701_gpu3.log` |

Per-run environment:

```bash
CUDA_VISIBLE_DEVICES=<gpu>
PYTHONNOUSERSITE=1
MPLCONFIGDIR=/tmp/matplotlib_dota2_s2_loss0_<rep>_20260701
```

## Startup Acceptance

Passed at about `2026-07-01 09:57 CST`.

New detached screens:

```text
3958878.paper_eval_dota2_s2_loss0_rep5407_e1_20260701_gpu3
3958092.paper_eval_dota2_s2_loss0_rep4407_e3_20260701_gpu1
3957052.paper_eval_dota2_s2_loss0_rep3407_e1_20260701_gpu0
```

GPU residency after launch:

```text
0, 4357 MiB, 29 %
1, 4359 MiB, 22 %
2, 20727 MiB, 0 %
3, 4363 MiB, 43 %
4, 14 MiB, 0 %
5, 14 MiB, 0 %
6, 22615 MiB, 30 %
```

Each launch log showed checkpoint loading and `Epoch(test)` progress:

- rep3407 loaded
  `work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610/epoch_1.pth`
  and reached `Epoch(test) [1100/6917]`.
- rep4407 loaded
  `work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep4407_20260610/epoch_3.pth`
  and reached `Epoch(test) [1050/6917]`.
- rep5407 loaded
  `work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep5407_20260610/epoch_1.pth`
  and reached `Epoch(test) [ 850/6917]`.

Scoped startup failure scan was clean for `Traceback`, CUDA OOM,
`out-of-memory`, `out of memory`, `libpng`, `CRC`, `NoneType`, `ValueError`,
`KeyboardInterrupt`, `loss: nan`, `loss: inf`, `grad_norm: nan`, and
`grad_norm: inf`.

## Completion Acceptance

Completed at about `2026-07-01 10:06 CST`.

The three eval screens exited, leaving only the monitor screen active:

```text
There is a screen on:
        3470174.s0_result_log_monitor_20260603     (06/03/26 19:55:37)     (Detached)
1 Socket in /run/screen/S-zwl.
```

GPU state after completion:

```text
0, 14 MiB, 0 %
1, 14 MiB, 0 %
2, 20727 MiB, 0 %
3, 14 MiB, 0 %
4, 14 MiB, 0 %
5, 14 MiB, 0 %
6, 22615 MiB, 30 %
```

| Run | JSON metrics | Artifacts |
| --- | --- | --- |
| DOTA2 S2 loss-0 rep3407 epoch 1 | `dota/mAP=0.6211206912994385`, `dota/AP50=0.621` | `preds.pkl`, copied config, `20260701_095527/20260701_095527.log`, `20260701_095527/20260701_095527.json` |
| DOTA2 S2 loss-0 rep4407 epoch 3 | `dota/mAP=0.6206368803977966`, `dota/AP50=0.621` | `preds.pkl`, copied config, `20260701_095534/20260701_095534.log`, `20260701_095534/20260701_095534.json` |
| DOTA2 S2 loss-0 rep5407 epoch 1 | `dota/mAP=0.621215283870697`, `dota/AP50=0.621` | `preds.pkl`, copied config, `20260701_095547/20260701_095547.log`, `20260701_095547/20260701_095547.json` |

Metrics match the expected S2 best-checkpoint values within evaluator rounding.

Scoped completion failure scan across all launch and runtime logs was clean for
`Traceback`, CUDA OOM, `out-of-memory`, `out of memory`, `libpng`, `CRC`,
`NoneType`, `ValueError`, `KeyboardInterrupt`, `loss: nan`, `loss: inf`,
`grad_norm: nan`, and `grad_norm: inf`.

Preserved claim boundary: S2 loss-0 is best-checkpoint evidence only; this
completion record does not promote final checkpoints as stable S2 evidence.
