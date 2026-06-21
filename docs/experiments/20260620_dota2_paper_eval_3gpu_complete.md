# 2026-06-20 DOTA2 Paper Evaluation 3-GPU Complete

## Scope

Analysis-only paper evaluation on existing DOTA2 checkpoints. No S4,
pseudo-labeling, FAIR1M, routing change, or new training was launched.

## Preflight

Command: `screen -ls`

```text
There is a screen on:
        3470174.s0_result_log_monitor_20260603     (06/03/26 19:55:37)     (Detached)
1 Socket in /run/screen/S-zwl.
```

Command:
`nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`

```text
0, 2351 MiB, 0 %
1, 14 MiB, 0 %
2, 14 MiB, 0 %
3, 14 MiB, 0 %
4, 14 MiB, 0 %
5, 14 MiB, 0 %
6, 14 MiB, 0 %
```

GPUs 1, 2, and 3 were idle; no remap was needed.

All requested config and checkpoint paths existed before launch.

## Launches

All jobs were launched from `/data5/2025/ldh/OpenRSD` at about
`2026-06-20 10:15 CST`.

| GPU | Screen | Workdir | Config | Checkpoint | Launch log |
| --- | --- | --- | --- | --- | --- |
| 1 | `paper_eval_dota2_s0_roi_trans_epoch12_20260620_gpu1` | `work_dirs/paper_eval_20260620/dota2_s0_roi_trans_epoch12` | `work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/G02_Baselines_Data1_DOTA2_M2_RoITrans_validpng_20260602.py` | `work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/epoch_12.pth` | `work_dirs/paper_eval_20260620/dota2_s0_roi_trans_epoch12/launch_20260620_gpu1.log` |
| 2 | `paper_eval_dota2_s1_epoch12_20260620_gpu2` | `work_dirs/paper_eval_20260620/dota2_s1_epoch12` | `work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607/roi-trans-le90_r50_fpn_remoteclip-s1-validpng-20260607_dota2.py` | `work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607/epoch_12.pth` | `work_dirs/paper_eval_20260620/dota2_s1_epoch12/launch_20260620_gpu2.log` |
| 3 | `paper_eval_dota2_s2_loss0_rep3407_epoch1_20260620_gpu3` | `work_dirs/paper_eval_20260620/dota2_s2_loss0_rep3407_epoch1` | `work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-ablate-loss0-s1e12-rep3407-20260610_dota2.py` | `work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610/epoch_1.pth` | `work_dirs/paper_eval_20260620/dota2_s2_loss0_rep3407_epoch1/launch_20260620_gpu3.log` |

## Startup Acceptance

Passed. New detached screens:

```text
872808.paper_eval_dota2_s2_loss0_rep3407_epoch1_20260620_gpu3
872798.paper_eval_dota2_s1_epoch12_20260620_gpu2
872692.paper_eval_dota2_s0_roi_trans_epoch12_20260620_gpu1
```

GPU residency after launch:

```text
0, 2351 MiB, 0 %
1, 4573 MiB, 38 %
2, 4509 MiB, 29 %
3, 4357 MiB, 38 %
4, 14 MiB, 0 %
5, 14 MiB, 0 %
6, 14 MiB, 0 %
```

Each launch log showed checkpoint loading and `Epoch(test)` progress:

- S0 loaded `work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/epoch_12.pth`
  and reached `Epoch(test) [ 750/6917]`.
- S1 loaded `work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607/epoch_12.pth`
  and reached `Epoch(test) [ 750/6917]`.
- S2 loaded `work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610/epoch_1.pth`
  and reached `Epoch(test) [ 750/6917]`.

Scoped startup failure scan was clean for `Traceback`, CUDA OOM,
`out-of-memory`, `out of memory`, `libpng`, `CRC`, `NoneType`, `ValueError`,
`KeyboardInterrupt`, `loss: nan`, `loss: inf`, `grad_norm: nan`, and
`grad_norm: inf`.

## Completion Acceptance

Completed at about `2026-06-20 10:31 CST`. The three eval screens exited,
leaving only `s0_result_log_monitor_20260603` active.

| Run | JSON metrics | Artifacts |
| --- | --- | --- |
| DOTA2 S0 RoI Transformer epoch 12 | `dota/mAP=0.6088330745697021`, `dota/AP50=0.609` | `preds.pkl`, `20260620_101512/20260620_101512.log`, `20260620_101512/20260620_101512.json` |
| DOTA2 S1 epoch 12 | `dota/mAP=0.6176869869232178`, `dota/AP50=0.618` | `preds.pkl`, `20260620_101519/20260620_101519.log`, `20260620_101519/20260620_101519.json` |
| DOTA2 S2 loss-0 rep3407 epoch 1 | `dota/mAP=0.6211206912994385`, `dota/AP50=0.621` | `preds.pkl`, `20260620_101519/20260620_101519.log`, `20260620_101519/20260620_101519.json` |

Final screen state:

```text
There is a screen on:
        3470174.s0_result_log_monitor_20260603     (06/03/26 19:55:38)     (Detached)
1 Socket in /run/screen/S-zwl.
```

Scoped completion failure scan across all launch and runtime logs was clean for
`Traceback`, CUDA OOM, `out-of-memory`, `out of memory`, `libpng`, `CRC`,
`NoneType`, `ValueError`, `KeyboardInterrupt`, `loss: nan`, `loss: inf`,
`grad_norm: nan`, and `grad_norm: inf`.
