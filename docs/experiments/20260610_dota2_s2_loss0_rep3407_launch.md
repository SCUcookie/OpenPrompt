# DOTA2 S2 Loss-0 Rep3407 Launch - 2026-06-10

This note records the controlled DOTA2 S2 hierarchy loss-0 replication launched
after the first loss-0 ablation produced an epoch-2 improvement over S1 but did
not improve the final epoch.

## Purpose

Replicate the DOTA2 S2 loss-0 signal before launching S3/S4 or any DIOR-R
detector relaunch.

Reference metrics on `DOTA2_1024_500/ss_val`:

| Run | Metric |
| --- | --- |
| Main DOTA2 S1 | `dota/mAP=0.6177`, `dota/AP50=0.6180` |
| S2 loss-0 first ablation, best observed | epoch 2 `0.6204 / 0.6200` |
| S2 loss-0 first ablation, final | epoch 4 `0.6179 / 0.6180` |

Treat best checkpoint and final checkpoint separately. If this replication has
an epoch that again exceeds S1, preserve that checkpoint as DOTA2 S2-ablation
evidence even if the final epoch regresses.

## Config

Workdir:

`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610`

Config:

`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-ablate-loss0-s1e12-rep3407-20260610_dota2.py`

The config was copied from the completed loss-0 ablation and changed only for
the new workdir plus fixed MMEngine randomness:

```python
randomness = dict(deterministic=False, seed=3407)
work_dir = 'work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610'
```

Controlled settings retained:

| Setting | Value |
| --- | --- |
| `load_from` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607/epoch_12.pth` |
| `hierarchy_loss_weight` | `0.0` in both cascade heads |
| `learnable_prompt_offsets` | `True` in both cascade heads |
| Optimizer LR | `5e-5` |
| `max_epochs` | `4` |
| `val_interval` | `1` |

## Launch

Preflight GPU checks on 2026-06-10 showed GPU 1 idle across three polls:
`14 MiB`, `0%` utilization. GPU 3 was occupied by another user's process and
was not used.

Screen:

`geonexus_dota2_s2_loss0_rep3407_20260610_gpu1`

Command:

```bash
cd /data5/2025/ldh/OpenRSD
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_dota2_s2_loss0_rep3407 PYTHONNOUSERSITE=1 \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py \
  work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-ablate-loss0-s1e12-rep3407-20260610_dota2.py \
  --work-dir work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610
```

Runtime log:

`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610/20260610_191026/20260610_191026.log`

## Startup Acceptance

Accepted at `2026-06-10 19:15:00 CST`:

```text
Epoch(train) [1][  200/39007]
```

At acceptance, PID `1559651` was running on GPU 1. A scoped scan found no
`Traceback`, CUDA OOM, `libpng`, `CRC`, `NoneType`, `ValueError`,
`KeyboardInterrupt`, or true `nan`/`inf` failure signature in the training log.

## Metric Status

Completed. Validation metrics from:

`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610/20260610_191026/vis_data/scalars.json`

| Epoch | `dota/mAP` | `dota/AP50` | S1 comparison |
| --- | --- | --- | --- |
| 1 | `0.621121` | `0.6210` | above S1 |
| 2 | `0.617161` | `0.6170` | below S1 |
| 3 | `0.612046` | `0.6120` | below S1 |
| 4 | `0.620299` | `0.6200` | above S1 |

Best checkpoint: epoch 1 `0.621121 / 0.6210`.

Final checkpoint: epoch 4 `0.620299 / 0.6200`, the strongest completed final
among the four loss-0 runs summarized on 2026-06-11.

See:

`/data5/2025/ldh/New/docs/experiments/20260611_dota2_s2_loss0_replicates_analysis.md`
