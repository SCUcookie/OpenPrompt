# 2026-06-22 DOTA2 S3 Continue13 Launch

## Scope

Launch a conservative 3-GPU continuation of the completed DOTA2 S3 confirm6
replicas `6407/7407/8407`, resuming from epoch 6 and extending to epoch 13.
Epoch 12 is the primary comparable result; epoch 13 is a stability/occupancy
tail. This does not introduce S4, pseudo-labeling, FAIR1M, or a route claim.

Related completion archive:
`docs/experiments/20260622_dota2_s3_confirm6_complete.md`.

## Preflight

Checks before launch:

- `screen -ls`: only `s0_result_log_monitor_20260603` remained.
- GPU 4 was occupied by another process.
- GPUs 1, 2, and 3 were idle, so no remap was needed.
- Launch timestamp: `2026-06-22 09:52 CST`.

## Configs

Each continuation workdir was created cleanly under
`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/`. The matching confirm6
config was copied into the new workdir and changed only for:

- `max_epochs = 13`
- `train_cfg = dict(max_epochs=13, type='EpochBasedTrainLoop', val_interval=1)`
- `work_dir = '<new continuation workdir>'`

Preserved settings include seeds, LR `5e-5`, batch size `2`, prompt embedding
artifact, scene-adapter settings, dataset paths, checkpoint interval `1`, and
validation interval `1`.

| Replica | GPU | PID | Screen | Config | Resume checkpoint | Launch log |
| --- | ---: | ---: | --- | --- | --- | --- |
| rep6407 | 1 | `3329109` | `geonexus_dota2_s3_continue13_rep6407_20260622_gpu1` | `work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep6407_continue13_20260622/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-loss0-best-rep6407-continue13-20260622_dota2.py` | `work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep6407_confirm6_20260621/epoch_6.pth` | `work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep6407_continue13_20260622/launch_20260622_gpu1.log` |
| rep7407 | 2 | `3330032` | `geonexus_dota2_s3_continue13_rep7407_20260622_gpu2` | `work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep7407_continue13_20260622/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-loss0-best-rep7407-continue13-20260622_dota2.py` | `work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep7407_confirm6_20260621/epoch_6.pth` | `work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep7407_continue13_20260622/launch_20260622_gpu2.log` |
| rep8407 | 3 | `3330569` | `geonexus_dota2_s3_continue13_rep8407_20260622_gpu3` | `work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep8407_continue13_20260622/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-loss0-best-rep8407-continue13-20260622_dota2.py` | `work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep8407_confirm6_20260621/epoch_6.pth` | `work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep8407_continue13_20260622/launch_20260622_gpu3.log` |

MMEngine startup rewrote each runtime config to the intended epoch-6 checkpoint
with `resume = True`.

## Commands

Launched from `/data5/2025/ldh/OpenRSD`:

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_dota2_s3_continue13_rep6407_20260622 PYTHONNOUSERSITE=1 \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py \
  work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep6407_continue13_20260622/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-loss0-best-rep6407-continue13-20260622_dota2.py \
  --work-dir work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep6407_continue13_20260622 \
  --resume /data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep6407_confirm6_20260621/epoch_6.pth
```

```bash
CUDA_VISIBLE_DEVICES=2 MPLCONFIGDIR=/tmp/matplotlib_dota2_s3_continue13_rep7407_20260622 PYTHONNOUSERSITE=1 \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py \
  work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep7407_continue13_20260622/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-loss0-best-rep7407-continue13-20260622_dota2.py \
  --work-dir work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep7407_continue13_20260622 \
  --resume /data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep7407_confirm6_20260621/epoch_6.pth
```

```bash
CUDA_VISIBLE_DEVICES=3 MPLCONFIGDIR=/tmp/matplotlib_dota2_s3_continue13_rep8407_20260622 PYTHONNOUSERSITE=1 \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py \
  work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep8407_continue13_20260622/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-loss0-best-rep8407-continue13-20260622_dota2.py \
  --work-dir work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep8407_continue13_20260622 \
  --resume /data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep8407_confirm6_20260621/epoch_6.pth
```

## Startup Acceptance

At `2026-06-22 09:59 CST`:

- Screens were detached and present.
- PIDs `3329109`, `3330032`, and `3330569` were GPU-resident on GPUs 1, 2,
  and 3.
- GPU memory was active on GPUs 1/2/3: rep6407 about `20860 MiB`, rep7407
  about `12596 MiB`, and rep8407 about `12596 MiB`.
- The scoped failure scan was clean.
- Acceptance marker reached cleanly:
  - rep6407: `06/22 09:58:15 - Epoch(train)  [7][  200/39007]`
  - rep7407: `06/22 09:58:15 - Epoch(train)  [7][  200/39007]`
  - rep8407: `06/22 09:58:14 - Epoch(train)  [7][  200/39007]`

## ETA

The 6-epoch confirm6 pack took about 12 hours 30-40 minutes. This 7-epoch
continuation is expected to run about 14 hours 30 minutes to 15 hours 30
minutes after startup, with roughly `+/- 1.5h` variance. A launch near
`2026-06-22 09:52 CST` puts expected completion near `2026-06-23 00:30 CST`.
