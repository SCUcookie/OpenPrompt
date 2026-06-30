# 2026-06-25 DIOR-R S3 Stability Long88 Launch

## Scope

Launch a clean 3-replica DIOR-R S3 long88 continuation from the completed
long60 `epoch_60.pth` checkpoints archived in
`New/docs/experiments/20260625_dior_r_s3_stability_long60_complete.md`.

This stays inside the current route gate: no S4, pseudo-labeling, FAIR1M, DOTA2
follow-up training, or route changes.

## Preflight

Checks immediately before launch on `2026-06-25 14:27 CST`:

- Source checkpoints confirmed:
  - `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep0_20260624/epoch_60.pth`
  - `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep1_20260624/epoch_60.pth`
  - `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep2_20260624/epoch_60.pth`
- `screen -ls` showed only `s0_result_log_monitor_20260603`.
- GPU preflight:

```text
0, GPU-dd3656f6-d07c-092d-ecfc-20f8141a7533, 14 MiB, 0 %
1, GPU-e5719698-edfe-bbe4-ff2c-686aff90c7df, 19731 MiB, 0 %
2, GPU-54fe0ffa-b4d2-d149-633e-e37e6e8e96c4, 14 MiB, 0 %
3, GPU-c69312cc-82d7-34eb-2ad2-93e414b706ce, 14 MiB, 0 %
4, GPU-dadccdb5-3cff-a378-4170-c3280452703d, 14 MiB, 0 %
5, GPU-6dd88eb7-c85d-e59d-1b47-30e9129f0eed, 14 MiB, 0 %
6, GPU-3775a44d-a175-9666-3200-7b6f2324adff, 22617 MiB, 13 %
```

GPU 1 and GPU 6 were occupied. GPUs 2, 3, and 4 were idle under the
`memory.used <= 4000 MiB` and `util <= 10%` gate, so the launch uses the
default mapping `rep0 -> GPU 2`, `rep1 -> GPU 3`, `rep2 -> GPU 4`.

## Configs

Prepared clean continuation workdirs under
`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/`:

- `roi_trans_remoteclip_s3_scene_adapter_stability_long88_rep0_20260625`
- `roi_trans_remoteclip_s3_scene_adapter_stability_long88_rep1_20260625`
- `roi_trans_remoteclip_s3_scene_adapter_stability_long88_rep2_20260625`

Each matching long60 config was copied into the new workdir and changed only
for:

- `max_epochs = 88`
- `train_cfg = dict(max_epochs=88, type='EpochBasedTrainLoop', val_interval=1)`
- `work_dir = '<new long88 workdir>'`
- `load_from = '<matching long60 epoch_60.pth>'`

Preserved settings include seeds `13407`, `14407`, `15407`, LR `2.5e-5`,
`param_scheduler = []`, prompt artifact path, scene-adapter settings,
checkpoint interval `1`, and validation interval `1`.

Config validation through
`PYTHONNOUSERSITE=1 /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/misc/print_config.py`
is required before launch.

## Launches

Launched from `/data5/2025/ldh/OpenRSD` at about `2026-06-25 14:30 CST`.

| Replica | GPU | PID | Screen | Config | Resume checkpoint | Launch log |
| --- | ---: | ---: | --- | --- | --- | --- |
| rep0 | 2 | `1652371` | `dior_r_s3_stability_long88_rep0_20260625_gpu2` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long88_rep0_20260625/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-stability-long88-rep0-20260625_dior_r.py` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep0_20260624/epoch_60.pth` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long88_rep0_20260625/launch_20260625_gpu2.log` |
| rep1 | 3 | `1652558` | `dior_r_s3_stability_long88_rep1_20260625_gpu3` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long88_rep1_20260625/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-stability-long88-rep1-20260625_dior_r.py` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep1_20260624/epoch_60.pth` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long88_rep1_20260625/launch_20260625_gpu3.log` |
| rep2 | 4 | `1652680` | `dior_r_s3_stability_long88_rep2_20260625_gpu4` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long88_rep2_20260625/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-stability-long88-rep2-20260625_dior_r.py` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep2_20260624/epoch_60.pth` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long88_rep2_20260625/launch_20260625_gpu4.log` |

## Startup Acceptance

Commands:

```bash
CUDA_VISIBLE_DEVICES=2 MPLCONFIGDIR=/tmp/matplotlib_dior_r_s3_long88_rep0_20260625 PYTHONNOUSERSITE=1 \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py \
  work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long88_rep0_20260625/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-stability-long88-rep0-20260625_dior_r.py \
  --work-dir work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long88_rep0_20260625 \
  --resume /data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep0_20260624/epoch_60.pth
```

```bash
CUDA_VISIBLE_DEVICES=3 MPLCONFIGDIR=/tmp/matplotlib_dior_r_s3_long88_rep1_20260625 PYTHONNOUSERSITE=1 \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py \
  work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long88_rep1_20260625/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-stability-long88-rep1-20260625_dior_r.py \
  --work-dir work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long88_rep1_20260625 \
  --resume /data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep1_20260624/epoch_60.pth
```

```bash
CUDA_VISIBLE_DEVICES=4 MPLCONFIGDIR=/tmp/matplotlib_dior_r_s3_long88_rep2_20260625 PYTHONNOUSERSITE=1 \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py \
  work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long88_rep2_20260625/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-stability-long88-rep2-20260625_dior_r.py \
  --work-dir work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long88_rep2_20260625 \
  --resume /data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep2_20260624/epoch_60.pth
```

## Startup Acceptance

Accepted on `2026-06-25` after all three conditions passed:

- screens detached and alive:
  - `1651863.dior_r_s3_stability_long88_rep0_20260625_gpu2`
  - `1652080.dior_r_s3_stability_long88_rep1_20260625_gpu3`
  - `1652223.dior_r_s3_stability_long88_rep2_20260625_gpu4`
- launch/runtime logs confirmed the intended resume checkpoints:
  - rep0: `Load checkpoint from ...stability_long60_rep0_20260624/epoch_60.pth`, `resumed epoch: 60, iter: 351720`
  - rep1: `Load checkpoint from ...stability_long60_rep1_20260624/epoch_60.pth`, `resumed epoch: 60, iter: 351720`
  - rep2: `Load checkpoint from ...stability_long60_rep2_20260624/epoch_60.pth`, `resumed epoch: 60, iter: 351720`
- each runtime log reached `Epoch(train) [61][ 200/5862]`:
  - rep0: `2026/06/25 14:31:40`
  - rep1: `2026/06/25 14:31:42`
  - rep2: `2026/06/25 14:31:40`

Runtime logs:

- rep0:
  `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long88_rep0_20260625/20260625_143004/20260625_143004.log`
- rep1:
  `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long88_rep1_20260625/20260625_143004/20260625_143004.log`
- rep2:
  `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long88_rep2_20260625/20260625_143004/20260625_143004.log`

Scoped failure scan across launch and runtime logs was clean for `Traceback`,
CUDA OOM, `out-of-memory`, `out of memory`, `libpng`, `CRC`, `NoneType`,
`ValueError`, `KeyboardInterrupt`, `loss: nan`, `loss: inf`, `grad_norm: nan`,
and `grad_norm: inf`.

Live GPU residency at acceptance:

```text
0, GPU-dd3656f6-d07c-092d-ecfc-20f8141a7533, 14 MiB, 0 %
1, GPU-e5719698-edfe-bbe4-ff2c-686aff90c7df, 19731 MiB, 0 %
2, GPU-54fe0ffa-b4d2-d149-633e-e37e6e8e96c4, 7125 MiB, 37 %
3, GPU-c69312cc-82d7-34eb-2ad2-93e414b706ce, 5633 MiB, 34 %
4, GPU-dadccdb5-3cff-a378-4170-c3280452703d, 9123 MiB, 41 %
5, GPU-6dd88eb7-c85d-e59d-1b47-30e9129f0eed, 14 MiB, 0 %
6, GPU-3775a44d-a175-9666-3200-7b6f2324adff, 22617 MiB, 73 %
```

`nvidia-smi --query-compute-apps` worker PIDs:

- GPU 2: `1652371`
- GPU 3: `1652558`
- GPU 4: `1652680`

`ps -p 1652371,1652558,1652680 -o pid,ppid,user,cmd --forest` confirmed the
three Python training commands and their matching resume checkpoints.

## ETA

This continuation resumes from epoch 60 and runs through epoch 88, so it has
28 resumed epochs left. Long32 and long60 each took about 13 hours for 28
resumed epochs; the expected runtime is at least 12 hours after startup.
