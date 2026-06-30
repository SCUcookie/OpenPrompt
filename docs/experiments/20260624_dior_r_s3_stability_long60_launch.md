# 2026-06-24 DIOR-R S3 Stability Long60 Launch

## Scope

Launch a clean 3-replica DIOR-R S3 long60 continuation from the completed
long32 `epoch_32.pth` checkpoints archived in
`New/docs/experiments/20260624_dior_r_s3_stability_long32_complete.md`.

This stays inside the current route gate: no S4, pseudo-labeling, FAIR1M, DOTA2
follow-up training, or route changes.

## Preflight

Checks immediately before launch on `2026-06-24 10:20 CST`:

- Source checkpoints confirmed:
  - `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long32_rep0_20260623/epoch_32.pth`
  - `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long32_rep1_20260623/epoch_32.pth`
  - `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long32_rep2_20260623/epoch_32.pth`
- `screen -ls` showed only `s0_result_log_monitor_20260603`.
- GPU preflight:

```text
0, GPU-dd3656f6-d07c-092d-ecfc-20f8141a7533, 14 MiB, 0 %
1, GPU-e5719698-edfe-bbe4-ff2c-686aff90c7df, 10727 MiB, 90 %
2, GPU-54fe0ffa-b4d2-d149-633e-e37e6e8e96c4, 14 MiB, 0 %
3, GPU-c69312cc-82d7-34eb-2ad2-93e414b706ce, 14 MiB, 0 %
4, GPU-dadccdb5-3cff-a378-4170-c3280452703d, 14 MiB, 0 %
5, GPU-6dd88eb7-c85d-e59d-1b47-30e9129f0eed, 14 MiB, 0 %
6, GPU-3775a44d-a175-9666-3200-7b6f2324adff, 22615 MiB, 22 %
```

GPU 1 and GPU 6 were occupied. GPUs 2, 3, and 4 were idle under the
`memory.used <= 4000 MiB` and `util <= 10%` gate, so the launch used the
default mapping `rep0 -> GPU 2`, `rep1 -> GPU 3`, `rep2 -> GPU 4`.

## Configs

Created clean continuation workdirs under
`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/`:

- `roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep0_20260624`
- `roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep1_20260624`
- `roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep2_20260624`

Each matching long32 config was copied into the new workdir and changed only
for:

- `max_epochs = 60`
- `train_cfg = dict(max_epochs=60, type='EpochBasedTrainLoop', val_interval=1)`
- `work_dir = '<new long60 workdir>'`

Preserved settings include seeds `13407`, `14407`, `15407`, LR `2.5e-5`,
`param_scheduler = []`, prompt artifact path, scene-adapter settings,
checkpoint interval `1`, and validation interval `1`.

Config validation through
`PYTHONNOUSERSITE=1 /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/misc/print_config.py`
succeeded for all three generated configs.

## Launches

Launched from `/data5/2025/ldh/OpenRSD` at about `2026-06-24 10:21 CST`.

| Replica | GPU | PID | Screen | Config | Resume checkpoint | Launch log |
| --- | ---: | ---: | --- | --- | --- | --- |
| rep0 | 2 | `2481407` | `dior_r_s3_stability_long60_rep0_20260624_gpu2` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep0_20260624/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-stability-long60-rep0-20260624_dior_r.py` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long32_rep0_20260623/epoch_32.pth` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep0_20260624/launch_20260624_gpu2.log` |
| rep1 | 3 | `2481491` | `dior_r_s3_stability_long60_rep1_20260624_gpu3` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep1_20260624/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-stability-long60-rep1-20260624_dior_r.py` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long32_rep1_20260623/epoch_32.pth` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep1_20260624/launch_20260624_gpu3.log` |
| rep2 | 4 | `2481542` | `dior_r_s3_stability_long60_rep2_20260624_gpu4` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep2_20260624/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-stability-long60-rep2-20260624_dior_r.py` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long32_rep2_20260623/epoch_32.pth` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep2_20260624/launch_20260624_gpu4.log` |

Commands:

```bash
CUDA_VISIBLE_DEVICES=2 MPLCONFIGDIR=/tmp/matplotlib_dior_r_s3_long60_rep0_20260624 PYTHONNOUSERSITE=1 \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py \
  work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep0_20260624/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-stability-long60-rep0-20260624_dior_r.py \
  --work-dir work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep0_20260624 \
  --resume /data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long32_rep0_20260623/epoch_32.pth
```

```bash
CUDA_VISIBLE_DEVICES=3 MPLCONFIGDIR=/tmp/matplotlib_dior_r_s3_long60_rep1_20260624 PYTHONNOUSERSITE=1 \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py \
  work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep1_20260624/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-stability-long60-rep1-20260624_dior_r.py \
  --work-dir work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep1_20260624 \
  --resume /data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long32_rep1_20260623/epoch_32.pth
```

```bash
CUDA_VISIBLE_DEVICES=4 MPLCONFIGDIR=/tmp/matplotlib_dior_r_s3_long60_rep2_20260624 PYTHONNOUSERSITE=1 \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py \
  work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep2_20260624/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-stability-long60-rep2-20260624_dior_r.py \
  --work-dir work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep2_20260624 \
  --resume /data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long32_rep2_20260623/epoch_32.pth
```

## Startup Acceptance

Accepted on `2026-06-24` after all three conditions passed:

- screens detached and alive:
  - `2480930.dior_r_s3_stability_long60_rep0_20260624_gpu2`
  - `2481007.dior_r_s3_stability_long60_rep1_20260624_gpu3`
  - `2481138.dior_r_s3_stability_long60_rep2_20260624_gpu4`
- launch/runtime logs confirmed the intended resume checkpoints:
  - rep0: `Load checkpoint from ...stability_long32_rep0_20260623/epoch_32.pth`, `resumed epoch: 32, iter: 187584`
  - rep1: `Load checkpoint from ...stability_long32_rep1_20260623/epoch_32.pth`, `resumed epoch: 32, iter: 187584`
  - rep2: `Load checkpoint from ...stability_long32_rep2_20260623/epoch_32.pth`, `resumed epoch: 32, iter: 187584`
- each runtime log reached `Epoch(train) [33][200/5862]`:
  - rep0: `2026/06/24 10:22:29`
  - rep1: `2026/06/24 10:22:28`
  - rep2: `2026/06/24 10:22:28`

Runtime logs:

- rep0:
  `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep0_20260624/20260624_102111/20260624_102111.log`
- rep1:
  `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep1_20260624/20260624_102111/20260624_102111.log`
- rep2:
  `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep2_20260624/20260624_102111/20260624_102111.log`

Scoped failure scan across launch and runtime logs was clean for `Traceback`,
CUDA OOM, `out-of-memory`, `out of memory`, `libpng`, `CRC`, `NoneType`,
`ValueError`, `KeyboardInterrupt`, `loss: nan`, `loss: inf`, `grad_norm: nan`,
and `grad_norm: inf`.

Live GPU residency at acceptance:

```text
0, GPU-dd3656f6-d07c-092d-ecfc-20f8141a7533, 14 MiB, 0 %
1, GPU-e5719698-edfe-bbe4-ff2c-686aff90c7df, 10727 MiB, 86 %
2, GPU-54fe0ffa-b4d2-d149-633e-e37e6e8e96c4, 6627 MiB, 34 %
3, GPU-c69312cc-82d7-34eb-2ad2-93e414b706ce, 6543 MiB, 43 %
4, GPU-dadccdb5-3cff-a378-4170-c3280452703d, 7065 MiB, 37 %
5, GPU-6dd88eb7-c85d-e59d-1b47-30e9129f0eed, 14 MiB, 0 %
6, GPU-3775a44d-a175-9666-3200-7b6f2324adff, 22615 MiB, 61 %
```

`nvidia-smi --query-compute-apps` worker PIDs:

- GPU 2: `2481407`
- GPU 3: `2481491`
- GPU 4: `2481542`

`ps -p 2481407,2481491,2481542 -o pid,ppid,user,cmd --forest` confirmed the
three Python training commands and their matching resume checkpoints.

## ETA

This continuation resumes from epoch 32 and runs through epoch 60, so it has
28 resumed epochs left. Long32 took about 13 hours for 28 resumed epochs; the
expected runtime is at least 12 hours after startup.
