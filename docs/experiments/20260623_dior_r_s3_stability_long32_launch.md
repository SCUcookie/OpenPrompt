# 2026-06-23 DIOR-R S3 Stability Long32 Launch

## Scope

Launch a clean 3-GPU DIOR-R S3 long-stability confirmation continuation from
the 2026-06-16 annealed stability epoch-4 checkpoints. This is a stability
confirmation route only; it does not change the current paper claim boundary
before results exist.

S4, pseudo-labeling, FAIR1M, and DOTA2 follow-up training remain paused.

## Preflight

Checks before launch on `2026-06-23`:

- Source checkpoints confirmed:
  - `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_e4_lr2p5e5_rep0_20260616/epoch_4.pth`
  - `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_e4_lr2p5e5_rep1_20260616/epoch_4.pth`
  - `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_e4_lr2p5e5_rep2_20260616/epoch_4.pth`
- Prompt artifact confirmed:
  `/data5/2025/ldh/New/artifacts/generated/remoteclip_vit_b32_dior_r_s2_hierarchy_prompt_embeddings.pt`
- `screen -ls` showed only `s0_result_log_monitor_20260603`.
- GPU preflight before launch:

```text
0, 23939 MiB, 0 %
1, 14 MiB, 0 %
2, 14 MiB, 0 %
3, 14 MiB, 0 %
4, 14 MiB, 0 %
5, 14 MiB, 0 %
6, 19317 MiB, 70 %
```

GPU 0 and GPU 6 were occupied by VLLM processes. GPUs 1, 2, and 3 were idle,
so no remap was needed.

## Configs

Each continuation workdir was created cleanly under
`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/`. The matching
`20260616` annealed stability config was copied into the new workdir and
changed only for:

- `max_epochs = 32`
- `train_cfg = dict(max_epochs=32, type='EpochBasedTrainLoop', val_interval=1)`
- `work_dir = '<new continuation workdir>'`

Preserved settings include:

- seeds `13407`, `14407`, `15407`
- LR `2.5e-5`
- `param_scheduler = []`
- prompt artifact path
- S3 scene-adapter settings
- checkpoint interval `1`
- validation interval `1`

## Launches

Launched from `/data5/2025/ldh/OpenRSD` at about `2026-06-23 09:40 CST`.

| Replica | GPU | PID | Screen | Config | Resume checkpoint | Launch log |
| --- | ---: | ---: | --- | --- | --- | --- |
| rep0 | 1 | `2262989` | `dior_r_s3_stability_long32_rep0_20260623_gpu1` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long32_rep0_20260623/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-stability-long32-rep0-20260623_dior_r.py` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_e4_lr2p5e5_rep0_20260616/epoch_4.pth` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long32_rep0_20260623/launch_20260623_gpu1.log` |
| rep1 | 2 | `2262985` | `dior_r_s3_stability_long32_rep1_20260623_gpu2` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long32_rep1_20260623/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-stability-long32-rep1-20260623_dior_r.py` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_e4_lr2p5e5_rep1_20260616/epoch_4.pth` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long32_rep1_20260623/launch_20260623_gpu2.log` |
| rep2 | 3 | `2262999` | `dior_r_s3_stability_long32_rep2_20260623_gpu3` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long32_rep2_20260623/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-stability-long32-rep2-20260623_dior_r.py` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_e4_lr2p5e5_rep2_20260616/epoch_4.pth` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long32_rep2_20260623/launch_20260623_gpu3.log` |

Commands:

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_dior_r_s3_long32_rep0_20260623 PYTHONNOUSERSITE=1 \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py \
  work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long32_rep0_20260623/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-stability-long32-rep0-20260623_dior_r.py \
  --work-dir work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long32_rep0_20260623 \
  --resume /data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_e4_lr2p5e5_rep0_20260616/epoch_4.pth
```

```bash
CUDA_VISIBLE_DEVICES=2 MPLCONFIGDIR=/tmp/matplotlib_dior_r_s3_long32_rep1_20260623 PYTHONNOUSERSITE=1 \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py \
  work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long32_rep1_20260623/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-stability-long32-rep1-20260623_dior_r.py \
  --work-dir work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long32_rep1_20260623 \
  --resume /data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_e4_lr2p5e5_rep1_20260616/epoch_4.pth
```

```bash
CUDA_VISIBLE_DEVICES=3 MPLCONFIGDIR=/tmp/matplotlib_dior_r_s3_long32_rep2_20260623 PYTHONNOUSERSITE=1 \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py \
  work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long32_rep2_20260623/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-stability-long32-rep2-20260623_dior_r.py \
  --work-dir work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long32_rep2_20260623 \
  --resume /data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_e4_lr2p5e5_rep2_20260616/epoch_4.pth
```

## Startup Acceptance

Accepted on `2026-06-23` after all three conditions passed:

- screens detached and alive:
  - `2262489.dior_r_s3_stability_long32_rep0_20260623_gpu1`
  - `2262486.dior_r_s3_stability_long32_rep1_20260623_gpu2`
  - `2262485.dior_r_s3_stability_long32_rep2_20260623_gpu3`
- launch/runtime logs confirmed the intended resume checkpoints:
  - rep0: `Load checkpoint from ...rep0_20260616/epoch_4.pth`, `resumed epoch: 4, iter: 23448`
  - rep1: `Load checkpoint from ...rep1_20260616/epoch_4.pth`, `resumed epoch: 4, iter: 23448`
  - rep2: `Load checkpoint from ...rep2_20260616/epoch_4.pth`, `resumed epoch: 4, iter: 23448`
- each runtime log reached `Epoch(train) [5][200/5862]`:
  - rep0: `2026/06/23 09:43:02`
  - rep1: `2026/06/23 09:43:02`
  - rep2: `2026/06/23 09:43:16`

Runtime logs:

- rep0:
  `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long32_rep0_20260623/20260623_094043/20260623_094043.log`
- rep1:
  `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long32_rep1_20260623/20260623_094043/20260623_094043.log`
- rep2:
  `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long32_rep2_20260623/20260623_094043/20260623_094043.log`

Scoped failure scan across launch and runtime logs was clean for `Traceback`,
CUDA OOM, `out-of-memory`, `out of memory`, `libpng`, `CRC`, `NoneType`,
`ValueError`, `KeyboardInterrupt`, `loss: nan`, `loss: inf`, `grad_norm: nan`,
and `grad_norm: inf`.

Live GPU residency at acceptance:

```text
0, 23939 MiB, 0 %
1, 7467 MiB, 37 %
2, 8561 MiB, 57 %
3, 5615 MiB, 36 %
4, 14 MiB, 0 %
5, 14 MiB, 0 %
6, 19807 MiB, 71 %
```

`nvidia-smi --query-compute-apps` worker PIDs:

- GPU 1: `2262989`
- GPU 2: `2262985`
- GPU 3: `2262999`

## ETA

This continuation resumes from epoch 4 and runs through epoch 32, so it has
28 resumed epochs left. Expected runtime is roughly `12.5h-14.5h` after
startup, placing likely completion late on `2026-06-23 CST`.
