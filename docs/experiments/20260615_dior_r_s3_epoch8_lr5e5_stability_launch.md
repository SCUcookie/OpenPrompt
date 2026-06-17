# DIOR-R S3 Epoch-8 LR5e-5 Stability Launch

Date: 2026-06-15

## Objective

Run a controlled S3 stability follow-up from each replica's S3 epoch-8 best checkpoint to test whether the scene-adapter gain remains present in final checkpoints.

This launch keeps pseudo-labeling, FAIR1M, S4, and submission-positioning work paused.

## Configuration

- rep0 config: `OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_epoch8_lr5e5_stability_rep0_20260615/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-epoch8-lr5e5-stability-rep0-20260615_dior_r.py`
- rep1 config: `OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_epoch8_lr5e5_stability_rep1_20260615/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-epoch8-lr5e5-stability-rep1-20260615_dior_r.py`
- rep2 config: `OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_epoch8_lr5e5_stability_rep2_20260615/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-epoch8-lr5e5-stability-rep2-20260615_dior_r.py`

Shared stability controls:

- `max_epochs=4`
- `train_cfg.val_interval=1`
- `ckpt_interval=1`
- `default_hooks.checkpoint.interval=1`
- `optim_wrapper.optimizer.lr=5e-5`
- `param_scheduler=[]`
- S3 scene-adapter settings unchanged: `use_scene_adapter=True`, `scene_adapter_dim=256`, `scene_adapter_identity_init=True`, `scene_adapter_residual_scale=0.1`

Replica mapping:

| replica | GPU | seed | source checkpoint | screen |
|---|---:|---:|---|---|
| rep0 | 0 | 13407 | `roi_trans_remoteclip_s3_scene_adapter_s2e12_rep0_20260614/epoch_8.pth` | `dior_r_s3_stability_lr5e5_rep0_20260615_gpu0` |
| rep1 | 1 | 14407 | `roi_trans_remoteclip_s3_scene_adapter_s2e12_rep1_20260614/epoch_8.pth` | `dior_r_s3_stability_lr5e5_rep1_20260615_gpu1` |
| rep2 | 2 | 15407 | `roi_trans_remoteclip_s3_scene_adapter_s2e12_rep2_20260614/epoch_8.pth` | `dior_r_s3_stability_lr5e5_rep2_20260615_gpu2` |

## Preflight

- Only existing screen before launch: `s0_result_log_monitor_20260603`.
- GPUs 0-2 idle at preflight: each reported 14 MiB used and 0% utilization.
- All three source `epoch_8.pth` checkpoints exist.
- Prompt embedding artifact exists: `/data5/2025/ldh/New/artifacts/generated/remoteclip_vit_b32_dior_r_s2_hierarchy_prompt_embeddings.pt`.
- Non-training config validation passed for load checkpoint, workdir, LR, epoch count, validation/checkpoint intervals, seed, prompt artifact path, and scene-adapter settings.

## Acceptance Checks

- Each launch log must show loading its own epoch-8 checkpoint.
- Each run must reach `Epoch(train) [1][200/5862]`.
- Scan logs for: `Traceback`, `CUDA OOM`, `out-of-memory`, `libpng`, `CRC`, `NoneType`, `ValueError`, `KeyboardInterrupt`, `loss: nan`, `loss: inf`, `grad_norm: nan`, `grad_norm: inf`.

Startup acceptance at 2026-06-15 09:43 Asia/Shanghai:

- rep0 loaded `roi_trans_remoteclip_s3_scene_adapter_s2e12_rep0_20260614/epoch_8.pth` and reached `Epoch(train) [1][ 200/5862]`.
- rep1 loaded `roi_trans_remoteclip_s3_scene_adapter_s2e12_rep1_20260614/epoch_8.pth` and reached `Epoch(train) [1][ 200/5862]`.
- rep2 loaded `roi_trans_remoteclip_s3_scene_adapter_s2e12_rep2_20260614/epoch_8.pth` and reached `Epoch(train) [1][ 200/5862]`.
- At acceptance, GPUs 0-2 were active and the three stability screens remained detached.
- Initial failure-signature scan found no matches.

## Completion Plan

- Parse each run's `vis_data/scalars.json` after completion.
- Confirm four validation points per replica.
- Archive epoch 1-4 metrics, best mean, and final mean.
- Compare against S3 epoch-8 best mean `0.6979`, original S3 final mean `0.6859`, and S2 final mean `0.6856`.
- Regenerate `New/artifacts/result_assets_20260614/all_experiment_results_20260614.csv` with the stability rows included.
