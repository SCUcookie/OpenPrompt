# DOTA2 GeoNexus S3 Scene-Adapter Loss-0 Best Replicas Launch

Date: 2026-06-15

## Objective

Launch three DOTA2 S3 scene-adapter replicas from the strongest DOTA2 S2
loss-0 early checkpoints. Keep S4, pseudo-labeling, and FAIR1M closed.

## Configuration

| Replica | GPU | Seed | Source checkpoint | Workdir |
| --- | ---: | ---: | --- | --- |
| rep3407 | 0 | 93407 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610/epoch_1.pth` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep3407_20260615` |
| rep4407 | 1 | 94407 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep4407_20260610/epoch_3.pth` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep4407_20260615` |
| rep5407 | 2 | 95407 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep5407_20260610/epoch_1.pth` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep5407_20260615` |

Each config was copied from the matching DOTA2 S2 loss-0 config and changed
from `HierarchyPromptShared2FCBBoxHead` to `PromptShared2FCBBoxHead` in both
cascade heads. Hierarchy-only keys were removed, and both heads now use:
`prompt_dim=512`, `use_scene_adapter=True`, `scene_adapter_dim=256`,
`scene_adapter_identity_init=True`, and `scene_adapter_residual_scale=0.1`.

Preserved controls:

- DOTA2 18-class list.
- `DOTADatasetClamp`.
- `train/annfiles_validpng_20260602/` and `ss_val/annfiles/`.
- Batch size `2`.
- LR `5e-5`, `max_epochs=4`, `val_interval=1`, checkpoint interval `1`.
- Prompt artifact `/data5/2025/ldh/New/artifacts/generated/remoteclip_vit_b32_dota2_s2_hierarchy_prompt_embeddings.pt`.

## Preflight

- Source checkpoints and prompt artifact existed.
- MMEngine config loading passed using the training bootstrap import order.
- Prompt artifact contained 18 embeddings.
- All configs had `num_classes=18`, both heads had scene-adapter enabled, and
  `load_from` matched the intended source checkpoint.
- GPU preflight: GPUs 0, 1, and 2 were idle at 14 MiB and 0% utilization.
  GPU 6 was occupied at about 20 GiB and was not used.

## Launch

Screens:

- `geonexus_dota2_s3_scene_rep3407_20260615_gpu0`
- `geonexus_dota2_s3_scene_rep4407_20260615_gpu1`
- `geonexus_dota2_s3_scene_rep5407_20260615_gpu2`

Launch logs:

- `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep3407_20260615/launch_20260615_gpu0.log`
- `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep4407_20260615/launch_20260615_gpu1.log`
- `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep5407_20260615/launch_20260615_gpu2.log`

## Startup Monitoring

As of the initial post-launch check, all three screens remained detached and
alive, GPUs 0-2 each had about 1941 MiB allocated, and logs were in DOTA2
annotation initialization at `1877/170831`. No failure signatures matched.

Startup acceptance is still pending until each runtime log reaches:
`Epoch(train) [1][  200/39007]`.

Failure scan patterns:
`Traceback`, `CUDA OOM`, `out-of-memory`, `out of memory`, `libpng`, `CRC`,
`NoneType`, `ValueError`, `KeyboardInterrupt`, `loss: nan`, `loss: inf`,
`grad_norm: nan`, `grad_norm: inf`.

## Completion Plan

After completion, require four validation points per replica. Archive best and
final metrics separately and compare against DOTA2 S1 `0.6177`, S2 loss-0 best
mean `0.6206`, and S2 loss-0 final mean `0.6167`. Treat DOTA2 S3 as
exploratory unless it beats S2 best mean or clearly stabilizes final
checkpoints.
