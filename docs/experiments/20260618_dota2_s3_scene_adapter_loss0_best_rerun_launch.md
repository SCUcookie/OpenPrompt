# DOTA2 GeoNexus S3 Scene-Adapter Loss-0 Best Replicas Rerun Launch

Date: 2026-06-18

## Objective

Rerun the established DOTA2 S3 scene-adapter replica family across three GPUs
in parallel, using the same loss-0-best source checkpoints and the same DOTA2
`ss_val` protocol.

## Configuration

| Replica | GPU | Seed | Source checkpoint | Workdir |
| --- | ---: | ---: | --- | --- |
| rep3407 | 2 | 93407 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610/epoch_1.pth` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep3407_20260618` |
| rep4407 | 3 | 94407 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep4407_20260610/epoch_3.pth` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep4407_20260618` |
| rep5407 | 4 | 95407 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep5407_20260610/epoch_1.pth` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep5407_20260618` |

The rerun keeps the approved S3 config family unchanged:

- `PromptShared2FCBBoxHead`
- `use_scene_adapter=True`
- `scene_adapter_dim=256`
- `scene_adapter_identity_init=True`
- `scene_adapter_residual_scale=0.1`

Preserved controls:

- DOTA2 18-class taxonomy.
- `DOTADatasetClamp`.
- `train/annfiles_validpng_20260602/` and `ss_val/annfiles/`.
- Batch size `2`.
- LR `5e-5`.
- `max_epochs=4`.
- Checkpoint interval `1`.

Prompt artifact:

- `/data5/2025/ldh/New/artifacts/generated/remoteclip_vit_b32_dota2_s2_hierarchy_prompt_embeddings.pt`

## Preflight

- The source checkpoints above exist.
- The prompt artifact above exists.
- The prior 2026-06-15 S3 launch note confirms the matching config shape and
  the established seed mapping for `rep3407`, `rep4407`, and `rep5407`.
- `screen -ls` shows only the detached `s0_result_log_monitor_20260603`
  entry in this session.
- The initial sandboxed shell could not access `/dev/nvidia*`; an escalated
  host check confirmed GPU access.
- GPUs 0 and 1 were occupied by unrelated live Python processes, and GPU 6 was
  occupied by a VLLM process. The rerun was launched on idle GPUs 2, 3, and 4.

## Status

Launched in detached `screen` sessions:

- `geonexus_dota2_s3_scene_rerun_rep3407_20260618_gpu2`
- `geonexus_dota2_s3_scene_rerun_rep4407_20260618_gpu3`
- `geonexus_dota2_s3_scene_rerun_rep5407_20260618_gpu4`

Launch logs:

- `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep3407_20260618/launch_20260618_gpu2.log`
- `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep4407_20260618/launch_20260618_gpu3.log`
- `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep5407_20260618/launch_20260618_gpu4.log`

Initial post-launch check showed all three screens detached and alive, with
GPUs 2, 3, and 4 each at about 1941 MiB.

Startup acceptance remains pending until each log reaches:
`Epoch(train) [1][  200/39007]` with a clean failure scan.
