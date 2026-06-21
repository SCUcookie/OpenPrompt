# 2026-06-20 DOTA2 S3 Scene-Adapter Long-12 Override Launch

## Scope

User-requested long-running 3-GPU experiment to occupy three GPUs for several
hours and test whether DOTA2 S3 scene-adapter behavior recovers beyond the
previous 4-epoch negative-to-neutral runs. This is an explicit route override:
S4, pseudo-labeling, FAIR1M, and routing changes remain paused.

## Rationale

Recent DOTA2 S3 scene-adapter replica packs completed cleanly but did not beat
the useful DOTA2 S2 loss-0 story by epoch 4. This launch extends the same S3
replica family to 12 epochs, preserving source checkpoints and seeds while
changing only run length and workdir.

## Preflight

Command: `screen -ls`

```text
There is a screen on:
        3470174.s0_result_log_monitor_20260603     (06/03/26 19:55:38)     (Detached)
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

## Configuration

Each config was copied from the matching 2026-06-19 S3 scene-adapter config
and changed only in:

- `max_epochs = 12`
- `train_cfg.max_epochs = 12`
- `work_dir = ..._long12_20260620`

Preserved controls:

- S3 `PromptShared2FCBBoxHead` with `use_scene_adapter=True`.
- DOTA2 valid-PNG taxonomy and `DOTADatasetClamp`.
- LR `5e-5`, batch size `2`, checkpoint interval `1`, validation interval `1`.
- Prompt artifact:
  `/data5/2025/ldh/New/artifacts/generated/remoteclip_vit_b32_dota2_s2_hierarchy_prompt_embeddings.pt`.

| Replica | GPU | Seed | Source checkpoint | Workdir |
| --- | ---: | ---: | --- | --- |
| rep3407 | 1 | 93407 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610/epoch_1.pth` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep3407_long12_20260620` |
| rep4407 | 2 | 94407 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep4407_20260610/epoch_3.pth` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep4407_long12_20260620` |
| rep5407 | 3 | 95407 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep5407_20260610/epoch_1.pth` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep5407_long12_20260620` |

## Launches

Launched from `/data5/2025/ldh/OpenRSD` at about `2026-06-20 10:47 CST`.

- `geonexus_dota2_s3_long12_rep3407_20260620_gpu1`
- `geonexus_dota2_s3_long12_rep4407_20260620_gpu2`
- `geonexus_dota2_s3_long12_rep5407_20260620_gpu3`

Launch logs:

- `work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep3407_long12_20260620/launch_20260620_gpu1.log`
- `work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep4407_long12_20260620/launch_20260620_gpu2.log`
- `work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep5407_long12_20260620/launch_20260620_gpu3.log`

## Live Status Acceptance

Checked on `2026-06-21 09:37 CST`. The pack is accepted as live and healthy,
but not complete. All three long-12 screens remain detached:

```text
919734.geonexus_dota2_s3_long12_rep5407_20260620_gpu3
919707.geonexus_dota2_s3_long12_rep4407_20260620_gpu2
919644.geonexus_dota2_s3_long12_rep3407_20260620_gpu1
```

Process residency:

```text
919644 SCREEN ... CUDA_VISIBLE_DEVICES=1 ... launch_20260620_gpu1.log
919707 SCREEN ... CUDA_VISIBLE_DEVICES=2 ... launch_20260620_gpu2.log
919734 SCREEN ... CUDA_VISIBLE_DEVICES=3 ... launch_20260620_gpu3.log
```

GPU residency:

```text
0, 14 MiB, 0 %
1, 23677 MiB, 36 %
2, 12679 MiB, 54 %
3, 16765 MiB, 35 %
4, 14 MiB, 0 %
5, 17565 MiB, 0 %
6, 17565 MiB, 0 %
```

The active Python worker PIDs reported by `nvidia-smi
--query-compute-apps` are `920119`, `920192`, and `920206` for the three
training jobs. GPUs 5 and 6 are occupied by VLLM engine processes, so they
remain excluded from any follow-up launch.

Runtime logs:

- rep3407:
  `work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep3407_long12_20260620/20260620_104724/20260620_104724.log`
- rep4407:
  `work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep4407_long12_20260620/20260620_104724/20260620_104724.log`
- rep5407:
  `work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep5407_long12_20260620/20260620_104724/20260620_104724.log`

Each run has checkpoint files through `epoch_10.pth` and has passed ten
complete validation rounds. Latest observed train markers:

- rep3407: `2026/06/21 09:37:15`, `Epoch(train) [11][10500/39007]`.
- rep4407: `2026/06/21 09:37:09`, `Epoch(train) [11][ 8950/39007]`.
- rep5407: `2026/06/21 09:37:16`, `Epoch(train) [11][ 9350/39007]`.

Scoped failure scan across all launch and runtime logs was clean for
`Traceback`, CUDA OOM, `out-of-memory`, `out of memory`, `libpng`, `CRC`,
`NoneType`, `ValueError`, `KeyboardInterrupt`, `loss: nan`, `loss: inf`,
`grad_norm: nan`, and `grad_norm: inf`.

## Current Metrics

Best and current/final are kept separate. These are current epoch-10 results,
not completion results.

| Replica | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 | Epoch 6 | Epoch 7 | Epoch 8 | Epoch 9 | Epoch 10 current | Best so far |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rep3407 | 0.6158 | 0.6216 | 0.6142 | 0.6134 | 0.6142 | 0.6169 | 0.6159 | 0.6146 | 0.6096 | 0.6119 | 0.6216 |
| rep4407 | 0.6177 | 0.6139 | 0.6154 | 0.6154 | 0.6154 | 0.6170 | 0.6149 | 0.6116 | 0.6151 | 0.6150 | 0.6177 |
| rep5407 | 0.6215 | 0.6211 | 0.6113 | 0.6197 | 0.6206 | 0.6211 | 0.6168 | 0.6185 | 0.6150 | 0.6151 | 0.6215 |

Current epoch-10 mean mAP is `0.6140`. Best-so-far mean across replicas is
`0.6203`. Final mean is still pending until epoch 12 validation completes.

## Follow-Up Gate

No fresh 3-GPU confirmation pack was launched during this archive update
because the June 20 long-12 pack is still active on GPUs 1, 2, and 3. The
follow-up pack from S2 loss-0 stability reps 6407/7407/8407 remains gated on
all three current screens exiting and a fresh idle-GPU preflight.
