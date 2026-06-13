# 2026-06-13 DIOR-R GeoNexus S1 S0-E52 Replicas Complete

## Scope

Completed two DIOR-R GeoNexus S1 RoI Transformer + RemoteCLIP replicas from the sanitized DIOR-R S0 RoI Transformer checkpoint:

- Source checkpoint: `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_roi_trans_sanitized_long_20260612_gpu1/epoch_52.pth`
- Prompt taxonomy: `/data5/2025/ldh/New/assets/hierarchies/dior_r_remote_sensing_taxonomy.json`
- Prompt artifact: `/data5/2025/ldh/New/artifacts/generated/remoteclip_vit_b32_dior_r_prompt_embeddings.pt`
- Schedule: 12 epochs, validation and checkpoint interval 4

Both replicas completed cleanly. `screen -ls` after completion showed no active S1 training screens; only `s0_result_log_monitor_20260603` remained detached.

## Replica 0

- GPU: 0
- Screen: `dior_r_geonexus_s1_s0e52_rep0_20260613_gpu0`
- Workdir: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s1_s0e52_rep0_20260613`
- Config: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s1_s0e52_rep0_20260613/roi-trans-le90_r50_fpn_remoteclip-s1-s0e52-rep0-20260613_dior_r.py`
- Launch log: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s1_s0e52_rep0_20260613/launch_20260613_gpu0.log`
- Runtime log: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s1_s0e52_rep0_20260613/20260613_112328/20260613_112328.log`
- Seed: 5407
- Checkpoint: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s1_s0e52_rep0_20260613/epoch_12.pth`

Metrics:

- Epoch 4: `dota/mAP=0.6515141726`, `dota/AP50=0.652`
- Epoch 8: `dota/mAP=0.6688991189`, `dota/AP50=0.669`
- Epoch 12 final/best: `dota/mAP=0.6750815511`, `dota/AP50=0.675`

## Replica 1

- GPU: 1
- Screen: `dior_r_geonexus_s1_s0e52_rep1_20260613_gpu1`
- Workdir: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s1_s0e52_rep1_20260613`
- Config: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s1_s0e52_rep1_20260613/roi-trans-le90_r50_fpn_remoteclip-s1-s0e52-rep1-20260613_dior_r.py`
- Launch log: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s1_s0e52_rep1_20260613/launch_20260613_gpu1.log`
- Runtime log: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s1_s0e52_rep1_20260613/20260613_112328/20260613_112328.log`
- Seed: 6407
- Checkpoint: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s1_s0e52_rep1_20260613/epoch_12.pth`

Metrics:

- Epoch 4: `dota/mAP=0.6574794650`, `dota/AP50=0.657`
- Epoch 8: `dota/mAP=0.6687465906`, `dota/AP50=0.669`
- Epoch 12 final/best: `dota/mAP=0.6689543724`, `dota/AP50=0.669`

## Failure Scan

Scoped scan over both completed S1 workdirs found no matches for:

`Traceback`, CUDA OOM/out-of-memory, `libpng`, `CRC`, `NoneType`, `ValueError`, `KeyboardInterrupt`, `loss: nan/inf`, or `grad_norm: nan/inf`.

## Interpretation

Replica 0 is the stronger DIOR-R GeoNexus S1 result and becomes the source checkpoint for DIOR-R S2 hierarchy-regularizer replicas. Replica 1 confirms the S1 result is stable around `0.669` to `0.675` AP50 on the sanitized DIOR-R protocol.
