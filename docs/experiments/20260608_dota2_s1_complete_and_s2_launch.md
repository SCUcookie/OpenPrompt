# DOTA2 S1 Completion And S2 Launch - 2026-06-08

This note records the clean GPU-1 DOTA2 GeoNexus S1 completion and the DOTA2
S2 hierarchy-regularizer launch decision.

## S1 Gate

| Run | Status | Checkpoint | Metric |
| --- | --- | --- | --- |
| DOTA2 RoI Transformer S0 | complete | `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/epoch_12.pth` | `0.6088 / 0.6090` |
| DOTA2 S1 GPU 1 | complete | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607/epoch_12.pth` | `0.6177 / 0.6180` |
| DOTA2 S1 GPU 6 LR 1e-4 | complete | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_lr1e4_20260607/epoch_12.pth` | final `0.5997 / 0.6000` |
| DOTA2 S1 GPU 0 LR 5e-5 | complete | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_lr5e5_20260607/epoch_12.pth` | final `0.6047 / 0.6050` |

GPU-1 S1 finished cleanly at `2026-06-08 09:33:42 +0800`. Its final epoch 12
result is above the S0 RoI Transformer gate and above the current GPU-0/GPU-6
replicate evidence, so S2 is initialized from GPU-1 `epoch_12.pth` without
waiting for the still-active S1 replicates.

## S2 Setup

Artifact:

`/data5/2025/ldh/New/artifacts/generated/remoteclip_vit_b32_dota2_s2_hierarchy_prompt_embeddings.pt`

Validation:

- `class_names`: 18
- `embeddings`: `[18, 512]`, finite, row norms approximately 1.0
- `relation_matrix`: `[18, 18]`, finite

Config:

`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_20260608/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-s1e12-20260608_dota2.py`

Workdir:

`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_20260608`

The S2 config replaces both cascade heads with
`HierarchyPromptShared2FCBBoxHead`, uses `num_classes=18`, loads the S2
hierarchy prompt artifact with `prompt_embedding_key='embeddings'` and
`hierarchy_relation_key='relation_matrix'`, sets `hierarchy_loss_weight=0.05`,
`hierarchy_target_self_weight=0.8`, and enables learnable prompt bias and
offsets.

## Preflight

- Prompt artifact validation passed.
- `tools/bootstrap_run.py tools/misc/print_config.py ... --cfg-options train_cfg.max_epochs=1`
  parsed successfully.
- Parsed config confirmed both cascade bbox heads are
  `HierarchyPromptShared2FCBBoxHead`, `num_classes=18`, and the DOTA2 S2
  artifact path is used.

## Launch Command

```bash
cd /data5/2025/ldh/OpenRSD
screen -dmS geonexus_dota2_roi_trans_s2_hierarchy_reg_s1e12_20260608_gpu1 bash -lc \
'CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/geonexus_dota2_s2_hierarchy_s1e12_20260608 \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py \
  tools/train.py \
  work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_20260608/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-s1e12-20260608_dota2.py \
  --work-dir work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_20260608 \
  2>&1 | tee work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_20260608/launch_20260608_gpu1.log'
```

Startup acceptance target: detached screen exists, GPU 1 shows a Python
process, log reaches `Epoch(train) [1][200/39007]`, and no `Traceback`, CUDA
OOM, `libpng`, `CRC`, `NoneType`, `ValueError`, `nan`, or `inf` appears before
acceptance.

Startup acceptance passed at `2026-06-08 11:00:05 +0800`:

- Screen `715887.geonexus_dota2_roi_trans_s2_hierarchy_reg_s1e12_20260608_gpu1`
  exists detached.
- `nvidia-smi` shows PID `716070` on GPU 1.
- Launch log reached `Epoch(train) [1][200/39007]`.
- Losses were finite and both `s0.loss_hierarchy` and `s1.loss_hierarchy`
  appeared.
- Strict failure scan found no `Traceback`, CUDA OOM, `libpng`, `CRC`,
  `NoneType`, `ValueError`, `nan`, or `inf` signature before acceptance.

## 2026-06-08 Three-GPU S2 Fill

Current DOTA2 S2 hierarchy-regularizer evidence:

- Main S2 run `roi_trans_remoteclip_s2_hierarchy_reg_s1e12_20260608` remains active on GPU 1.
- Epoch-4 validation was `dota/mAP=0.6038`, `dota/AP50=0.6040`, below main S1 `0.6177 / 0.6180`.
- S3/S4 remain paused until S2 produces stronger stable evidence.

Two stabilization variants were launched from the same main S1 checkpoint
`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607/epoch_12.pth`:

| GPU | Run | Config change | Screen | Acceptance |
| --- | --- | --- | --- | --- |
| 0 | `roi_trans_remoteclip_s2_hierarchy_reg_s1e12_lr1e4_20260608` | optimizer LR `1e-4`; hierarchy weight kept `0.05`; `learnable_prompt_offsets=True` | `geonexus_dota2_roi_trans_s2_hierarchy_reg_s1e12_lr1e4_20260608_gpu0` | `Epoch(train) [1][200/39007]`, finite losses, `s0.loss_hierarchy=0.0383`, `s1.loss_hierarchy=0.0414` |
| 6 | `roi_trans_remoteclip_s2_hierarchy_reg_s1e12_hw1e2_20260608` | optimizer LR kept `0.0025`; both cascade hierarchy weights `0.01`; `learnable_prompt_offsets=True` | `geonexus_dota2_roi_trans_s2_hierarchy_reg_s1e12_hw1e2_20260608_gpu6` | `Epoch(train) [1][200/39007]`, finite losses, `s0.loss_hierarchy=0.0074`, `s1.loss_hierarchy=0.0078` |

Before launch, GPUs 0 and 6 passed three idle polls with `memory.used=14 MiB`
and `utilization.gpu=0%`. The fresh launch logs had no `Traceback`, CUDA OOM,
`libpng`, `CRC`, `NoneType`, `ValueError`, `nan`, or `inf` reject signature at
acceptance.

Live process snapshot at `2026-06-08 19:49 CST`:

- GPU 0: PID `2711973`, low-LR S2 variant, latest observed memory about `15400 MiB`.
- GPU 1: PID `716070`, main S2 hierarchy regularizer, latest observed memory about `19116 MiB`.
- GPU 6: PID `2711971`, reduced-hierarchy S2 variant, latest observed memory about `19644 MiB`.

Estimated finish from log ETAs at `2026-06-08 19:46:53 CST`:

- GPU 1 main S2: about `2026-06-09 08:15 CST`.
- GPU 0 low-LR S2: about `2026-06-09 19:04 CST`.
- GPU 6 reduced-hierarchy S2: about `2026-06-09 19:16 CST`.

Detailed live-log snapshot:

`/data5/2025/ldh/New/logs/dota2_three_gpu_s2_20260608.log`
