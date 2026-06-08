# DOTA2 S1 Completion And S2 Launch - 2026-06-08

This note records the clean GPU-1 DOTA2 GeoNexus S1 completion and the DOTA2
S2 hierarchy-regularizer launch decision.

## S1 Gate

| Run | Status | Checkpoint | Metric |
| --- | --- | --- | --- |
| DOTA2 RoI Transformer S0 | complete | `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/epoch_12.pth` | `0.6088 / 0.6090` |
| DOTA2 S1 GPU 1 | complete | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607/epoch_12.pth` | `0.6177 / 0.6180` |
| DOTA2 S1 GPU 6 LR 1e-4 | active | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_lr1e4_20260607/epoch_8.pth` | epoch 8 `0.6017 / 0.6020` |
| DOTA2 S1 GPU 0 LR 5e-5 | active | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_lr5e5_20260607/epoch_8.pth` | epoch 8 `0.6046 / 0.6050` |

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
