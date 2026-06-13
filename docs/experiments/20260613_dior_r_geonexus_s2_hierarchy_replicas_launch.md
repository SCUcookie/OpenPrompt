# 2026-06-13 DIOR-R GeoNexus S2 Hierarchy Replicas Launch

## Scope

Launched exactly three DIOR-R GeoNexus S2 hierarchy-regularizer replicas from completed S1 rep0 epoch 12.

- Source checkpoint: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s1_s0e52_rep0_20260613/epoch_12.pth`
- Prompt artifact: `/data5/2025/ldh/New/artifacts/generated/remoteclip_vit_b32_dior_r_s2_hierarchy_prompt_embeddings.pt`
- Taxonomy: `/data5/2025/ldh/New/assets/hierarchies/dior_r_remote_sensing_taxonomy.json`
- Templates: `/data5/2025/ldh/New/assets/prompts/prompt_templates.json`
- RemoteCLIP checkpoint: `/data5/2025/ldh/OpenRSD/checkpoints/remoteclip/RemoteCLIP-ViT-B-32.pt`

Artifact validation:

- `class_names`: 20, matching the DIOR-R S1 config order
- `embeddings`: shape `[20, 512]`, finite, row norms `0.9999998807907104` to `1.0000001192092896`
- `relation_matrix`: shape `[20, 20]`, finite, nonnegative, row sums `0.9999998807907104` to `1.0000001192092896`

## Replicas

Replica 0:

- GPU: 0
- Screen: `dior_r_geonexus_s2_s1e12_rep0_20260613_gpu0`
- Seed: 7407
- Workdir: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep0_20260613`
- Config: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep0_20260613/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-s1e12-rep0-20260613_dior_r.py`
- Launch log: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep0_20260613/launch_20260613_gpu0.log`
- Runtime log: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep0_20260613/20260613_154141/20260613_154141.log`
- Startup acceptance: reached `Epoch(train) [1][200/5862]` at 2026-06-13 15:42:37 CST

Replica 1:

- GPU: 1
- Screen: `dior_r_geonexus_s2_s1e12_rep1_20260613_gpu1`
- Seed: 8407
- Workdir: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep1_20260613`
- Config: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep1_20260613/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-s1e12-rep1-20260613_dior_r.py`
- Launch log: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep1_20260613/launch_20260613_gpu1.log`
- Runtime log: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep1_20260613/20260613_154141/20260613_154141.log`
- Startup acceptance: reached `Epoch(train) [1][200/5862]` at 2026-06-13 15:42:37 CST

Replica 2:

- GPU: 2
- Screen: `dior_r_geonexus_s2_s1e12_rep2_20260613_gpu2`
- Seed: 9407
- Workdir: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep2_20260613`
- Config: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep2_20260613/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-s1e12-rep2-20260613_dior_r.py`
- Launch log: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep2_20260613/launch_20260613_gpu2.log`
- Runtime log: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep2_20260613/20260613_154141/20260613_154141.log`
- Startup acceptance: reached `Epoch(train) [1][200/5862]` at 2026-06-13 15:42:38 CST

## Validation

- GPUs 0, 1, and 2 were idle before launch; unrelated GPU 4 activity was left untouched.
- All configs parsed and models built with `HierarchyPromptShared2FCBBoxHead`.
- Non-strict S1 epoch-12 load succeeded; missing keys were limited to new S2 prompt offsets and hierarchy relation buffers.
- Scoped failure scan over all three S2 workdirs found no matches for `Traceback`, CUDA OOM/out-of-memory, `libpng`, `CRC`, `NoneType`, `ValueError`, `KeyboardInterrupt`, `loss: nan/inf`, or `grad_norm: nan/inf`.
