# DIOR-R GeoNexus S3 Scene-Adapter Replicas Launch

Date: 2026-06-14

## Purpose

Launch three DIOR-R GeoNexus S3 scene-adapter replicas from the strongest
completed S2 checkpoint, rep4 epoch 12.

## Shared Protocol

- Source checkpoint: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep4_20260614/epoch_12.pth`
- Source S2 metric: rep4 epoch 12, `dota/mAP=0.6914003491`
- Prompt artifact: `/data5/2025/ldh/New/artifacts/generated/remoteclip_vit_b32_dior_r_s2_hierarchy_prompt_embeddings.pt`
- Template config: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep4_20260614/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-s1e12-rep4-20260614_dior_r.py`
- Dataset root: `data/DIOR_R_dota_sanitized_invalidsize_20260612/`
- Train labels/images: `train_val/labelTxt/`, `train_val/images/`
- Test labels/images: `test/labelTxt/`, `test/images/`
- Classes: 20
- `max_epochs`: 12
- `val_interval`: 4
- `ckpt_interval`: 4

## S3 Head Changes

The two cascade bbox heads use `PromptShared2FCBBoxHead` from
`geonexus_mmrotate.prompt_bbox_head` with `prompt_dim=512`,
`use_scene_adapter=True`, `scene_adapter_dim=256`,
`scene_adapter_identity_init=True`, and `scene_adapter_residual_scale=0.1`.
S2-only hierarchy fields were removed.

## Preflight

- GPUs 0, 1, and 2 were idle before launch: each reported 14 MiB used and 0%
  utilization.
- No active DIOR-R training screens were present before launch.
- Source checkpoint and prompt artifact existed.
- Generated configs passed field checks for S3 imports, head type, scene-adapter
  settings, seed, workdir, source checkpoint, `max_epochs=12`,
  `val_interval=4`, and `ckpt_interval=4`.

## Replicas

| Replica | GPU | Seed | Screen | Workdir | Config | Launch log | Runtime log | Startup status |
| --- | ---: | ---: | --- | --- | --- | --- | --- | --- |
| rep0 | 0 | 13407 | `dior_r_geonexus_s3_s2e12_rep0_20260614_gpu0` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_s2e12_rep0_20260614` | `roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-s2e12-rep0-20260614_dior_r.py` | `launch_20260614_gpu0.log` | `20260614_161203/20260614_161203.log` | accepted: loaded S2 rep4 epoch 12 and reached `Epoch(train) [1][200/5862]`; no failure signatures |
| rep1 | 1 | 14407 | `dior_r_geonexus_s3_s2e12_rep1_20260614_gpu1` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_s2e12_rep1_20260614` | `roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-s2e12-rep1-20260614_dior_r.py` | `launch_20260614_gpu1.log` | `20260614_161203/20260614_161203.log` | accepted: loaded S2 rep4 epoch 12 and reached `Epoch(train) [1][200/5862]`; no failure signatures |
| rep2 | 2 | 15407 | `dior_r_geonexus_s3_s2e12_rep2_20260614_gpu2` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_s2e12_rep2_20260614` | `roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-s2e12-rep2-20260614_dior_r.py` | `launch_20260614_gpu2.log` | `20260614_161203/20260614_161203.log` | accepted: loaded S2 rep4 epoch 12 and reached `Epoch(train) [1][200/5862]`; no failure signatures |

## Launch Commands

```bash
cd /data5/2025/ldh/OpenRSD

screen -dmS dior_r_geonexus_s3_s2e12_rep0_20260614_gpu0 bash -lc 'CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib_dior_r_s3_rep0_20260614 PYTHONNOUSERSITE=1 /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_s2e12_rep0_20260614/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-s2e12-rep0-20260614_dior_r.py --work-dir work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_s2e12_rep0_20260614 > work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_s2e12_rep0_20260614/launch_20260614_gpu0.log 2>&1'

screen -dmS dior_r_geonexus_s3_s2e12_rep1_20260614_gpu1 bash -lc 'CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_dior_r_s3_rep1_20260614 PYTHONNOUSERSITE=1 /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_s2e12_rep1_20260614/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-s2e12-rep1-20260614_dior_r.py --work-dir work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_s2e12_rep1_20260614 > work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_s2e12_rep1_20260614/launch_20260614_gpu1.log 2>&1'

screen -dmS dior_r_geonexus_s3_s2e12_rep2_20260614_gpu2 bash -lc 'CUDA_VISIBLE_DEVICES=2 MPLCONFIGDIR=/tmp/matplotlib_dior_r_s3_rep2_20260614 PYTHONNOUSERSITE=1 /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_s2e12_rep2_20260614/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-s2e12-rep2-20260614_dior_r.py --work-dir work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_s2e12_rep2_20260614 > work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_s2e12_rep2_20260614/launch_20260614_gpu2.log 2>&1'
```

## Startup Verification

- Checkpoint load confirmed for all replicas from S2 rep4 epoch 12.
- All three replicas reached `Epoch(train) [1][200/5862]` by
  2026-06-14 16:13:52 CST.
- Failure scan patterns checked: `Traceback`, `CUDA OOM`, `out-of-memory`,
  `libpng`, `CRC`, `NoneType`, `ValueError`, `KeyboardInterrupt`,
  `loss: nan`, `loss: inf`, `grad_norm: nan`, `grad_norm: inf`.
- Failure scan result: no matches across the three new S3 workdirs at startup
  acceptance.
- GPU snapshot after startup: GPU0 6567 MiB / 39%, GPU1 5287 MiB / 56%, GPU2
  5873 MiB / 54%.
