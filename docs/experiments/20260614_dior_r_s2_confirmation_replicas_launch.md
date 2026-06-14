# DIOR-R GeoNexus S2 Confirmation Replicas Launch

Date: 2026-06-14

## Purpose

Launch three independent DIOR-R GeoNexus S2 hierarchy confirmation replicas from the original S1 source checkpoint, preserving the completed S2 protocol while varying only replica seed and work directory.

## Shared Protocol

- Source checkpoint: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s1_s0e52_rep0_20260613/epoch_12.pth`
- Prompt artifact: `/data5/2025/ldh/New/artifacts/generated/remoteclip_vit_b32_dior_r_s2_hierarchy_prompt_embeddings.pt`
- Template config: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep0_20260613/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-s1e12-rep0-20260613_dior_r.py`
- Dataset root: `data/DIOR_R_dota_sanitized_invalidsize_20260612/`
- Train labels/images: `train_val/labelTxt/`, `train_val/images/`
- Test labels/images: `test/labelTxt/`, `test/images/`
- `max_epochs`: 12
- `val_interval`: 4
- `ckpt_interval`: 4
- `hierarchy_loss_weight`: 0.05

## Preflight

- GPUs 0, 1, and 2 were idle before launch: each reported 14 MiB used and 0% utilization.
- No active `tools/train.py` or `tools/bootstrap_run.py` process was present before launch.
- Source checkpoint and prompt artifact existed.
- New configs were copied from S2 rep0 and diffed against the template; only `randomness.seed` and `work_dir` changed.

## Replicas

| Replica | GPU | Seed | Screen | Workdir | Config | Launch log | Runtime log | Startup status |
| --- | ---: | ---: | --- | --- | --- | --- | --- | --- |
| rep3 | 0 | 10407 | `dior_r_geonexus_s2_s1e12_rep3_20260614_gpu0` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep3_20260614` | `roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-s1e12-rep3-20260614_dior_r.py` | `launch_20260614_gpu0.log` | `20260614_111933/20260614_111933.log` | accepted: reached `Epoch(train) [1][200/5862]`; no failure signatures |
| rep4 | 1 | 11407 | `dior_r_geonexus_s2_s1e12_rep4_20260614_gpu1` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep4_20260614` | `roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-s1e12-rep4-20260614_dior_r.py` | `launch_20260614_gpu1.log` | `20260614_111933/20260614_111933.log` | accepted: reached `Epoch(train) [1][200/5862]`; no failure signatures |
| rep5 | 2 | 12407 | `dior_r_geonexus_s2_s1e12_rep5_20260614_gpu2` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep5_20260614` | `roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-s1e12-rep5-20260614_dior_r.py` | `launch_20260614_gpu2.log` | `20260614_111934/20260614_111934.log` | accepted: reached `Epoch(train) [1][200/5862]`; no failure signatures |

## Launch Commands

```bash
cd /data5/2025/ldh/OpenRSD

screen -dmS dior_r_geonexus_s2_s1e12_rep3_20260614_gpu0 bash -lc 'CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib_dior_r_s2_rep3_20260614 PYTHONNOUSERSITE=1 /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep3_20260614/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-s1e12-rep3-20260614_dior_r.py --work-dir work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep3_20260614 > work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep3_20260614/launch_20260614_gpu0.log 2>&1'

screen -dmS dior_r_geonexus_s2_s1e12_rep4_20260614_gpu1 bash -lc 'CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib_dior_r_s2_rep4_20260614 PYTHONNOUSERSITE=1 /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep4_20260614/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-s1e12-rep4-20260614_dior_r.py --work-dir work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep4_20260614 > work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep4_20260614/launch_20260614_gpu1.log 2>&1'

screen -dmS dior_r_geonexus_s2_s1e12_rep5_20260614_gpu2 bash -lc 'CUDA_VISIBLE_DEVICES=2 MPLCONFIGDIR=/tmp/matplotlib_dior_r_s2_rep5_20260614 PYTHONNOUSERSITE=1 /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep5_20260614/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-s1e12-rep5-20260614_dior_r.py --work-dir work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep5_20260614 > work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep5_20260614/launch_20260614_gpu2.log 2>&1'
```

## Startup Verification

- Checkpoint load confirmed for all replicas from `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s1_s0e52_rep0_20260613/epoch_12.pth`.
- All three replicas reached `Epoch(train) [1][200/5862]` by 2026-06-14 11:20:56 Asia/Shanghai.
- Failure scan patterns checked: `Traceback`, `CUDA OOM`, `out-of-memory`, `libpng`, `CRC`, `NoneType`, `ValueError`, `KeyboardInterrupt`, `loss: nan`, `loss: inf`, `grad_norm: nan`, `grad_norm: inf`.
- Failure scan result: no matches across the three new workdirs at startup acceptance.
- GPU snapshot after launch: GPU0 9477 MiB / 37%, GPU1 7633 MiB / 39%, GPU2 7641 MiB / 42%.

## Follow-up

After completion, archive best/final epoch 4/8/12 metrics in JSON and Markdown, then rerun `/data5/2025/ldh/New/scripts/make_result_assets_20260614.py` to refresh `/data5/2025/ldh/New/artifacts/result_assets_20260614/all_experiment_results_20260614.csv`.
