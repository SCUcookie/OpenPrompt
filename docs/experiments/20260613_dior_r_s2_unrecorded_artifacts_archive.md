# 2026-06-13 DIOR-R S2 Unrecorded Artifacts Archive

## Scope

This archive records the DIOR-R S2 hierarchy-regularizer state that existed outside `New/` after the launch record was written, but had not yet been consolidated in a `New/docs/experiments` archive note.

Already-recorded context remains in:

- `/data5/2025/ldh/New/docs/experiments/20260613_dior_r_geonexus_s1_s0e52_replicas_launch.md`
- `/data5/2025/ldh/New/docs/experiments/20260613_dior_r_geonexus_s1_s0e52_replicas_metrics.json`
- `/data5/2025/ldh/New/docs/experiments/20260613_dior_r_geonexus_s2_hierarchy_replicas_launch.md`
- `/data5/2025/ldh/New/docs/experiments/20260613_dior_r_s0_retinanet_sanitized_long_complete.md`

## Newly Archived Outside-New State

Snapshot time: `2026-06-13 15:51:12 CST`.

Active screens:

- `dior_r_geonexus_s2_s1e12_rep0_20260613_gpu0`
- `dior_r_geonexus_s2_s1e12_rep1_20260613_gpu1`
- `dior_r_geonexus_s2_s1e12_rep2_20260613_gpu2`
- Existing monitor: `s0_result_log_monitor_20260603`

GPU status at archive time:

- GPU 0: S2 rep0 active, about `8849 MiB`, `36%` utilization in the latest poll before this archive.
- GPU 1: S2 rep1 active, about `8103 MiB`, `40%` utilization in the latest poll before this archive.
- GPU 2: S2 rep2 active, about `9011 MiB`, `37%` utilization in the latest poll before this archive.
- GPU 4: unrelated activity remained active, about `19053 MiB`, `68%` utilization; it was not disturbed.

## Run Artifact Inventory

Replica 0:

- Workdir: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep0_20260613`
- Config: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep0_20260613/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-s1e12-rep0-20260613_dior_r.py`
- Launch log: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep0_20260613/launch_20260613_gpu0.log`
- Runtime log: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep0_20260613/20260613_154141/20260613_154141.log`
- Latest archived progress: epoch 1, iteration `2900/5862` at `2026/06/13 15:51:06`; loss `0.4504`, grad norm `3.9133`, `s0.loss_hierarchy=0.0068`, `s1.loss_hierarchy=0.0074`.

Replica 1:

- Workdir: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep1_20260613`
- Config: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep1_20260613/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-s1e12-rep1-20260613_dior_r.py`
- Launch log: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep1_20260613/launch_20260613_gpu1.log`
- Runtime log: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep1_20260613/20260613_154141/20260613_154141.log`
- Latest archived progress: epoch 1, iteration `2900/5862` at `2026/06/13 15:51:10`; loss `0.4777`, grad norm `4.0088`, `s0.loss_hierarchy=0.0061`, `s1.loss_hierarchy=0.0073`.

Replica 2:

- Workdir: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep2_20260613`
- Config: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep2_20260613/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-s1e12-rep2-20260613_dior_r.py`
- Launch log: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep2_20260613/launch_20260613_gpu2.log`
- Runtime log: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep2_20260613/20260613_154141/20260613_154141.log`
- Latest archived progress: epoch 1, iteration `2900/5862` at `2026/06/13 15:51:12`; loss `0.5317`, grad norm `4.4102`, `s0.loss_hierarchy=0.0076`, `s1.loss_hierarchy=0.0079`.

## Failure Scan

Scoped scan over all three S2 workdirs found no matches for:

`Traceback`, CUDA OOM/out-of-memory, `libpng`, `CRC`, `NoneType`, `ValueError`, `KeyboardInterrupt`, `loss: nan/inf`, or `grad_norm: nan/inf`.

## Archive Status

The unrecorded operational state has now been written into `New/`. The S2 jobs remain active and should receive a separate completion archive after epoch 12 validation finishes.
