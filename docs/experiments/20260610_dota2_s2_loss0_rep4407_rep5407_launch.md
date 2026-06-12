# DOTA2 S2 Loss-0 Rep4407/Rep5407 Launch - 2026-06-10

This note records two controlled DOTA2 S2 hierarchy loss-0 replications
launched to maintain three concurrent GeoNexus GPU experiments.

## Purpose

Keep the DOTA2 S2 loss-0 signal under controlled replication while `rep3407`
continues on GPU 1. Do not use these runs to open S3/S4, FAIR1M,
pseudo-labeling, or DIOR-R detector training gates until best and final
checkpoints are compared separately against DOTA2 S1 `0.6177 / 0.6180`.

## Configs

Both configs were copied from:

`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-ablate-loss0-s1e12-rep3407-20260610_dota2.py`

Expected diff only:

- `randomness = dict(deterministic=False, seed=4407)` and the `rep4407`
  workdir/name.
- `randomness = dict(deterministic=False, seed=5407)` and the `rep5407`
  workdir/name.

Controlled settings retained:

| Setting | Value |
| --- | --- |
| `load_from` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607/epoch_12.pth` |
| `hierarchy_loss_weight` | `0.0` in both cascade heads |
| `learnable_prompt_offsets` | `True` in both cascade heads |
| Optimizer LR | `5e-5` |
| `max_epochs` | `4` |
| `val_interval` | `1` |

## Launch

Preflight GPU checks on `2026-06-10 20:58:57`, `20:59:17`, and `20:59:37`
CST showed GPUs 0 and 2 at `14 MiB` and `0%` utilization. GPU 3 was occupied
by another user and was not used.

### Rep4407

Workdir:

`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep4407_20260610`

Config:

`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep4407_20260610/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-ablate-loss0-s1e12-rep4407-20260610_dota2.py`

Screen:

`geonexus_dota2_s2_loss0_rep4407_20260610_gpu0`

Runtime log:

`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep4407_20260610/20260610_210021/20260610_210021.log`

Launch command:

```bash
cd /data5/2025/ldh/OpenRSD
CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib_dota2_s2_loss0_rep4407 PYTHONNOUSERSITE=1 \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py \
  work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep4407_20260610/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-ablate-loss0-s1e12-rep4407-20260610_dota2.py \
  --work-dir work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep4407_20260610
```

### Rep5407

Workdir:

`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep5407_20260610`

Config:

`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep5407_20260610/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-ablate-loss0-s1e12-rep5407-20260610_dota2.py`

Screen:

`geonexus_dota2_s2_loss0_rep5407_20260610_gpu2`

Runtime log:

`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep5407_20260610/20260610_210021/20260610_210021.log`

Launch command:

```bash
cd /data5/2025/ldh/OpenRSD
CUDA_VISIBLE_DEVICES=2 MPLCONFIGDIR=/tmp/matplotlib_dota2_s2_loss0_rep5407 PYTHONNOUSERSITE=1 \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py \
  work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep5407_20260610/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-ablate-loss0-s1e12-rep5407-20260610_dota2.py \
  --work-dir work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep5407_20260610
```

## Startup Status

At `2026-06-10 21:00:51 CST`, `nvidia-smi` showed:

| Run | GPU | PID | GPU memory |
| --- | --- | --- | --- |
| `rep4407` | 0 | `1610248` | `1920 MiB` |
| `rep3407` | 1 | `1559651` | `19810 MiB` |
| `rep5407` | 2 | `1610251` | `1920 MiB` |

At `2026-06-10 21:05 CST`, both new processes remained GPU-resident and their
screens remained detached. The logs had reached MMEngine hook setup but had not
yet emitted `Epoch(train) [1][  200/39007]`. Host `ps` showed both new Python
processes in `wait_on_page_bit_common`, consistent with startup I/O wait.

Scoped scans through the current log content found no `Traceback`, CUDA OOM,
`libpng`, `CRC`, `NoneType`, `ValueError`, `KeyboardInterrupt`, or true
`nan`/`inf` failure signature. Startup acceptance remains pending until the
`Epoch(train) [1][  200/39007]` line appears and the failure scan remains clean.

## Metric Status

Completed. Validation metrics are from each run's `vis_data/scalars.json`.

### Rep4407 Metrics

Scalars:

`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep4407_20260610/20260610_210021/vis_data/scalars.json`

| Epoch | `dota/mAP` | `dota/AP50` | S1 comparison |
| --- | --- | --- | --- |
| 1 | `0.615142` | `0.6150` | below S1 |
| 2 | `0.615141` | `0.6150` | below S1 |
| 3 | `0.620637` | `0.6210` | above S1 |
| 4 | `0.614786` | `0.6150` | below S1 |

Best checkpoint: epoch 3 `0.620637 / 0.6210`.

Final checkpoint: epoch 4 `0.614786 / 0.6150`, unstable final.

### Rep5407 Metrics

Scalars:

`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep5407_20260610/20260610_210021/vis_data/scalars.json`

| Epoch | `dota/mAP` | `dota/AP50` | S1 comparison |
| --- | --- | --- | --- |
| 1 | `0.621215` | `0.6210` | above S1 |
| 2 | `0.620206` | `0.6200` | above S1 |
| 3 | `0.616239` | `0.6160` | below S1 |
| 4 | `0.614990` | `0.6150` | below S1 |

Best checkpoint: epoch 1 `0.621215 / 0.6210`.

Final checkpoint: epoch 4 `0.614990 / 0.6150`, unstable final.

See:

`/data5/2025/ldh/New/docs/experiments/20260611_dota2_s2_loss0_replicates_analysis.md`
