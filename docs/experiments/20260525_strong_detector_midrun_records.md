# Strong Detector Mid-Run Records

Date: 2026-05-25

Purpose: preserve the live S0 detector evidence before final cleanup, metric
JSON export, and qualitative figure generation. These numbers are still
closed-set detector baselines only; they are not prompt/VLM evidence.

## Live Status Snapshot

Snapshot time: about 2026-05-25 21:20 server time.

| Detector | Status | Current/Final state | Best mAP so far | Current ETA |
|---|---|---:|---:|---:|
| Oriented R-CNN | completed | epoch 12 final | 0.2561 | done |
| RoI Transformer | running | epoch 10 train active | 0.2402 at epoch 8 | about 18-20 min |
| ReDet | completed | epoch 12 final | 0.1221 at epoch 12 | done |

Active screen sessions:

- RoI Transformer: `geonexus_roi_trans_lr001`
- ReDet: completed; screen session exited after epoch-12 validation.

## Commands And Artifacts

RoI Transformer:

```bash
CUDA_VISIBLE_DEVICES=2 /data1/anaconda3/envs/zwl_mmrotate/bin/python \
  tools/bootstrap_run.py tools/train.py \
  mmrotate_configs/strong_baseline_dota15/roi-trans-le90_r50_fpn_amp-1x_dota15.py \
  --work-dir work_dirs/strong_baseline_dota15/roi_trans_lr001_rerun
```

- Work dir: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/roi_trans_lr001_rerun/`
- Log: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/roi_trans_lr001_rerun/train.log`
- Best checkpoint so far: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/roi_trans_lr001_rerun/epoch_8.pth`

ReDet:

```bash
CUDA_VISIBLE_DEVICES=4 /data1/anaconda3/envs/zwl_mmrotate/bin/python \
  tools/bootstrap_run.py tools/train.py \
  mmrotate_configs/strong_baseline_dota15/redet-le90_re50_refpn_amp-1x_dota15.py \
  --work-dir work_dirs/strong_baseline_dota15/redet_scratch_rerun
```

- Work dir: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/redet_scratch_rerun/`
- Log: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/redet_scratch_rerun/train.log`
- Final checkpoint: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/redet_scratch_rerun/epoch_12.pth`

## mAP Progression

| Epoch | RoI Transformer mAP | RoI Transformer AP50 | ReDet mAP | ReDet AP50 |
|---:|---:|---:|---:|---:|
| 1 | 0.0901 | 0.0900 | 0.0133 | 0.0130 |
| 2 | 0.1362 | 0.1360 | 0.0244 | 0.0240 |
| 3 | 0.1709 | 0.1710 | 0.0509 | 0.0510 |
| 4 | 0.1899 | 0.1900 | 0.0745 | 0.0750 |
| 5 | 0.2075 | 0.2070 | 0.0852 | 0.0850 |
| 6 | 0.2088 | 0.2090 | 0.0931 | 0.0930 |
| 7 | 0.2240 | 0.2240 | 0.1063 | 0.1060 |
| 8 | 0.2402 | 0.2400 | 0.1110 | 0.1110 |
| 9 | 0.2391 | 0.2390 | 0.1169 | 0.1170 |
| 10 | pending | pending | 0.1199 | 0.1200 |
| 11 | pending | pending | 0.1198 | 0.1200 |
| 12 | pending | pending | 0.1221 | 0.1220 |

## Current Class-Wise Snapshot

RoI Transformer, epoch 9 validation:

| Class | Recall | AP |
|---|---:|---:|
| plane | 0.328 | 0.343 |
| baseball-diamond | 0.315 | 0.326 |
| bridge | 0.045 | 0.073 |
| ground-track-field | 0.441 | 0.367 |
| small-vehicle | 0.029 | 0.091 |
| large-vehicle | 0.381 | 0.342 |
| ship | 0.059 | 0.091 |
| tennis-court | 0.758 | 0.725 |
| basketball-court | 0.217 | 0.217 |
| storage-tank | 0.027 | 0.091 |
| soccer-ball-field | 0.349 | 0.298 |
| roundabout | 0.081 | 0.091 |
| harbor | 0.409 | 0.391 |
| swimming-pool | 0.125 | 0.153 |
| helicopter | 0.244 | 0.227 |
| container-crane | 0.000 | 0.000 |
| mAP |  | 0.239 |

ReDet, epoch 12 final validation:

| Class | Recall | AP |
|---|---:|---:|
| plane | 0.207 | 0.144 |
| baseball-diamond | 0.239 | 0.174 |
| bridge | 0.000 | 0.000 |
| ground-track-field | 0.097 | 0.030 |
| small-vehicle | 0.019 | 0.091 |
| large-vehicle | 0.331 | 0.294 |
| ship | 0.059 | 0.091 |
| tennis-court | 0.626 | 0.564 |
| basketball-court | 0.119 | 0.007 |
| storage-tank | 0.010 | 0.091 |
| soccer-ball-field | 0.215 | 0.102 |
| roundabout | 0.000 | 0.000 |
| harbor | 0.264 | 0.183 |
| swimming-pool | 0.049 | 0.091 |
| helicopter | 0.013 | 0.091 |
| container-crane | 0.000 | 0.000 |
| mAP |  | 0.122 |

## Figures And Tables To Generate

Keep generated figures under `docs/experiments/figures/` and small metric
tables under `docs/experiments/`. Do not commit checkpoints or full work dirs.

Required mid-result artifacts:

| Artifact | Source | Output |
|---|---|---|
| Detector mAP curve | train logs and `vis_data/scalars.json` | `figures/20260525_s0_map_curve.png` |
| Detector AP50 curve | train logs and `vis_data/scalars.json` | `figures/20260525_s0_ap50_curve.png` |
| Final detector comparison table | Oriented R-CNN, RoI Transformer, ReDet validation metrics | `20260525_s0_detector_comparison.md` |
| Class-wise AP heatmap/table | final/best class AP blocks | `figures/20260525_s0_class_ap_heatmap.png` and markdown table |
| Qualitative detections | best checkpoint for each detector with `tools/test.py --show-dir` | `figures/qualitative/<detector>/` |
| Failure examples | low-AP classes: small-vehicle, ship, bridge, container-crane | selected image grid after qualitative export |

Immediate next actions after RoI Transformer finishes:

1. Parse final RoI Transformer epoch 10-12 mAP and class-wise AP.
2. Decide RoI best checkpoint by validation mAP.
3. Run standalone visual export for Oriented R-CNN, RoI Transformer best, and ReDet epoch 12 on a small validation subset or selected images.
4. Save small JSON summaries for ReDet and RoI Transformer, matching the existing Oriented R-CNN metric JSON style.
5. Generate the curve/table figures and update the S0 detector comparison record.
