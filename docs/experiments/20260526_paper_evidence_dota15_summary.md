# DOTA v1.5 Paper Evidence Summary - 2026-05-26

This record separates closed-set strong-detector evidence from the GeoNexus RemoteCLIP scaffold diagnostics. Large prediction dumps, figures, and logs remain under `OpenRSD/work_dirs`; compact metrics stay in this docs directory.

## Main Comparison

| Detector | Checkpoint | Best epoch | mAP / AP50 | Prediction dump | Confusion matrix | Qualitative examples | Benchmark |
|---|---:|---:|---:|---|---|---|---|
| RoI Transformer | `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/roi_trans_lr001_3x/epoch_34.pth` | 34 | 0.2644 / 0.264 | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/roi_trans_epoch34/preds.pkl` | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/roi_trans_epoch34/confusion/confusion_matrix.png` | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/roi_trans_epoch34/qualitative/` | Failed: legacy `pretrained` config key |
| Oriented R-CNN | `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/oriented_rcnn_3x_loadfrom/epoch_33.pth` | 33 | 0.2620 / 0.262 | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/oriented_rcnn_epoch33/preds.pkl` | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/oriented_rcnn_epoch33/confusion/confusion_matrix.png` | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/oriented_rcnn_epoch33/qualitative/` | Failed: legacy `pretrained` config key |
| ReDet pretrained | `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/redet_pretrained_rerun/epoch_12.pth` | 12 | 0.2382 / 0.238 | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/redet_pretrained_epoch12/preds.pkl` | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/redet_pretrained_epoch12/confusion/confusion_matrix.png` | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/redet_pretrained_epoch12/qualitative/` | Failed: legacy `pretrained` config key |

Raw MMRotate test output JSONs:

| Detector | Raw JSON | mAP | AP50 | mean iter time |
|---|---|---:|---:|---:|
| RoI Transformer | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/roi_trans_epoch34/20260526_104338/20260526_104338.json` | 0.2644376755 | 0.264 | 0.2256 s |
| Oriented R-CNN | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/oriented_rcnn_epoch33/20260526_104337/20260526_104337.json` | 0.2619543076 | 0.262 | 0.1426 s |
| ReDet pretrained | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/redet_pretrained_epoch12/20260526_104338/20260526_104338.json` | 0.2382197678 | 0.238 | 0.1423 s |

## Per-Epoch AP50 Curves

Values are copied from the tracked S0 metric JSONs:

| Epoch | RoI Transformer | Oriented R-CNN | ReDet pretrained |
|---:|---:|---:|---:|
| 1 | 0.243 | 0.237 | 0.101 |
| 2 | 0.249 | 0.240 | 0.145 |
| 3 | 0.244 | 0.236 | 0.176 |
| 4 | 0.245 | 0.248 | 0.191 |
| 5 | 0.241 | 0.238 | 0.201 |
| 6 | 0.244 | 0.241 | 0.213 |
| 7 | 0.253 | 0.259 | 0.225 |
| 8 | 0.248 | 0.256 | 0.223 |
| 9 | 0.249 | 0.257 | 0.230 |
| 10 | 0.243 | 0.235 | 0.230 |
| 11 | 0.250 | 0.258 | 0.236 |
| 12 | 0.249 | 0.255 | 0.238 |
| 13 | 0.255 | 0.253 | - |
| 14 | 0.250 | 0.242 | - |
| 15 | 0.252 | 0.259 | - |
| 16 | 0.247 | 0.249 | - |
| 17 | 0.250 | 0.249 | - |
| 18 | 0.244 | 0.258 | - |
| 19 | 0.254 | 0.249 | - |
| 20 | 0.256 | 0.248 | - |
| 21 | 0.254 | 0.260 | - |
| 22 | 0.263 | 0.259 | - |
| 23 | 0.253 | 0.255 | - |
| 24 | 0.249 | 0.252 | - |
| 25 | 0.260 | 0.262 | - |
| 26 | 0.262 | 0.261 | - |
| 27 | 0.262 | 0.258 | - |
| 28 | 0.260 | 0.258 | - |
| 29 | 0.257 | 0.258 | - |
| 30 | 0.260 | 0.260 | - |
| 31 | 0.258 | 0.257 | - |
| 32 | 0.261 | 0.258 | - |
| 33 | 0.261 | 0.262 | - |
| 34 | 0.264 | 0.262 | - |
| 35 | 0.261 | 0.259 | - |
| 36 | 0.261 | 0.261 | - |

## Qualitative Artifacts

Each detector has 5 `good` and 5 `bad` rendered examples. Per-image ranking used mean detection confidence as a fallback because MMDet's per-image HBB mAP helper cannot compare 5-parameter rotated boxes directly.

## Efficiency Notes

The planned `tools/analysis_tools/benchmark.py` runs failed for all three detectors with `TypeError: __init__() got an unexpected keyword argument 'pretrained'`. This is an OpenMMLab API compatibility issue in the benchmark path, not a failure of the detector checkpoints or prediction dumps. The MMRotate test logs still record per-iteration wall time in the raw JSON table above, but those are not a replacement for controlled benchmark numbers.

## GeoNexus Scaffold Separation

The active `dota_v15_geonexus_remoteclip` screen is a separate scaffold diagnostic run on GPU 2. It should not be reported as paper-facing S1 evidence until RemoteCLIP prompt logic is integrated into the selected MMRotate detector.
