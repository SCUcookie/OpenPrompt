# 2026-06-02 OpenRSD DOTA2 Epoch-12 ss_val Evaluation

## Result

- Experiment: OpenRSD DOTA2 epoch-12 checkpoint evaluation on `DOTA2_1024_500/ss_val`.
- Dataset/protocol: DOTA2 tiled `ss_val`, `DETAILDOTAMetric`, IoU 0.5.
- Config: `/data5/2025/ldh/OpenRSD/work_dirs/opensrd_formal_dota2_ssval_eval/a10_formal_dota2_eval_no_star.py`.
- Checkpoint: `/data5/2025/ldh/OpenRSD/work_dirs/opensrd_step2_dota2_nozero_full_20260531/epoch_12.pth`.
- Work dir: `/data5/2025/ldh/OpenRSD/work_dirs/opensrd_dota2_epoch12_ssval_eval_20260602`.
- Predictions: `/data5/2025/ldh/OpenRSD/work_dirs/opensrd_dota2_epoch12_ssval_eval_20260602/preds.pkl`.
- Launch log: `/data5/2025/ldh/OpenRSD/work_dirs/opensrd_dota2_epoch12_ssval_eval_20260602/launch_20260602.log`.
- Metric summary: `docs/experiments/20260602_opensrd_dota2_epoch12_ssval_metrics.json`.

Final evaluation:

- `dota/mAP=0.4202`
- `dota/AP50=0.4200`

## Class AP

| Class | AP | Recall | GTs | Detections |
| --- | ---: | ---: | ---: | ---: |
| airport | 0.1482 | 0.6535 | 101 | 38417 |
| baseball-diamond | 0.4202 | 0.7397 | 653 | 47493 |
| basketball-court | 0.5519 | 0.7540 | 439 | 56720 |
| bridge | 0.3228 | 0.6371 | 1466 | 71237 |
| container-crane | 0.0002 | 0.1972 | 71 | 15612 |
| ground-track-field | 0.3922 | 0.8816 | 397 | 63203 |
| harbor | 0.6230 | 0.7850 | 6381 | 88318 |
| helicopter | 0.0579 | 0.5704 | 284 | 18001 |
| helipad | 0.0007 | 0.3333 | 6 | 8756 |
| large-vehicle | 0.6163 | 0.8171 | 14169 | 319872 |
| plane | 0.7961 | 0.8828 | 8718 | 60345 |
| roundabout | 0.5133 | 0.7306 | 620 | 52872 |
| ship | 0.7783 | 0.8683 | 45001 | 195139 |
| small-vehicle | 0.2874 | 0.4405 | 150145 | 706574 |
| soccer-ball-field | 0.3412 | 0.6023 | 430 | 81109 |
| storage-tank | 0.5492 | 0.6549 | 10557 | 202228 |
| swimming-pool | 0.3230 | 0.5921 | 2324 | 90248 |
| tennis-court | 0.8425 | 0.9385 | 1870 | 81742 |

## Comparison

The prior official DOTA2 `ss_val` evaluator result was `dota/mAP=0.6510` and
`dota/AP50=0.6510`. This epoch-12 OpenRSD checkpoint is lower by `0.2308` mAP
and `0.2310` AP50.

Keep the claim narrow: this run makes the completed OpenRSD DOTA2 training gate
measurable on `ss_val`, but it is not GeoNexus S2/S3 evidence.
