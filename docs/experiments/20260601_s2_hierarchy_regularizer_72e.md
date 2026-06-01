# 2026-06-01 S2 Hierarchy Regularizer 72e

## Result

- Experiment: GeoNexus S2 hierarchy regularizer, frozen-backbone RoI Transformer continuation.
- Dataset/protocol: DOTA v1.5 reduced tiled split, MMRotate DOTAMetric validation.
- Config: `/data5/2025/ldh/OpenRSD/mmrotate_configs/geonexus_dota15/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-72e_dota15.py`.
- Base run loaded from: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_frozen_12e/epoch_12.pth`.
- Work dir: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_frozen_72e`.
- Final checkpoint: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_frozen_72e/epoch_72.pth`.
- Training log: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_frozen_72e/20260531_205243/20260531_205243.log`.
- Metric source: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_frozen_72e/20260531_205243/vis_data/scalars.json`.

Final epoch-72 validation:

- `dota/mAP=0.3738`
- `dota/AP50=0.3740`

Best observed validation in this 72e run:

- epoch 56: `dota/mAP=0.3757`, `dota/AP50=0.3760`

## Comparison

| Run | Best epoch | Best mAP / AP50 | Final mAP / AP50 |
| --- | ---: | ---: | ---: |
| S0 RoI Transformer 3x | 34 | 0.2644 / 0.2640 | 0.2612 / 0.2610 |
| S1 frozen-backbone RemoteCLIP prompts | 6 | 0.2666 / 0.2670 | 0.2506 / 0.2510 |
| S2 hierarchy regularizer 12e | 11 | 0.3652 / 0.3650 | 0.3644 / 0.3640 |
| S2 hierarchy regularizer 72e | 56 | 0.3757 / 0.3760 | 0.3738 / 0.3740 |

The 72e continuation improves the best S2 12e result by about `+0.0105` mAP and
the final S2 12e result by about `+0.0094` mAP. Against the primary S0 RoI
Transformer baseline, the 72e best checkpoint is about `+0.1113` mAP. Against
the strongest S1 frozen-backbone prompt result, it is about `+0.1091` mAP.

Treat this as stronger S2 evidence than the 12e run. The longer 144e S2 run and
the active S3 scene-adapter run should still finish before final paper wording
claims convergence or context-adapter effects.

## Active Follow-Up At Archive Time

- S3 scene adapter 72e was active in screen `3845911.geonexus_s3_scene_adapter_72e` on GPU 1, PID `3846084`.
- S2 hierarchy regularizer 144e was active in screen `3891792.geonexus_s2_hierarchy_reg_144e` on GPU 2, PID `3891957`.
- S3 scene adapter 144e remained pending in `New/queues/geonexus_gpu_queue_20260531.json`, waiting for the S3 72e checkpoint.

