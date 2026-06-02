# 2026-06-02 S3 Scene Adapter 72e

## Result

- Experiment: GeoNexus S3 hierarchy plus scene adapter, RoI Transformer.
- Dataset/protocol: DOTA v1.5 reduced tiled split, MMRotate DOTAMetric validation.
- Config: `/data5/2025/ldh/OpenRSD/mmrotate_configs/geonexus_dota15/roi-trans-le90_r50_fpn_remoteclip-s3-72e_dota15.py`.
- Work dir: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_scene_adapter_72e`.
- Final checkpoint: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_scene_adapter_72e/epoch_72.pth`.
- Queue log: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_scene_adapter_72e/queue_launch_retry_20260601.log`.
- Metric source: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_scene_adapter_72e/20260601_095547/vis_data/scalars.json`.
- Metric summary: `docs/experiments/20260602_s3_scene_adapter_72e_metrics.json`.

Final epoch-72 validation:

- `dota/mAP=0.3759`
- `dota/AP50=0.3760`

Best observed validation in this 72e run:

- epoch 51: `dota/mAP=0.3800`, `dota/AP50=0.3800`

## Comparison

| Run | Best epoch | Best mAP / AP50 | Final mAP / AP50 |
| --- | ---: | ---: | ---: |
| S2 hierarchy regularizer 72e | 56 | 0.3757 / 0.3760 | 0.3738 / 0.3740 |
| S3 scene adapter 72e | 51 | 0.3800 / 0.3800 | 0.3759 / 0.3760 |

S3 72e is completed evidence and is slightly ahead of the S2 72e record on both
best observed validation mAP and final validation mAP. Keep claims cautious
until the longer S3 144e continuation and other active follow-up runs complete.

## Queue Follow-Up

- `geonexus_s3_scene_adapter_144e` launched automatically on GPU 6 at `2026-06-02 00:37:05`.
- Active screen observed: `425331.geonexus_s3_scene_adapter_144e`.
- Training log: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_scene_adapter_144e/queue_launch_20260531.log`.
- The 144e run loaded from `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_scene_adapter_72e/epoch_72.pth` and produced normal MMEngine training output.
