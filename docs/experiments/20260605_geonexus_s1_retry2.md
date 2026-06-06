# GeoNexus S1 Retry 2 Metrics - 2026-06-05

GeoNexus S1 RemoteCLIP prompt-head rerun retry 2 completed 36 epochs on the controlled DOTA v1.5 reduced tiled split.

| Item | Value |
| --- | --- |
| Work dir | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_rerun_20260603` |
| Config | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_rerun_20260603/roi-trans-le90_r50_fpn_remoteclip-s1-rerun-20260603_dota15.py` |
| Log | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_rerun_20260603/20260604_100546/20260604_100546.log` |
| Scalar source | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_rerun_20260603/20260604_100546/vis_data/scalars.json` |

## Metrics

| Epoch | DOTAMetric mAP | AP50 | Notes |
| --- | ---: | ---: | --- |
| 32 | 0.3800 | 0.3800 | Best observed validation; scalar `dota/mAP=0.37997525930404663`, `dota/AP50=0.38` |
| 36 | 0.3793 | 0.3790 | Final validation; scalar `dota/mAP=0.37929123640060425`, `dota/AP50=0.379` |

Use `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_rerun_20260603/epoch_32.pth` as the S2 initialization checkpoint.
