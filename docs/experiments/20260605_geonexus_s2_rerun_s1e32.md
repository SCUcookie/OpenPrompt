# GeoNexus S2 Rerun From S1 Epoch 32 - 2026-06-05

GeoNexus S2 hierarchy regularizer reran for 12 epochs from the completed S1 retry 2 epoch-32 checkpoint.

| Item | Value |
| --- | --- |
| Base checkpoint | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_rerun_20260603/epoch_32.pth` |
| Work dir | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_rerun_s1e32_20260604` |
| Config | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_rerun_s1e32_20260604/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-rerun-s1e32-20260604_dota15.py` |
| Log | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_rerun_s1e32_20260604/20260604_184151/20260604_184151.log` |
| Scalar source | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_rerun_s1e32_20260604/20260604_184151/vis_data/scalars.json` |

## Metrics

| Epoch | DOTAMetric mAP | AP50 | Notes |
| --- | ---: | ---: | --- |
| 4 | 0.3858 | 0.3860 | Best observed validation; scalar `dota/mAP=0.3858095109462738`, `dota/AP50=0.386` |
| 12 | 0.3784 | 0.3780 | Final validation; scalar `dota/mAP=0.378351092338562`, `dota/AP50=0.378` |

Use `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_rerun_s1e32_20260604/epoch_4.pth` as the S3 initialization checkpoint.
