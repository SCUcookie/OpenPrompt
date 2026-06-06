# GeoNexus S2 Refinement From S2 Epoch 4 LR 1e-4 - 2026-06-06

GeoNexus S2 hierarchy-stabilization refinement completed 12 epochs from the S2 rerun epoch-4 checkpoint with optimizer LR `1e-4`.

| Item | Value |
| --- | --- |
| Base checkpoint | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_rerun_s1e32_20260604/epoch_4.pth` |
| Work dir | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_refine_s2e4_lr1e4_20260606` |
| Config | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_refine_s2e4_lr1e4_20260606/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-refine-s2e4-lr1e4-20260606_dota15.py` |
| Launch log | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_refine_s2e4_lr1e4_20260606/launch_20260606_gpu1.log` |
| Scalar source | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_refine_s2e4_lr1e4_20260606/20260606_095623/vis_data/scalars.json` |

## Metrics

| Epoch | DOTAMetric mAP | AP50 |
| --- | ---: | ---: |
| 1 | 0.3804 | 0.3800 |
| 2 | 0.3764 | 0.3760 |
| 3 | 0.3781 | 0.3780 |
| 4 | 0.3784 | 0.3780 |
| 5 | 0.3801 | 0.3800 |
| 6 | 0.3771 | 0.3770 |
| 7 | 0.3757 | 0.3760 |
| 8 | 0.3746 | 0.3750 |
| 9 | 0.3757 | 0.3760 |
| 10 | 0.3774 | 0.3770 |
| 11 | 0.3765 | 0.3770 |
| 12 | 0.3765 | 0.3760 |

Best observed epoch: 1, `dota/mAP=0.3804`, `dota/AP50=0.3800`.

Final epoch: 12, `dota/mAP=0.3765`, `dota/AP50=0.3760`.

## Decision

The LR `1e-4` refinement did not improve the S2 rerun best epoch 4 `0.3858/0.3860`. Continue with a lower-LR S2 refinement from the same S2 epoch-4 checkpoint before considering any S3/S4 follow-up.
