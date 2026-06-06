# GeoNexus S3 Rerun From S2 Epoch 4 - 2026-06-05

GeoNexus S3 scene adapter reran for 12 epochs from the S2 rerun epoch-4 checkpoint.

| Item | Value |
| --- | --- |
| Base checkpoint | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_rerun_s1e32_20260604/epoch_4.pth` |
| Work dir | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_scene_adapter_rerun_s2e4_20260605` |
| Config | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_scene_adapter_rerun_s2e4_20260605/roi-trans-le90_r50_fpn_remoteclip-s3-rerun-s2e4-20260605_dota15.py` |
| Log | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_scene_adapter_rerun_s2e4_20260605/launch_20260605_gpu1.log` |

## Metrics

| Epoch | DOTAMetric mAP | AP50 |
| --- | ---: | ---: |
| 1 | 0.3704 | 0.3700 |
| 2 | 0.3827 | 0.3830 |
| 3 | 0.3675 | 0.3670 |
| 4 | 0.3662 | 0.3660 |
| 5 | 0.3794 | 0.3790 |
| 6 | 0.3709 | 0.3710 |
| 7 | 0.3727 | 0.3730 |
| 8 | 0.3718 | 0.3720 |
| 9 | 0.3759 | 0.3760 |
| 10 | 0.3788 | 0.3790 |
| 11 | 0.3754 | 0.3750 |
| 12 | 0.3756 | 0.3760 |

Best observed epoch: 2, `dota/mAP=0.3827`, `dota/AP50=0.3830`.

Final epoch: 12, `dota/mAP=0.3756`, `dota/AP50=0.3760`.

## Interpretation

Current controlled progression:

| Stage | Best | Final |
| --- | --- | --- |
| S1 retry 2 | epoch 32, 0.3800 / 0.3800 | epoch 36, 0.3793 / 0.3790 |
| S2 rerun | epoch 4, 0.3858 / 0.3860 | epoch 12, 0.3784 / 0.3780 |
| S3 rerun | epoch 2, 0.3827 / 0.3830 | epoch 12, 0.3756 / 0.3760 |

Hierarchy is currently positive, but the scene adapter is not stable enough to support the paper claim. Do not launch S4 from this S3 result. The next diagnostic priority is an S3 repair run from S2 epoch 4 with identity-initialized scene adapter and reduced residual scale.
