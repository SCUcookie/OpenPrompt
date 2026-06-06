# GeoNexus S3 Adapter-Off Rerun From S2 Epoch 4 - 2026-06-06

GeoNexus S3 adapter-off diagnostic completed 12 epochs from the S2 rerun epoch-4 checkpoint.

| Item | Value |
| --- | --- |
| Base checkpoint | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_rerun_s1e32_20260604/epoch_4.pth` |
| Work dir | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_adapter_off_rerun_s2e4_20260605` |
| Config | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_adapter_off_rerun_s2e4_20260605/roi-trans-le90_r50_fpn_remoteclip-s3-adapter-off-rerun-s2e4-20260605_dota15.py` |
| Log | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_adapter_off_rerun_s2e4_20260605/launch_20260605_gpu3.log` |

## Metrics

| Epoch | DOTAMetric mAP | AP50 |
| --- | ---: | ---: |
| 1 | 0.3650 | 0.3650 |
| 2 | 0.3665 | 0.3660 |
| 3 | 0.3772 | 0.3770 |
| 4 | 0.3640 | 0.3640 |
| 5 | 0.3722 | 0.3720 |
| 6 | 0.3745 | 0.3750 |
| 7 | 0.3719 | 0.3720 |
| 8 | 0.3639 | 0.3640 |
| 9 | 0.3706 | 0.3710 |
| 10 | 0.3730 | 0.3730 |
| 11 | 0.3758 | 0.3760 |
| 12 | 0.3758 | 0.3760 |

Best observed epoch: 3, `dota/mAP=0.3772`, `dota/AP50=0.3770`.

Final epoch: 12, `dota/mAP=0.3758`, `dota/AP50=0.3760`.

## Decision

The adapter-off run stayed below the stop threshold `0.3827`, below the prior S3 rerun best `0.3827/0.3830`, and below the S2 rerun best `0.3858/0.3860`.

Stop S3 diagnostics and do not launch S4 from S3. The next paper-path diagnostic is S2 stabilization/refinement from S2 epoch 4, not S4.
