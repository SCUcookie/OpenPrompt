# GeoNexus S3 Identity Rerun From S2 Epoch 4 - 2026-06-05

GeoNexus S3 reran for 12 epochs from the S2 rerun epoch-4 checkpoint with the scene adapter identity-initialized and residual scale reduced to `0.1`.

| Item | Value |
| --- | --- |
| Base checkpoint | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_rerun_s1e32_20260604/epoch_4.pth` |
| Work dir | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_scene_adapter_identity_rerun_s2e4_20260605` |
| Config | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_scene_adapter_identity_rerun_s2e4_20260605/roi-trans-le90_r50_fpn_remoteclip-s3-identity-rerun-s2e4-20260605_dota15.py` |
| Log | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_scene_adapter_identity_rerun_s2e4_20260605/launch_20260605_gpu1.log` |

## Metrics

| Epoch | DOTAMetric mAP | AP50 |
| --- | ---: | ---: |
| 1 | 0.3618 | 0.3620 |
| 2 | 0.3729 | 0.3730 |
| 3 | 0.3727 | 0.3730 |
| 4 | 0.3766 | 0.3770 |
| 5 | 0.3671 | 0.3670 |
| 6 | 0.3663 | 0.3660 |
| 7 | 0.3736 | 0.3740 |
| 8 | 0.3789 | 0.3790 |
| 9 | 0.3806 | 0.3810 |
| 10 | 0.3794 | 0.3790 |
| 11 | 0.3793 | 0.3790 |
| 12 | 0.3792 | 0.3790 |

Best observed epoch: 9, `dota/mAP=0.3806`, `dota/AP50=0.3810`.

Final epoch: 12, `dota/mAP=0.3792`, `dota/AP50=0.3790`.

## Interpretation

The identity repair did not recover the prior S3 rerun best `0.3827/0.3830` and remains below the S2 rerun best `0.3858/0.3860`.

Run one final S3 adapter-off diagnostic from S2 epoch 4. If adapter-off does not recover at least the prior S3 rerun best, stop S3 and do not launch S4 from S3.
