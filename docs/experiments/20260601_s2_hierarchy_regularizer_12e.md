# 2026-06-01 S2 Hierarchy Regularizer 12e

## Result

- Experiment: GeoNexus S2 hierarchy regularizer, frozen-backbone RoI Transformer path.
- Dataset/protocol: DOTA v1.5 reduced tiled split, MMRotate DOTAMetric validation.
- Config: `/data5/2025/ldh/OpenRSD/mmrotate_configs/geonexus_dota15/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-12e_dota15.py`.
- Work dir: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_frozen_12e`.
- Final checkpoint: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_frozen_12e/epoch_12.pth`.
- Final log: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_frozen_12e/20260531_180850/20260531_180850.log`.

Final epoch-12 validation:

- `dota/mAP=0.3644`
- `dota/AP50=0.3640`

Best observed validation in this 12e run:

- epoch 11: `dota/mAP=0.3652`, `dota/AP50=0.3650`

## Interpretation

This is the first clearly positive hierarchy-regularizer result. Compared with
the recorded RoI Transformer S0 baseline (`mAP=0.2644`) and S1 frozen-backbone
prompt result (`mAP=0.2666`), the final S2 hierarchy-regularizer checkpoint is
about `+0.1000` mAP over S0 and `+0.0978` mAP over S1 on the same reduced tiled
DOTA v1.5 protocol.

Treat this as paper-facing evidence, but not yet final claim evidence until the
already-launched 72e continuation and at least one repeat/secondary-detector
check are inspected.

## Follow-Up

- Monitor `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_frozen_72e`.
- The original S3 72e queue launch failed before training because the inherited
  S3 base config nested `roi_head.bbox_head` incorrectly. Relaunch S3 from a
  corrected child config before interpreting scene-adapter results.
