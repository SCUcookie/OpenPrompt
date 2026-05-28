# Experiment: S0 and OpenRSD Reproduction Status

Date: 2026-05-28

Status: active / protocol correction required

Primary workspaces:

- OpenRSD reference workspace: `/data5/2025/ldh/OpenRSD`
- Project workspace: `/data5/2025/ldh/New`

## Purpose

Restart the S0 detector baseline work and prepare a valid OpenRSD reproduction path. The immediate goal was to understand why local RoI Transformer, Oriented R-CNN, and ReDet-style S0 results were far below paper-reported DOTA numbers, then separate usable diagnostics from invalid paper comparisons.

## What Was Run

### Completed Oriented R-CNN Runs

All of these used the local `/data5/2025/ldh/OpenPrompt/DOTA` DOTA tree with `1411` train images and `458` validation images.

| Run | Work dir | Status | Best checkpoint | Best mAP/AP50 | Final mAP/AP50 |
| --- | --- | --- | --- | --- | --- |
| S0 full DOTA Oriented R-CNN 3x | `/data5/2025/ldh/OpenRSD/work_dirs/s0_full_dota15_oriented_rcnn_3x` | complete | `epoch_36.pth` | `0.2859 / 0.2860` | `0.2859 / 0.2860` |
| A1 repeat seed 3407 | `/data5/2025/ldh/OpenRSD/work_dirs/route_a1_repeat_seed3407` | complete | `epoch_31.pth` | `0.2895 / 0.2900` | `0.2843 / 0.2840` |
| A1 repeat seed 9281 | `/data5/2025/ldh/OpenRSD/work_dirs/route_a1_repeat_seed9281` | complete | `epoch_33.pth` | `0.2918 / 0.2920` | `0.2864 / 0.2860` |

Current strongest diagnostic detector checkpoint:

```text
/data5/2025/ldh/OpenRSD/work_dirs/route_a1_repeat_seed9281/epoch_33.pth
```

### RoI Transformer Runs

| Run | Work dir | Status | Best checkpoint | Best mAP/AP50 | Note |
| --- | --- | --- | --- | --- | --- |
| RoITrans fair repeat seed 3407 | `/data5/2025/ldh/OpenRSD/work_dirs/roi_trans_fair_repeat_seed3407` | running at record time | `epoch_32.pth` so far | `0.2605 / 0.2610` | still active in screen `geonexus_roi_seed3407` |
| S0 full DOTA RoITrans 3x | `/data5/2025/ldh/OpenRSD/work_dirs/s0_full_dota15_roi_trans_3x` | failed | n/a | n/a | CUDA OOM during epoch 1 |

The failed RoITrans run should not be relaunched unchanged.

## Why These S0 Numbers Are Low

These runs are not valid paper-protocol S0 comparisons.

Main issue: they used the original-image DOTA tree at `/data5/2025/ldh/OpenPrompt/DOTA`, not official paper-style tiled DOTA patches. The directory contains only:

- train images: `1411`
- val images: `458`
- train annotations: `1411`
- val annotations: `458`

For DOTA detector papers and MMRotate-style S0 reproduction, the standard protocol uses many clipped image patches, typically around `1024 x 1024` with overlap, followed by DOTA-style prediction merging/submission or equivalent tiled validation. Training on huge original images resized to `640` or `1024` heavily damages dense small objects. This is visible in the logs: `small-vehicle` recall is about `0.04`.

Therefore:

- `0.26-0.29` mAP here is a diagnostic result for the current local reduced/unsliced setup.
- It must not be compared against RoI Transformer, Oriented R-CNN, ReDet, or OpenRSD paper numbers.
- It should not be used as the main S0 table in the paper.

## Correct S0/OpenRSD Reproduction Path

The next valid S0 baseline must use the downloaded tiled OpenRSD/DOTA2 folders:

```text
/data5/2025/ldh/OpenRSD/data/DOTA2_1024_500/train
/data5/2025/ldh/OpenRSD/data/DOTA2_1024_500/ss_val
```

Required structure to verify after download:

```text
DOTA2_1024_500/train/images
DOTA2_1024_500/train/annfiles
DOTA2_1024_500/train/Step6_Format_labels
DOTA2_1024_500/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl
DOTA2_1024_500/ss_val/images
DOTA2_1024_500/ss_val/annfiles
```

Required root support files:

```text
Neg_supports_v2.pkl
normalized_class_dict.pkl
7_25_pca_meta_DINOv2_256.pkl
```

First OpenRSD-native S0 configs to run:

```text
/data5/2025/ldh/OpenRSD/M_configs/G02_Baselines/Data1_DOTA2/G02_Baselines_Data1_DOTA2_M2_RoITrans.py
/data5/2025/ldh/OpenRSD/M_configs/G02_Baselines/Data1_DOTA2/G02_Baselines_Data1_DOTA2_M5_ORCNN_R50.py
```

OpenRSD checkpoint/eval target after data is available:

```text
/data5/2025/ldh/OpenRSD/results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24.pth
```

## Data Download Decision

Minimum required to unblock OpenRSD/DOTA2 reproduction:

1. `DOTA2_1024_500/train`
2. `DOTA2_1024_500/ss_val`
3. `MINI_Test_Dataset/Data1_DOTA2`
4. `Neg_supports_v2.pkl`
5. `normalized_class_dict.pkl`
6. `7_25_pca_meta_DINOv2_256.pkl`

Useful cross-dataset OpenRSD eval data after DOTA2 works:

1. `DIOR_R_dota/train_val`
2. `DIOR_R_dota/test`
3. `FAIR1M_2_800_400/train`
4. `FAIR1M_2_800_400/ss_val`
5. `xView_New_800_600/train`
6. `xView_New_800_600/test`
7. `HRSC2016_DOTA/train`
8. `HRSC2016_DOTA/test`
9. `WHU_Mix/train`

Hold `Spacenet_Merge` until its validation folder is confirmed, because only `train` was listed.

## Immediate Next Actions

1. Let `geonexus_roi_seed3407` finish and archive final RoITrans metrics.
2. When `DOTA2_1024_500` finishes downloading, run folder-count checks before launching any OpenRSD jobs.
3. Launch OpenRSD-native S0 RoITrans and ORCNN on the tiled DOTA2 data.
4. Only after S0 is near paper-protocol behavior, compare GeoNexus/OpenPrompt modules against it.

## Related OpenRSD Archive Files

```text
/data5/2025/ldh/OpenRSD/work_dirs/S0_STATUS_ARCHIVE_20260528_1945.md
/data5/2025/ldh/OpenRSD/work_dirs/S0_OFFICIAL_DOTA_PROTOCOL_20260528.md
```
