# OpenRSD Data Minimum for Reproduction

Date: 2026-05-28

## Minimal Download Set

To reproduce the OpenRSD DOTA2 baseline and evaluate our work against it, download:

```text
DOTA2_1024_500/train
DOTA2_1024_500/ss_val
MINI_Test_Dataset/Data1_DOTA2
Neg_supports_v2.pkl
normalized_class_dict.pkl
7_25_pca_meta_DINOv2_256.pkl
```

## Expected DOTA2 Layout

After download, the following paths should exist under `/data5/2025/ldh/OpenRSD/data`:

```text
DOTA2_1024_500/train/images
DOTA2_1024_500/train/annfiles
DOTA2_1024_500/train/Step6_Format_labels
DOTA2_1024_500/train/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl
DOTA2_1024_500/ss_val/images
DOTA2_1024_500/ss_val/annfiles
```

If `train/annfiles` is absent but `train/Step6_Format_labels` exists, the OpenRSD prompt-training configs can still work, but plain detector baselines may need their `train_ann_file` adjusted.

## First Commands After Download

Run these checks before launching training:

```bash
find -L /data5/2025/ldh/OpenRSD/data/DOTA2_1024_500 -maxdepth 2 -type d | sort
find -L /data5/2025/ldh/OpenRSD/data/DOTA2_1024_500/train/images -maxdepth 1 -type f | wc -l
find -L /data5/2025/ldh/OpenRSD/data/DOTA2_1024_500/train/annfiles -maxdepth 1 -type f | wc -l
find -L /data5/2025/ldh/OpenRSD/data/DOTA2_1024_500/train/Step6_Format_labels -maxdepth 1 -type f | wc -l
find -L /data5/2025/ldh/OpenRSD/data/DOTA2_1024_500/ss_val/images -maxdepth 1 -type f | wc -l
find -L /data5/2025/ldh/OpenRSD/data/DOTA2_1024_500/ss_val/annfiles -maxdepth 1 -type f | wc -l
ls -lh /data5/2025/ldh/OpenRSD/data/Neg_supports_v2.pkl
ls -lh /data5/2025/ldh/OpenRSD/data/normalized_class_dict.pkl
ls -lh /data5/2025/ldh/OpenRSD/data/7_25_pca_meta_DINOv2_256.pkl
```

## First Reproduction Targets

Run these before any new GeoNexus claims:

```text
M_configs/G02_Baselines/Data1_DOTA2/G02_Baselines_Data1_DOTA2_M2_RoITrans.py
M_configs/G02_Baselines/Data1_DOTA2/G02_Baselines_Data1_DOTA2_M5_ORCNN_R50.py
```

Then evaluate the OpenRSD formal checkpoint if data support is complete:

```text
results/MMR_AD_A10_flex_rtm_v3_1_formal/epoch_24.pth
```
