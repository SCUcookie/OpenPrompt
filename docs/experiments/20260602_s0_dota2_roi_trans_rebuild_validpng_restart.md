# 2026-06-02 S0 DOTA2 RoI Transformer Valid-PNG Restart

## Scope

This record is only for the S0 DOTA2 RoI Transformer recovery attempt after the
`s0_dota2_roi_trans_rebuild_20260601` dataloader PNG decode failure. It is not
S1/S2/S3 evidence.

## Data Repair

- Failed source run: `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260601`.
- Preserved failed log: `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260601/queue_launch_20260601.log`.
- Full decode scan: `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/corrupt_train_pngs_scan_summary_20260602.txt`.
- Corrupt list: `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/corrupt_train_pngs_20260602.txt`.
- Scan result: `170878` train PNGs scanned with `Pillow.Image.open.load`; `47` corrupt PNGs found.
- Filtered annotation dir: `/data5/2025/ldh/OpenRSD/data/DOTA2_1024_500/train/annfiles_validpng_20260602`.
- Filter summary: `170878` source annotations, `170831` symlinked valid annotations, `47` excluded, `0` missing corrupt-image annotations.

Original images, original annotations, and `train/annfiles` were not modified.

## Restart

- Work dir: `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart`.
- Config: `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/G02_Baselines_Data1_DOTA2_M2_RoITrans_validpng_20260602.py`.
- Train annotations: `train/annfiles_validpng_20260602/`.
- Validation annotations: `ss_val/annfiles/`.
- `resume = False`.
- `load_from = None`.
- Screen: `s0_dota2_roi_trans_rebuild_validpng_20260602_gpu0`.
- Launch log: `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/launch_20260602.log`.

The restart completed the training annotation preparation pass:

```text
100%|██████████| 170831/170831 [1:58:06<00:00, 24.11it/s]
```

No `libpng`, `NoneType`, `Traceback`, or `CRC` crash signature was present in
the launch log through the last check.

## Completion

The requested startup acceptance threshold was reached early in epoch 1. The
old failure point was `Epoch(train) [1][1400/39022]`, and the intended restart
verification was at least `Epoch(train) [1][1600/39022]`.

```text
06/02 14:17:31 - mmengine - INFO - Epoch(train)  [1][ 1600/39007] ...
```

The run then completed epoch 12 and saved the final checkpoint:

- Checkpoint: `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/epoch_12.pth`.
- Final validation timestamp: `2026-06-03 14:31:57 +0800`.
- Final metrics: `dota/mAP=0.6088`, `dota/AP50=0.6090`.
- Metric summary: `docs/experiments/20260603_s0_dota2_roi_trans_validpng_metrics.json`.

```text
06/03 14:20:11 - mmengine - INFO - Saving checkpoint at 12 epochs
06/03 14:31:57 - mmengine - INFO - Epoch(val) [12][6917/6917]    dota/mAP: 0.6088  dota/AP50: 0.6090  data_time: 0.0061  time: 0.0947
```

No `libpng`, `NoneType`, `Traceback`, or `CRC` crash signature was present in
the final launch log. The iteration denominator is `39007` after filtering
corrupt-image annotations.

This is completed S0 DOTA2 RoI Transformer valid-PNG evidence on
`DOTA2_1024_500/ss_val`. Do not cite this as GeoNexus S1/S2/S3/S4 evidence.
