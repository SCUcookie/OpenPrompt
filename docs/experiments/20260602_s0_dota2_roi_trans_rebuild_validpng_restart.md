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

## Verification Status

The requested acceptance threshold was reached. The old failure point was
`Epoch(train) [1][1400/39022]`, and the intended restart verification was at
least `Epoch(train) [1][1600/39022]`.

As of `2026-06-02 14:46 +0800`, the restart screen was still present and the
training PID `1077281` was active on GPU 0. The launch log had advanced past
the old crash point and the acceptance threshold:

```text
06/02 14:17:31 - mmengine - INFO - Epoch(train)  [1][ 1600/39007] ...
06/02 14:46:15 - mmengine - INFO - Epoch(train)  [1][ 9500/39007] ...
```

No `libpng`, `NoneType`, `Traceback`, or `CRC` crash signature was present in
the launch log through the latest targeted check. The iteration denominator is
`39007` after filtering corrupt-image annotations.

This verifies S0 DOTA2 RoI Transformer valid-PNG recovery past the previous
PNG decode failure point. Do not cite this as S1/S2/S3/S4 evidence or as a
completed training run until the run finishes and final metrics are parsed.
