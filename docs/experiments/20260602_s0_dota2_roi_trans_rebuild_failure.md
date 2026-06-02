# 2026-06-02 S0 DOTA2 RoI Transformer Rebuild Failure

## Launch

- Experiment: `s0_dota2_roi_trans_rebuild_20260601`.
- Screen: `s0_dota2_roi_trans_rebuild_20260601_gpu0`.
- GPU: 0.
- Work dir: `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260601`.
- Log: `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260601/queue_launch_20260601.log`.
- Queue metadata was marked non-pending in `New/queues/geonexus_gpu_queue_20260531.json` with `launched_manually=true`.

## Failure

The run started normal MMEngine training and reached epoch 1 iteration 1400/39022,
then exited after a PNG decode failure:

```text
libpng error: IDAT: CRC error
AttributeError: 'NoneType' object has no attribute 'shape'
```

This is a data/decode failure in the training dataloader, not a transient GPU
failure. The run was not restarted blindly. Preserve the full workdir log before
retrying, and repair or exclude the unreadable DOTA2 image referenced by the
dataloader path once identified.
