# 2026-06-19 DOTA2 S3 Scene-Adapter Loss-0-Best Rerun Complete

## Scope

This note archives the completed DOTA2 S3 scene-adapter loss-0-best rerun
launched on 2026-06-18.

Reference launch note:
`New/docs/experiments/20260618_dota2_s3_scene_adapter_loss0_best_rerun_launch.md`.

Dataset/protocol: valid-PNG `DOTA2_1024_500/ss_val`, MMRotate `DOTAMetric`
mAP at IoU 0.5.

## Completion Status

All three replicas completed through epoch 4. No active training screens remain
for this rerun.

Scoped failure scans over the three launch logs found no matches for
`Traceback`, CUDA OOM, `out-of-memory`, `out of memory`, `libpng`, `CRC`,
`NoneType`, `ValueError`, `KeyboardInterrupt`, `loss: nan`, `loss: inf`,
`grad_norm: nan`, or `grad_norm: inf`.

## Metrics

| Replica | Seed | Source checkpoint | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 final | Best |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| rep3407 | 93407 | S2 rep3407 epoch 1 | 0.6189 | 0.6213 | 0.6141 | 0.6130 | epoch 2 |
| rep4407 | 94407 | S2 rep4407 epoch 3 | 0.6156 | 0.6160 | 0.6160 | 0.6155 | epoch 2/3 |
| rep5407 | 95407 | S2 rep5407 epoch 1 | 0.6207 | 0.6207 | 0.6133 | 0.6165 | epoch 1/2 |

Aggregate:

- S3 rerun best mean: `0.6193`.
- S3 rerun final mean: `0.6150`.

## Comparisons

- DOTA2 S1 final: `0.6177`.
- DOTA2 S2 loss-0 best mean: `0.620606`.
- DOTA2 S2 loss-0 final mean: `0.616655`.
- This rerun remains below the useful DOTA2 S2 best/final story.

## Sources

- rep3407 launch log:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep3407_20260618/launch_20260618_gpu2.log`
- rep4407 launch log:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep4407_20260618/launch_20260618_gpu3.log`
- rep5407 launch log:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep5407_20260618/launch_20260618_gpu4.log`

## Decision

The rerun remains exploratory/negative-to-neutral DOTA2 S3 evidence. It is
still below the useful DOTA2 S2 best/final story. Keep the DOTA2 follow-up
training route paused unless explicitly overridden.
