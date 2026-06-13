# 2026-06-13 DIOR-R RetinaNet S0 Sanitized Long Complete

## Scope

Recorded the completed DIOR-R RetinaNet S0 continuation on the sanitized DIOR-R protocol.

- Workdir: `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_retinanet_sanitized_long_20260612_gpu2`
- Config: `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_retinanet_sanitized_long_20260612_gpu2/dior_r_s0_retinanet_sanitized_long_20260612_gpu2.py`
- Runtime logs:
  - `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_retinanet_sanitized_long_20260612_gpu2/20260612_181213/20260612_181213.log`
  - `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_retinanet_sanitized_long_20260612_gpu2/20260612_235311/20260612_235311.log`
- Final checkpoint: `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_retinanet_sanitized_long_20260612_gpu2/epoch_96.pth`

## Result

- Epoch 96 final/best: `dota/mAP=0.5694328547`, `dota/AP50=0.569`

## Interpretation

This is secondary, below-baseline evidence. It remains below the DIOR-R RoI Transformer S0/S1 path and should not be treated as paper-leading DIOR-R evidence.

## Failure Scan

Scoped scan over the RetinaNet S0 continuation workdir found no matches for:

`Traceback`, CUDA OOM/out-of-memory, `libpng`, `CRC`, `NoneType`, `ValueError`, `KeyboardInterrupt`, `loss: nan/inf`, or `grad_norm: nan/inf`.
