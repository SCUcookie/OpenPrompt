# 2026-07-13 Phase A Three-GPU Analysis Launch

## Scope

Concurrent paper-finalization analysis only. No detector training was
launched. The launch follows `docs/experiments/20260713_paper_finalization_schedule.md`.

## Preflight

- Repository commit: `c44ad28d` (`0713`)
- GPUs 0, 1, and 4 were idle at the successful launches; GPU 2 and GPU 6 were occupied.
- Baseline checkpoint: `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_roi_trans_sanitized_long_20260612_gpu1/epoch_52.pth`
- GeoNexus checkpoint: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_s2e12_rep0_20260614/epoch_8.pth`
- Test image set: `11726.png 12003.png 14830.png 17650.png` (curated IDs from the July 13 schedule; existence checked before launch).

## Launch Record

| GPU | Screen | Job | Output |
| --- | --- | --- | --- |
| 0 | `tgrs_efficiency_baseline_20260713_gpu0` | A3 baseline params/FLOPs/FPS | `/data5/2025/ldh/OpenRSD/work_dirs/paper_analysis_20260713/efficiency_baseline.json` |
| 1 | `tgrs_efficiency_geonexus_20260713_gpu1` | A3 GeoNexus params/FLOPs/FPS | `/data5/2025/ldh/OpenRSD/work_dirs/paper_analysis_20260713/efficiency_geonexus.json` |
| 4 | `tgrs_qualitative_20260713_gpu4` | A2 baseline/GeoNexus strip | `/data5/2025/ldh/OpenRSD/work_dirs/paper_analysis_20260713/qualitative/` |

Launch timestamp: `2026-07-13` server local time; see the launch logs for the
exact commands. The first A2 GPU-3 attempt failed on an incorrect checkpoint
path and was replaced by the successful GPU-4 launch. The host NVIDIA driver
became temporarily unavailable after jobs completed; no job remains running.

## Acceptance

Each screen must show model initialization and progress without traceback,
CUDA OOM, decode/CRC errors, or NaN/Inf signatures. Efficiency jobs must
write JSON containing params, FLOPs or an explicit FLOPs warning, median FPS,
latency, and GPU name. The qualitative job must write the stitched PNG.

## Results

- Baseline: `55.39M` parameters, `19.13 FPS`, `52.27 ms` median latency;
  FLOPs analysis returned `None` with a recorded warning.
- GeoNexus SCA: `58.31M` parameters, `18.96 FPS`, `52.75 ms` median latency;
  FLOPs analysis returned `None` with a recorded warning.
- Qualitative strip: `2048x1024` PNG, exit status `0`.
