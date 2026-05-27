# DOTA v1.5 GeoNexus S1 Archive - 2026-05-26

This archive records the DOTA v1.5 work started on 2026-05-26 and completed on 2026-05-27. It separates paper-facing MMRotate RoI Transformer S1 evidence from the scaffold diagnostics. Scaffold metrics are not paper evidence.

## Paper-Facing Baselines

The closed-set S0 baselines are already archived in `20260526_paper_evidence_dota15_summary.md`:

| Detector | Best checkpoint | DOTAMetric mAP / AP50 |
|---|---|---:|
| RoI Transformer 3x | `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/roi_trans_lr001_3x/epoch_34.pth` | 0.2644 / 0.2640 |
| Oriented R-CNN 3x | `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/oriented_rcnn_3x_loadfrom/epoch_33.pth` | 0.2620 / 0.2620 |
| ReDet pretrained | `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/redet_pretrained_rerun/epoch_12.pth` | 0.2382 / 0.2380 |

## Code Fixes

MMRotate S1 required these fixes before reliable training:

- `/data5/2025/ldh/OpenRSD/tools/bootstrap_run.py`: preload installed OpenMMLab packages and re-add the repo root without putting repo-local `mmengine`, `mmdet`, or `mmrotate` ahead of installed packages.
- `/data5/2025/ldh/OpenRSD/geonexus_mmrotate/prompt_bbox_head.py`: register `PromptShared2FCBBoxHead` into `mmrotate.registry.MODELS`.
- Runtime configs set `gpu_assign_thr=256` in RPN and both cascade-stage assigners so dense DOTA target assignment falls back to CPU instead of OOMing on large images.
- Offset/logit ablation configs were corrected from `resume=True` to `resume=False`; they load S0 epoch 34 as weights only and do not restore incompatible optimizer state.

## Completed S1 Result Selection

All selected MMRotate S1 variants reached epoch 36 with DOTAMetric validation records. The current S1 candidate is the frozen-backbone S1 checkpoint at epoch 6; do not cite epoch 36 as the best result because multiple runs peaked earlier.

| Variant | Work dir | Selected checkpoint | Best epoch | Best DOTAMetric mAP / AP50 | Final epoch 36 mAP / AP50 | Role |
|---|---|---|---:|---:|---:|---|
| Frozen backbone | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_frozen_backbone` | `epoch_6.pth` | 6 | 0.2666 / 0.2670 | 0.2506 / 0.2510 | Current strongest S1 candidate |
| Main sanitized S1 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1` | `epoch_32.pth` | 32 | 0.2651 / 0.2650 | 0.2597 / 0.2600 | Supporting S1 ablation |
| Learnable prompt offsets | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_offsets` | `epoch_29.pth` | 29 | 0.2654 / 0.2650 | 0.2541 / 0.2540 | Supporting S1 ablation |
| Prompt logit scale 5.0 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_logit5` | `epoch_21.pth` | 21 | 0.2606 / 0.2610 | 0.2507 / 0.2510 | Diagnostic ablation only |

Metric source files:

- Frozen backbone: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_frozen_backbone/20260526_211216/vis_data/scalars.json`
- Main sanitized S1: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1/20260526_210320/vis_data/scalars.json`
- Learnable prompt offsets: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_offsets/20260526_211754/vis_data/scalars.json`
- Prompt logit scale 5.0: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_logit5/20260526_211754/vis_data/scalars.json`

Checkpoint existence was confirmed for the promoted/supporting S1 selections:

- Frozen backbone epoch 6: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_frozen_backbone/epoch_6.pth`
- Main sanitized epoch 32: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1/epoch_32.pth`
- Learnable prompt offsets epoch 29: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_offsets/epoch_29.pth`

The S1 validation curves are completed MMRotate DOTAMetric evidence on the same reduced DOTA v1.5 tiled split as S0. Full paper artifact parity with S0 is now complete for frozen-backbone S1 epoch 6.

## Frozen-Backbone S1 Evidence Package

Canonical evidence directory: `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/geonexus_s1_frozen_epoch6_cuda`

| Artifact | Path / value |
|---|---|
| Checkpoint | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_frozen_backbone/epoch_6.pth` |
| Re-run DOTAMetric | `dota/mAP=0.2666`, `dota/AP50=0.2670` |
| Raw JSON | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/geonexus_s1_frozen_epoch6_cuda/20260527_094448/20260527_094448.json` |
| Prediction dump | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/geonexus_s1_frozen_epoch6_cuda/preds.pkl` |
| Test log | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/geonexus_s1_frozen_epoch6_cuda/test.log` |
| Confusion matrix | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/geonexus_s1_frozen_epoch6_cuda/confusion/confusion_matrix.png` |
| Qualitative examples | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/geonexus_s1_frozen_epoch6_cuda/qualitative_confidence/` |
| Benchmark | `12.6 FPS`, `79.4 ms/img`, `546 MB CUDA`; log `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/geonexus_s1_frozen_epoch6_cuda/benchmark.log` |

The standard `tools/analysis_tools/analyze_results.py` qualitative path emitted the expected per-image mAP fallback warning and stalled before writing PNGs. The packaged qualitative examples therefore use `/data5/2025/ldh/New/scripts/render_mmrotate_qualitative_confidence.py`, which ranks the existing `preds.pkl` by mean detection confidence and renders 5 `good` plus 5 `bad` examples with the MMRotate visualizer. `qualitative_confidence/ranking_summary.txt` records the selected tile indices and confidence scores.

## Main S1 Run

| Field | Value |
|---|---|
| Screen | `roi_trans_remoteclip_s1_gpu1_cpuassign` |
| GPU | 1 |
| Work dir | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1` |
| Runtime config | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1/roi_trans_remoteclip_s1_sanitized.py` |
| Current log | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1/screen_train_gpu1_cpuassign.log` |
| Checkpoints selected for evidence | `epoch_32.pth` as supporting ablation; `epoch_36.pth` as completed final checkpoint |
| Completion state | Completed 36 epochs with finite training losses and validation at epoch 36 |
| Best validation observed | Epoch 32 DOTAMetric `dota/mAP=0.2651`, `dota/AP50=0.2650` |
| Final validation observed | Epoch 36 DOTAMetric `dota/mAP=0.2597`, `dota/AP50=0.2600` |

The epoch-32 result is slightly above the S0 RoI Transformer best AP50 of `0.2640`, but it is not the strongest S1 result after the frozen-backbone run completed.

## Parallel S1 Variants

| Variant | Screen | GPU | Config | Log | Completed status |
|---|---|---:|---|---|---|
| Frozen backbone | `s1_frozen_gpu4` | 4 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_frozen_backbone/roi_trans_remoteclip_s1_frozen_backbone.py` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_frozen_backbone/train.log` | Completed 36 epochs; best epoch 6 `mAP=0.2666`, `AP50=0.2670` |
| Learnable prompt offsets | `s1_offsets_gpu0` | 0 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_offsets/roi_trans_remoteclip_s1_offsets.py` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_offsets/train_v3.log` | Completed 36 epochs; best epoch 29 `mAP=0.2654`, `AP50=0.2650` |
| Prompt logit scale 5.0 | `s1_logit5_gpu3` | 3 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_logit5/roi_trans_remoteclip_s1_logit5.py` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_logit5/train_v3.log` | Completed 36 epochs; best epoch 21 `mAP=0.2606`, `AP50=0.2610` |

All three variants use `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/roi_trans_lr001_3x/epoch_34.pth` as `load_from` and `resume=False`.

## Failed Attempts And Resolutions

| Issue | Evidence | Resolution |
|---|---|---|
| Initial S1 OOM during RPN assignment | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1/screen_train.log` | Added `gpu_assign_thr=256` to assigners |
| Epoch-1 validation subprocess imported repo-local OpenMMLab packages | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1/screen_train_gpu1.log` | Adjusted `tools/bootstrap_run.py` path ordering |
| Offset/logit ablations crashed with optimizer parameter-group mismatch | `train_v2.log` in each ablation work dir | Set `resume=False`; keep S0 checkpoint as `load_from` only |

## Scaffold Status

The scaffold run remains diagnostic-only:

- Output dir: `/data5/2025/ldh/New/outputs/dota_v15_geonexus`
- Preserved dir: `/data5/2025/ldh/New/outputs/preserved_20260526_dota_v15_geonexus_remoteclip`
- Metrics record: `/data5/2025/ldh/New/docs/experiments/20260526_dota15_geonexus_remoteclip_scaffold_metrics.json`
- Known failures: missing `cv2` in an earlier launch, `scene_adapter` feature shape mismatch (`1x256` into `512x512`), unstable `positive_cls_acc`, incomplete checkpoint/metric state in the preserved diagnostic record.

Do not cite scaffold metrics in paper tables or claims.

## Process Status

The MMRotate S1 set completed after the original archive-time live-process snapshot. Earlier failed or corrected attempts remain in the logs and are retained as provenance. Completed evidence should be taken from the `20260526_210320`, `20260526_211216`, and `20260526_211754` run directories listed above, not from the earlier OOM/import/resume-fix logs.

## Next Actions

1. Optionally generate the same compact test JSON for main sanitized epoch 32 and offsets epoch 29 so the supporting ablations are reproducible from selected checkpoints, not only training validation curves.
2. Start S2 hierarchical prompt bank from the frozen-backbone S1 setup.
