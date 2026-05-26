# DOTA v1.5 GeoNexus S1 Archive - 2026-05-26

This archive records the DOTA v1.5 work done on 2026-05-26. It separates paper-facing MMRotate RoI Transformer S1 evidence from the scaffold diagnostics. Scaffold metrics are not paper evidence.

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

## Main S1 Run

| Field | Value |
|---|---|
| Screen | `roi_trans_remoteclip_s1_gpu1_cpuassign` |
| GPU | 1 |
| Work dir | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1` |
| Runtime config | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1/roi_trans_remoteclip_s1_sanitized.py` |
| Current log | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1/screen_train_gpu1_cpuassign.log` |
| Checkpoints observed | `epoch_1.pth`, `epoch_2.pth` |
| State at archive time | Training epoch 3 with finite losses |
| Validation observed | Epoch 2 DOTAMetric `dota/mAP=0.2510`, `dota/AP50=0.2510` |

The epoch-2 result is below the S0 RoI Transformer best AP50 of `0.2640`, but it is an early S1 checkpoint. Do not add S1 to the main comparison table until the run finishes or a best-checkpoint validation is selected.

## Parallel S1 Variants

| Variant | Screen | GPU | Config | Log | Status at archive time |
|---|---|---:|---|---|---|
| Frozen backbone | `s1_frozen_gpu4` | 4 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_frozen_backbone/roi_trans_remoteclip_s1_frozen_backbone.py` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_frozen_backbone/train.log` | Training epoch 1 with finite losses |
| Learnable prompt offsets | `s1_offsets_gpu0` | 0 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_offsets/roi_trans_remoteclip_s1_offsets.py` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_offsets/train_v3.log` | Training epoch 1 with finite losses |
| Prompt logit scale 5.0 | `s1_logit5_gpu3` | 3 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_logit5/roi_trans_remoteclip_s1_logit5.py` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_logit5/train_v3.log` | Training epoch 1 with finite losses |

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

## Active Processes At Archive Time

| GPU | Process/session | Purpose |
|---:|---|---|
| 0 | `s1_offsets_gpu0` | S1 learnable prompt offsets ablation |
| 1 | `roi_trans_remoteclip_s1_gpu1_cpuassign` | Main S1 MMRotate run |
| 2 | `dota_v15_geonexus_remoteclip` | Scaffold diagnostic run |
| 3 | `s1_logit5_gpu3` | S1 prompt-logit-scale ablation |
| 4 | `s1_frozen_gpu4` | S1 frozen-backbone ablation |
| 5 | unrelated `lyc` process | left untouched |
| 6 | unrelated `lyc` process | left untouched |

Expected finish window for the MMRotate S1 set was approximately 02:10-03:30 on 2026-05-27, assuming no OOM or crash.

## Next Actions

1. Let the four MMRotate S1 runs finish or reach useful validation checkpoints.
2. Extract `dota/mAP` and `dota/AP50` from each run's `vis_data/*.json` or log.
3. Update `20260526_paper_evidence_dota15_summary.md`; only promote S1 into the main comparison when a completed DOTAMetric best checkpoint is selected.
4. If all S1 variants remain below the S0 RoI Transformer AP50 `0.2640`, prioritize staged unfreezing: freeze the detector trunk first, train prompt projection/bias, then unfreeze RoI heads.
