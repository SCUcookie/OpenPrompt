# DOTA v1.5 GeoNexus S2 Archive - 2026-05-27

This record archives the first paper-facing S2 hierarchical prompt run on the reduced tiled DOTA v1.5 split. It should be interpreted as comparable to S1, not as a stable improvement, until repeat/robustness checks are complete.

## Run Identity

| Field | Value |
|---|---|
| Machine | `nuosen` |
| Git commit (`New`) | `276d6ec` |
| GPU inventory | 7x NVIDIA GeForce RTX 4090 |
| Dataset | DOTA v1.5 reduced tiled split |
| Dataset root | `/data5/2025/ldh/OpenPrompt/DOTA/` |
| Base detector | MMRotate RoI Transformer, le90, ResNet-50 FPN |
| S2 work dir | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_frozen_offsets_12e` |
| Runtime config | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_frozen_offsets_12e/roi_trans_remoteclip_s2_hierarchy_frozen_offsets_12e.py` |
| Prompt embeddings | `/data5/2025/ldh/New/artifacts/generated/remoteclip_vit_b32_dota15_s2_hierarchy_prompt_embeddings.pt` |
| Initialization | S1 frozen epoch 6 weights with prompt embedding keys removed |
| Init checkpoint | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_frozen_offsets_12e/s1_epoch6_without_prompt_embeddings.pth` |

## Training Result Selection

S2 trained for 12 epochs with finite losses and validation at every epoch. The best checkpoint is epoch 1; later epochs decline and should not be cited as the selected S2 result.

| Epoch | DOTAMetric mAP | AP50 |
|---:|---:|---:|
| 1 | 0.266634 | 0.267000 |
| 2 | 0.263598 | 0.264000 |
| 3 | 0.264125 | 0.264000 |
| 4 | 0.262464 | 0.262000 |
| 5 | 0.260594 | 0.261000 |
| 6 | 0.260324 | 0.260000 |
| 7 | 0.254936 | 0.255000 |
| 8 | 0.255635 | 0.256000 |
| 9 | 0.258762 | 0.259000 |
| 10 | 0.258160 | 0.258000 |
| 11 | 0.255964 | 0.256000 |
| 12 | 0.259702 | 0.260000 |

Selected checkpoint:

`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_frozen_offsets_12e/epoch_1.pth`

Metric source:

`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_frozen_offsets_12e/20260527_114723/vis_data/scalars.json`

## Evidence Package

Canonical evidence directory:

`/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/geonexus_s2_hierarchy_offsets_epoch1_cuda`

| Artifact | Path / value |
|---|---|
| Re-run DOTAMetric | `dota/mAP=0.2666`, `dota/AP50=0.2670` |
| Raw JSON/log dir | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/geonexus_s2_hierarchy_offsets_epoch1_cuda/20260527_152713/` |
| Prediction dump | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/geonexus_s2_hierarchy_offsets_epoch1_cuda/preds.pkl` |
| Test log | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/geonexus_s2_hierarchy_offsets_epoch1_cuda/test.log` |
| Runtime config copy | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/geonexus_s2_hierarchy_offsets_epoch1_cuda/roi_trans_remoteclip_s2_hierarchy_frozen_offsets_12e.py` |
| Qualitative examples | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/geonexus_s2_hierarchy_offsets_epoch1_cuda/qualitative_confidence/` |
| Qualitative ranking | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/geonexus_s2_hierarchy_offsets_epoch1_cuda/qualitative_confidence/ranking_summary.txt` |
| Benchmark | `11.6 FPS`, `86.2 ms/img`, `546 MB CUDA` |
| Benchmark log | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/geonexus_s2_hierarchy_offsets_epoch1_cuda/benchmark.log` |

## Class-Wise Re-Run Metrics

| Class | GTs | Dets | Recall | AP |
|---|---:|---:|---:|---:|
| plane | 2550 | 1282 | 0.324 | 0.350 |
| baseball-diamond | 213 | 308 | 0.376 | 0.357 |
| bridge | 466 | 327 | 0.079 | 0.091 |
| ground-track-field | 145 | 568 | 0.469 | 0.400 |
| small-vehicle | 43337 | 3189 | 0.031 | 0.088 |
| large-vehicle | 5139 | 4064 | 0.420 | 0.411 |
| ship | 10765 | 2179 | 0.091 | 0.091 |
| tennis-court | 763 | 998 | 0.781 | 0.726 |
| basketball-court | 143 | 271 | 0.329 | 0.315 |
| storage-tank | 2940 | 212 | 0.026 | 0.091 |
| soccer-ball-field | 149 | 301 | 0.450 | 0.376 |
| roundabout | 185 | 237 | 0.103 | 0.114 |
| harbor | 2102 | 2104 | 0.432 | 0.412 |
| swimming-pool | 576 | 314 | 0.125 | 0.146 |
| helicopter | 78 | 128 | 0.231 | 0.205 |
| container-crane | 14 | 17 | 0.071 | 0.091 |

## Failure Notes

- The standard `tools/analysis_tools/analyze_results.py` path again emitted the per-image mAP fallback warning. It progressed through the validation set but did not write a `confusion_matrix.png` before the tool session stopped returning output. This mirrors the S1 archive issue, so the packaged qualitative evidence uses `New/scripts/render_mmrotate_qualitative_confidence.py`.
- A first qualitative attempt without `tools/bootstrap_run.py` failed because direct `PYTHONPATH=/data5/2025/ldh/OpenRSD` exposed repo-local OpenMMLab shims. The successful attempt used `tools/bootstrap_run.py`.
- A first benchmark attempt inside the sandbox failed because CUDA was not visible. The successful benchmark was rerun outside the sandbox with `CUDA_VISIBLE_DEVICES=0`.

## Interpretation

S2 epoch 1 is numerically comparable to the S1 frozen epoch 6 candidate (`0.2666/0.2670`) and only marginally above it in the raw scalar value (`0.266634` vs. S1 `0.266591`). Treat this as hierarchy prompt parity/slight improvement, not as a robust gain, until repeat seed and prompt robustness checks are complete.

## Next Actions

1. Start S3 hierarchy plus scene/context adapter from the strongest S1/S2 checkpoint.
2. Repeat S1 frozen and S2 hierarchy-offset candidates with alternate seeds/configs.
3. Run prompt robustness evaluations for class names, aliases, parent prompts, and full hierarchy prompts.
