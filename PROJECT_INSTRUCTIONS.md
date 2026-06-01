# Project Instructions

This file is the persistent project memory. Keep it updated when the research
scope, repository structure, server workflow, or experiment protocol changes.
Paper-first rule: if the research direction, claim, experiment sequence, or
submission target changes, update the canonical manuscript and this file before
changing code, configs, or secondary docs.

## Research Direction

Project name: GeoNexus-RSD.

Primary goal: hierarchy- and context-aware vision-language prompting for
DOTA-style oriented remote sensing object detection.

Practical first target: IEEE JSTARS. Consider TGRS or ISPRS P&RS only if final
results are strong across at least two datasets. Consider GRSL, IGARSS, or a
workshop if results are modest or incomplete.

Main paper claim:

Hierarchy- and context-aware vision-language prompting improves fine-grained
oriented object detection and semi-supervised pseudo-label quality in remote
sensing imagery.

Core modules:

1. Hierarchical prompt bank.
2. Scene/context prompt adapter.
3. VLM-assisted pseudo-label purification.

Secondary only:

- Routing is optional after the three core modules are stable.
- Compression is a later-paper topic.
- Segmentation is not the primary task for this paper.

Do not claim open-vocabulary detection unless the final system uses real
vision-language embeddings and evaluates a real open-vocabulary or vocabulary
robustness setting.

## Current Code Reality

This repository is a research scaffold, not a competitive detector yet.

Current limitations:

- The local backbone is lightweight.
- The current text embedder is deterministic hash-based unless replaced.
- Official DOTA validation is integrated for the reduced tiled baseline, but the current validation mAP is still extremely low.
- The reduced DOTA v1.0 validation result is `map50=3.326794065590851e-06` on 4055 images; treat it as a pipeline sanity check only, not a paper result.

Current server evidence:

- The matched DOTA v1.5 baseline training and validation evaluation have completed.
- The v1.5 validation result is `map50=1.0926445202230628e-05` on 4055 images; it is still only a sanity-check baseline.
- The baseline comparison should stay tied to the reduced tiled setup and the same dataset/version split used for the recorded metrics.

Current diagnosis:

- Quick baseline diagnostics show the issue is not thresholding; decoded scores stay above the tested thresholds.
- Predictions collapse toward `small-vehicle`, `harbor`, `plane`, and `ship`, and a spot-checked validation tile shows center-biased boxes with very low same-class IoU.
- `QueryGenerator` computes `query_centers`, but the current box heads do not consume them, so the scaffold currently regresses boxes without an explicit spatial anchor.
- The anchor-repair quick test completed and wrote `outputs/dota_v15_anchor_repair/epoch_001.pt`; final training metrics were `loss=0.07363908355801901`, `loss_cls=0.001671954903589549`, `loss_box=0.035983564312892485`, `positive_cls_acc=0.5529336195676059`, and `positive_box_l1=0.10294117139314753`.
- S0 strong-detector baselines are complete for the controlled DOTA v1.5 split.
- The best current S0 detector is RoI Transformer 3x, epoch 34, with MMRotate DOTAMetric `dota/mAP=0.2644` and `dota/AP50=0.2640`; checkpoint `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/roi_trans_lr001_3x/epoch_34.pth`; metric summary `docs/experiments/20260526_roi_transformer_3x_dota15_metrics.json`.
- GeoNexus S2 hierarchy regularizer 12e completed on the same DOTA v1.5 reduced tiled split. Final epoch 12: `dota/mAP=0.3644`, `dota/AP50=0.3640`; best observed epoch 11: `dota/mAP=0.3652`, `dota/AP50=0.3650`; checkpoint `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_frozen_12e/epoch_12.pth`; metric summary `docs/experiments/20260601_s2_hierarchy_regularizer_12e_metrics.json`.
- GeoNexus S2 hierarchy regularizer 72e completed on the same DOTA v1.5 reduced tiled split. Final epoch 72: `dota/mAP=0.3738`, `dota/AP50=0.3740`; best observed epoch 56: `dota/mAP=0.3757`, `dota/AP50=0.3760`; checkpoint `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_frozen_72e/epoch_72.pth`; metric summary `docs/experiments/20260601_s2_hierarchy_regularizer_72e_metrics.json`. Treat this as the strongest completed S2 hierarchy-regularizer evidence, while still waiting for the active 144e and S3 runs before final convergence/context claims.
- S3 scene-adapter 72e first queue launch failed before training because the inherited base config nested `roi_head.bbox_head` incorrectly and the child config also dropped full RCNN assigner definitions. The owned child config `/data5/2025/ldh/OpenRSD/mmrotate_configs/geonexus_dota15/roi-trans-le90_r50_fpn_remoteclip-s3-72e_dota15.py` was corrected to inherit from S1 directly, define scene-adapter heads with a proper `bbox_head` list, and keep full assigner/sampler configs. The failed base file `/data5/2025/ldh/OpenRSD/mmrotate_configs/geonexus_dota15/roi-trans-le90_r50_fpn_remoteclip-s3_dota15.py` is owned by `nobody:nogroup`; avoid relying on it until permissions are fixed.
- `New/queues/geonexus_gpu_queue_20260531.json` has launched S3 scene-adapter 72e on GPU 1 and S2 hierarchy regularizer 144e on GPU 2. At the 2026-06-01 archive check, S3 72e was active under screen `3845911.geonexus_s3_scene_adapter_72e`, PID `3846084`, with latest completed validation epoch 21 (`dota/mAP=0.3562`, `dota/AP50=0.3560`) and best epoch 14 (`dota/mAP=0.3675`, `dota/AP50=0.3680`). S2 144e was active under screen `3891792.geonexus_s2_hierarchy_reg_144e`, PID `3891957`, with latest completed validation epoch 19 and best epoch 10 (`dota/mAP=0.3770`, `dota/AP50=0.3770`). S3 scene-adapter 144e remains pending, waiting for S3 72e `epoch_72.pth`.
- Oriented R-CNN 3x is the close secondary baseline, with best epoch 33/34 `dota/mAP=0.2620` and `dota/AP50=0.2620`; primary checkpoint path for the summary is `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/oriented_rcnn_3x_loadfrom/epoch_33.pth`; metric summary `docs/experiments/20260526_oriented_rcnn_3x_dota15_metrics.json`.
- ReDet pretrained completed 12 epochs with best/final `dota/mAP=0.2382` and `dota/AP50=0.2380`; checkpoint `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/redet_pretrained_rerun/epoch_12.pth`; metric summary `docs/experiments/20260526_redet_pretrained_dota15_metrics.json`.
- The earlier corrected Oriented R-CNN 12-epoch baseline remains an archived reference with `map=0.2561` and `AP50=0.2560`; checkpoint `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/oriented_rcnn/epoch_12.pth`; metric summary `docs/experiments/20260525_oriented_rcnn_dota15_epoch12_metrics.json`.
- The key strong-baseline fix was validation/test pipeline ordering: resize the image before `LoadAnnotations`, then convert qbox to rbox and pack explicit meta keys. Loading annotations before resize produced near-zero AP by evaluating against mis-scaled GT boxes.
- The earlier RoI Transformer 1x low-LR rerun and ReDet scratch rerun are superseded by the completed 3x RoI Transformer and pretrained ReDet records. Keep their logs only as troubleshooting history.
- Mid-run detector curves, class-wise snapshots, and figure/table TODOs are recorded in `docs/experiments/20260525_strong_detector_midrun_records.md`.
- The complete paper-indicator experiment matrix and current download/staging list are recorded in `docs/setup/complete_experiment_plan.md`.
- Use `docs/experiments/20260524_dota_v15_anchor_repair_quick_test.md` and `docs/setup/strong_baseline_checklist.md` as the active planning anchors.
- S1 may start only after real VLM embedding support passes a smoke test. `/data1/anaconda3/envs/zwl_mmrotate/bin/python` currently has `torch` but is missing both `open_clip` and `clip`; the RemoteCLIP checkpoint symlink exists at `/data5/2025/ldh/OpenRSD/checkpoints/remoteclip/RemoteCLIP-ViT-B-32.pt`.

Paper-level claims require:

- A credible oriented detector baseline, preferably from MMRotate or an
  equivalent strong implementation.
- Real text/image embeddings such as CLIP, SkyCLIP, or RemoteCLIP.
- Verified tiling, class mapping, rotated IoU/NMS, and mAP.
- Complete ablations with real numbers.

## Experiment Sequence

Run experiments in this order:

1. S0: strong closed-set oriented detector sweep on DOTA v1.5. Complete; use RoI Transformer 3x epoch 34 as the primary detector checkpoint unless simplicity/stability is prioritized over the small mAP lead.
2. S1: flat class-name prompt classifier with real VLM text embeddings.
3. S2: hierarchical prompt bank.
4. S3: hierarchy plus scene/context adapter.
5. S4: hierarchy plus context plus VLM-assisted pseudo-label purification.
6. S5: optional routing ablation.

Do not add S5 to the main story unless S2-S4 already show stable gains.

If the scaffold baseline is still near zero after diagnosis, pause S1-S5 and
fix the detector-localization path first or pivot to the stronger detector path.

Required final analyses:

- Main comparison table.
- Core ablation table.
- Prompt robustness table.
- Pseudo-label quality table.
- Efficiency table.
- Qualitative detections.
- Accepted/rejected pseudo-label examples.
- Confusion matrix or fine-grained class-pair analysis.

No final submission may contain pending/planned result tables.

## Paper-First Workflow

Canonical manuscript source:

- `docs/geonexus_short_paper.tex`

Supporting drafts and presentation notes may exist, but they must not override
the canonical manuscript. Keep method wording aligned with the real claim:
hierarchical prompts, scene/context adaptation, and VLM-assisted pseudo-label
purification. When code exposes routing or compression hooks, document them as
optional ablations or future work unless measured results justify making them
central.

Before any paper-facing claim is added:

- identify which experiment record supports it
- link the config and command used to produce it
- record whether the run used DOTA v1.0 or DOTA v1.5, and do not mix those numbers with later DOTA v2 results
- record whether embeddings are hash fallback or real VLM embeddings
- record whether metrics are from scaffold evaluation or accepted DOTA-style
  evaluation

## Local And Server Workflow

Use GitHub as the shared code and result-metadata transport between:

- local machine: code editing, documentation, lightweight tests
- experiment server: training, evaluation, logs, heavy artifacts

Recommended loop:

1. Local: edit code/docs/configs.
2. Local: run unit tests and smoke tests.
3. Local: commit and push to GitHub.
4. Server: pull GitHub.
5. Server: link datasets/checkpoints outside Git.
6. Server: run experiments.
7. Server: save logs and small structured summaries in Git-tracked locations.
8. Server: keep large outputs/checkpoints outside Git.
9. Server: commit and push code/config/log-summary changes.
10. Local: pull and continue analysis or code improvement.

Never put datasets, model checkpoints, raw large logs, or generated training
directories in Git. Commit configs, scripts, documentation, small metrics JSON,
environment notes, and experiment summaries.

## Repository Boundary

Tracked in Git:

- `assets/hierarchies/`
- `assets/prompts/`
- `configs/`
- `docs/`
- `scripts/`
- `src/`
- `tests/`
- root metadata and instruction files
- canonical paper source only, not duplicate generated PDFs

Ignored or external:

- `DOTA/`
- `DOTAv2/`
- `images/`
- `labels/`
- `outputs/`
- `checkpoints/`
- `artifacts/generated/`
- `wandb/`
- generated PDFs and LaTeX auxiliary files

## Future-Agent Prompt

When starting a new coding session, give the agent this instruction:

Read `PROJECT_INSTRUCTIONS.md`, then inspect the current Git status. Preserve
unrelated user changes. Continue the GeoNexus-RSD baseline-first JSTARS path:
do not make unsupported performance claims, keep routing/compression secondary,
maintain the local/server GitHub workflow, start with DOTA v1.0 or DOTA v1.5
if those server assets are already staged, and update the canonical manuscript
before code/docs when the research direction changes.
