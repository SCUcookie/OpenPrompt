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
- GeoNexus S3 scene-adapter 72e completed on the same DOTA v1.5 reduced tiled split. Final epoch 72: `dota/mAP=0.3759`, `dota/AP50=0.3760`; best observed epoch 51: `dota/mAP=0.3800`, `dota/AP50=0.3800`; checkpoint `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_scene_adapter_72e/epoch_72.pth`; metric summary `docs/experiments/20260602_s3_scene_adapter_72e_metrics.json`. Treat this as completed S3 evidence; do not make stronger context-adapter claims until S3 144e and the active follow-up runs finish.
- `New/queues/geonexus_gpu_queue_20260531.json` launched S3 scene-adapter 144e on GPU 6 at `2026-06-02 00:37:05` after detecting the S3 72e checkpoint. It is active under screen `425331.geonexus_s3_scene_adapter_144e` and loaded from `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_scene_adapter_72e/epoch_72.pth`. S2 144e remains active under screen `3891792.geonexus_s2_hierarchy_reg_144e`, PID `3891957`.
- Active process snapshot at `2026-06-02 16:25 +0800`: `docs/experiments/20260602_active_process_snapshot_1625.md`. It records four active experiment screens/PIDs: S2 hierarchy regularizer 144e on GPU 2, S3 scene adapter 144e on GPU 6, S0 DOTA2 RoITrans valid-PNG on GPU 0, and S0 DOTA2 ORCNN R50 valid-PNG on GPU 1. S2 epoch-144 validation had logged `dota/mAP=0.3723`, `dota/AP50=0.3720`; parse and archive those metrics before making final S2 144e comparisons. S3 144e was at epoch 75 with about 12h48m remaining; S0 RoITrans was at epoch 2 with about 1d2h remaining; ORCNN was still in filtered annotation preparation and needs epoch-1 iteration verification after training starts.
- The manual S0 DOTA2 RoI Transformer rebuild `s0_dota2_roi_trans_rebuild_20260601` was launched on GPU 0 and marked `launched_manually=true` in the queue metadata, then failed during epoch 1 with `libpng error: IDAT: CRC error` and `AttributeError: 'NoneType' object has no attribute 'shape'` from image loading. Preserve `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260601/queue_launch_20260601.log`; failure note `docs/experiments/20260602_s0_dota2_roi_trans_rebuild_failure.md`.
- S0 DOTA2 RoI Transformer recovery on `2026-06-02`: a full Pillow decode scan of `/data5/2025/ldh/OpenRSD/data/DOTA2_1024_500/train/images` found `47` corrupt PNGs out of `170878`; corrupt list `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/corrupt_train_pngs_20260602.txt`; scan summary `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/corrupt_train_pngs_scan_summary_20260602.txt`. Filtered annotation symlink dir `/data5/2025/ldh/OpenRSD/data/DOTA2_1024_500/train/annfiles_validpng_20260602` links `170831` valid annotations and excludes the `47` corrupt-image annotations. Restart config `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/G02_Baselines_Data1_DOTA2_M2_RoITrans_validpng_20260602.py`; restart workdir/log `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/launch_20260602.log`; screen `s0_dota2_roi_trans_rebuild_validpng_20260602_gpu0`. The restart completed dataset preparation to `170831/170831`, passed the old crash point `Epoch(train) [1][1400/39007]`, reached the acceptance threshold `Epoch(train) [1][1600/39007]`, and was latest checked at `Epoch(train) [1][9500/39007]` on `2026-06-02 14:46 +0800` without `libpng`/`NoneType`/`CRC`/`Traceback` signatures. Treat this only as verified S0 DOTA2 RoI Transformer valid-PNG recovery, not S1/S2/S3/S4 evidence and not final completed-run evidence; note `docs/experiments/20260602_s0_dota2_roi_trans_rebuild_validpng_restart.md`.
- S0 DOTA2 Oriented R-CNN R50 valid-PNG baseline launched on GPU 1 at `2026-06-02 14:53 +0800`: workdir `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_orcnn_r50_validpng_20260602`; runtime config `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_orcnn_r50_validpng_20260602/G02_Baselines_Data1_DOTA2_M5_ORCNN_R50_validpng_20260602.py`; launch log `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_orcnn_r50_validpng_20260602/launch_20260602.log`; screen `s0_dota2_orcnn_r50_validpng_20260602_gpu1`; PID `1598732`. The runtime config changes only `train_ann_file` to `train/annfiles_validpng_20260602/` and keeps `ss_val/annfiles/`, `load_from = None`, `resume = False`, `max_epochs = 12`, `val_interval = 4`, and `ckpt_interval = 4`. Launch verification passed and the log entered the filtered annotation preparation pass, latest observed around `4295/170831` at `2026-06-02 14:59 +0800` with no `libpng`/`NoneType`/`CRC`/`Traceback` signatures. Training-iteration verification is still pending; do not cite as completed ORCNN evidence until `Epoch(train)` reaches at least `[1][1600/39007]` or the equivalent denominator without PNG-related crashes; note `docs/experiments/20260602_s0_dota2_orcnn_r50_validpng_launch.md`.
- OpenRSD DOTA2 epoch-12 checkpoint evaluation on `DOTA2_1024_500/ss_val` completed on `2026-06-02`: `dota/mAP=0.4202`, `dota/AP50=0.4200`. Checkpoint `/data5/2025/ldh/OpenRSD/work_dirs/opensrd_step2_dota2_nozero_full_20260531/epoch_12.pth`; config `/data5/2025/ldh/OpenRSD/work_dirs/opensrd_formal_dota2_ssval_eval/a10_formal_dota2_eval_no_star.py`; predictions `/data5/2025/ldh/OpenRSD/work_dirs/opensrd_dota2_epoch12_ssval_eval_20260602/preds.pkl`; metric summary `docs/experiments/20260602_opensrd_dota2_epoch12_ssval_metrics.json`. This is below the prior official DOTA2 `ss_val` evaluator result `dota/mAP=0.6510`, `dota/AP50=0.6510` by `-0.2308` mAP and `-0.2310` AP50. Keep claims narrow: this is DOTA2 `ss_val` evidence for the completed OpenRSD DOTA2 epoch-12 checkpoint, not a GeoNexus S2/S3 result.
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
