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
- GeoNexus S2 hierarchy regularizer 72e completed on the same DOTA v1.5 reduced tiled split. Final epoch 72: `dota/mAP=0.3738`, `dota/AP50=0.3740`; best observed epoch 56: `dota/mAP=0.3757`, `dota/AP50=0.3760`; checkpoint `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_frozen_72e/epoch_72.pth`; metric summary `docs/experiments/20260601_s2_hierarchy_regularizer_72e_metrics.json`.
- GeoNexus S2 hierarchy regularizer 144e completed on the same DOTA v1.5 reduced tiled split. Final epoch 144: `dota/mAP=0.3723`, `dota/AP50=0.3720`; best observed epoch 30: `dota/mAP=0.3819`, `dota/AP50=0.3820`; checkpoint `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_frozen_144e/epoch_144.pth`; metric summary `docs/experiments/20260602_s2_hierarchy_regularizer_144e_metrics.json`. Treat best and final numbers separately: the 144e best is the strongest observed S2 validation point, while the 144e final is slightly below the 72e final.
- S3 scene-adapter 72e first queue launch failed before training because the inherited base config nested `roi_head.bbox_head` incorrectly and the child config also dropped full RCNN assigner definitions. The owned child config `/data5/2025/ldh/OpenRSD/mmrotate_configs/geonexus_dota15/roi-trans-le90_r50_fpn_remoteclip-s3-72e_dota15.py` was corrected to inherit from S1 directly, define scene-adapter heads with a proper `bbox_head` list, and keep full assigner/sampler configs. The failed base file `/data5/2025/ldh/OpenRSD/mmrotate_configs/geonexus_dota15/roi-trans-le90_r50_fpn_remoteclip-s3_dota15.py` is owned by `nobody:nogroup`; avoid relying on it until permissions are fixed.
- GeoNexus S3 scene-adapter 72e completed on the same DOTA v1.5 reduced tiled split. Final epoch 72: `dota/mAP=0.3759`, `dota/AP50=0.3760`; best observed epoch 51: `dota/mAP=0.3800`, `dota/AP50=0.3800`; checkpoint `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_scene_adapter_72e/epoch_72.pth`; metric summary `docs/experiments/20260602_s3_scene_adapter_72e_metrics.json`. Treat this as completed S3 evidence; do not make stronger context-adapter claims until S3 144e and the active follow-up runs finish.
- GeoNexus S3 scene-adapter 144e completed on the same DOTA v1.5 reduced tiled split. Final epoch 144: `dota/mAP=0.3712`, `dota/AP50=0.3710`; best observed epochs 65 and 73 tied at rounded log `dota/mAP=0.3813`, `dota/AP50=0.3810`; checkpoint `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_scene_adapter_144e/epoch_144.pth`; metric summary `docs/experiments/20260603_s3_scene_adapter_144e_metrics.json`. Treat best and final separately: the best S3 144e validation is slightly below the S2 144e best, and the final S3 144e is below the S3 72e final.
- Current GPU status at `2026-06-03 17:08 +0800`: our active valid-PNG DOTA2 baseline jobs occupy GPUs 0, 1, 2, 4, and 6; other users occupy GPUs 3 and 5. The DOTA2 RoI Transformer valid-PNG recovery completed and released GPU 0, then RTMDet-L was launched there.
- The manual S0 DOTA2 RoI Transformer rebuild `s0_dota2_roi_trans_rebuild_20260601` was launched on GPU 0 and marked `launched_manually=true` in the queue metadata, then failed during epoch 1 with `libpng error: IDAT: CRC error` and `AttributeError: 'NoneType' object has no attribute 'shape'` from image loading. Preserve `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260601/queue_launch_20260601.log`; failure note `docs/experiments/20260602_s0_dota2_roi_trans_rebuild_failure.md`.
- S0 DOTA2 RoI Transformer valid-PNG recovery completed on `2026-06-03`: a full Pillow decode scan of `/data5/2025/ldh/OpenRSD/data/DOTA2_1024_500/train/images` found `47` corrupt PNGs out of `170878`; corrupt list `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/corrupt_train_pngs_20260602.txt`; scan summary `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/corrupt_train_pngs_scan_summary_20260602.txt`. Filtered annotation symlink dir `/data5/2025/ldh/OpenRSD/data/DOTA2_1024_500/train/annfiles_validpng_20260602` links `170831` valid annotations and excludes the `47` corrupt-image annotations. Restart config `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/G02_Baselines_Data1_DOTA2_M2_RoITrans_validpng_20260602.py`; launch log `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/launch_20260602.log`; checkpoint `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/epoch_12.pth`. Final epoch 12 on `DOTA2_1024_500/ss_val`: `dota/mAP=0.6088`, `dota/AP50=0.6090` at `2026-06-03 14:31:57 +0800`; metric summary `docs/experiments/20260603_s0_dota2_roi_trans_validpng_metrics.json`; record `docs/experiments/20260602_s0_dota2_roi_trans_rebuild_validpng_restart.md`. Treat this as completed S0 DOTA2 `ss_val` evidence only, not GeoNexus S1/S2/S3/S4 evidence.
- S0 DOTA2 Oriented R-CNN R50 valid-PNG baseline launched on GPU 1 at `2026-06-02 14:53 +0800`: workdir `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_orcnn_r50_validpng_20260602`; runtime config `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_orcnn_r50_validpng_20260602/G02_Baselines_Data1_DOTA2_M5_ORCNN_R50_validpng_20260602.py`; launch log `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_orcnn_r50_validpng_20260602/launch_20260602.log`; screen `s0_dota2_orcnn_r50_validpng_20260602_gpu1`; PID `1598732`. It passed filtered annotation preparation and entered training, but failed at `Epoch(train) [1][300/39007]` with CUDA out-of-memory while computing anchor-target IoU. Preserve the log and do not cite as completed ORCNN evidence. The next ORCNN retry should reduce memory pressure before relaunching.
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
- GeoNexus S1 rerun on `2026-06-03` passed the real RemoteCLIP smoke test
  (`classes=16`, `embedding_shape=[16, 512]`) and launched on GPU 1, then
  failed at `2026-06-03 18:02:19 +0800` after epoch 1 iter 190 with CUDA OOM
  during RPN target assignment. No epoch checkpoint was produced. S2 must stay
  queued behind a successful S1 rerun checkpoint.
- 2026-06-04 GPU pruning is archived in
  `docs/experiments/20260604_gpu_pruning_and_next_priority.md`: lower-priority
  `zwl` jobs on GPUs 0/1/2/4 were stopped after checkpoint confirmation, GPU 3
  was left untouched, GeoNexus S1 retry 2 stayed active on GPU 5 with current
  best epoch 25 `dota/mAP=0.376255`, and DOTA2 ORCNN stayed active on GPU 6
  with current best epoch 8 `dota/mAP=0.585885`. The next priority is to finish
  and archive S1, then launch the next S2 hierarchy-regularizer rerun from the
  best S1 checkpoint before restarting secondary DOTA2 baselines.

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

For active experiment monitoring, every pass must check `screen -ls`,
`nvidia-smi`, and the active run log before reporting status. If a run is gone
or the log shows a failure, first read the traceback and classify the reason.
For `CUDA out of memory`, wait for an allowed physical GPU with
`memory.used <= 4000 MiB` and `util <= 10%` for three consecutive polls before
restarting there. For `libpng`, `CRC`, `NoneType`, or other data-read errors,
identify the bad file/sample first and do not relaunch unchanged unless that
input is fixed or excluded. For import/config errors, fix the environment or
config before relaunch. For an unknown traceback, record the traceback and
allow one clean-GPU relaunch; if the same traceback repeats, stop and fix it.
Cap automatic retries at three per experiment. Each retry must use a new log
name containing the retry index and GPU, and the handoff note must record the
failure reason plus restart command. If `last_checkpoint` exists, resume from
it; otherwise restart from epoch 0. Do not launch S2 until the current S1 rerun
successfully completes and produces the intended initialization checkpoint.

## Active Server Runs

- 2026-06-03 19:15 CST: GeoNexus S1 RemoteCLIP prompt-head rerun retry 1 ran
  on physical GPU 1 in screen `geonexus_s1_rerun_retry1_20260603_gpu1`; PID
  `3300816`.
  Config:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_rerun_20260603/roi-trans-le90_r50_fpn_remoteclip-s1-rerun-20260603_dota15.py`.
  Log:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_rerun_20260603/launch_retry1_20260603_gpu1.log`.
  The first launch failed at `2026-06-03 18:02:19 +0800` with CUDA OOM after
  epoch 1 iter 190 and no checkpoint. Retry 1 was started only after physical
  GPU 1 passed three consecutive OOM-retry polls below `4000 MiB` and `10%`.
  Startup check reached `Epoch(train) [1][30/1410]` at
  `2026-06-03 19:18:07 +0800`, then failed at
  `2026-06-03 19:19:11 +0800` with the same CUDA OOM class at iter 190. No
  checkpoint was produced and no S1 screen remained active. The S1 rerun config
  was then patched to add `gpu_assign_thr=256` to the RPN and both cascade RCNN
  assigners, matching the corrected S2/S3 dense-assignment memory mitigation.
  Retry 2 may start only after the CUDA-OOM GPU gate passes again. Keep S2
  blocked until this S1 rerun produces the intended checkpoint.

- 2026-06-03 09:13 CST: OpenRSD S0 DOTA2 Oriented R-CNN R50 valid-PNG
  memory-safe retry is running on physical GPU 6 in screen
  `s0_dota2_orcnn_r50_validpng_bs1_20260603_gpu6`.
  Config:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_orcnn_r50_validpng_bs1_20260603/G02_Baselines_Data1_DOTA2_M5_ORCNN_R50_validpng_bs1_20260603.py`.
  Log:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_orcnn_r50_validpng_bs1_20260603/launch_20260603.log`.
  This retry uses the DOTA2 valid-PNG train annotations
  `train/annfiles_validpng_20260602/`, `ss_val/annfiles/` validation,
  batch size 1, lr 0.00125, 800x800 scale, 12 epochs, val/ckpt interval 4.
  Accepted startup check: dataset prep completed, training passed the previous
  OOM point `[1][300/78014]`, and reached `[1][1800/78014]` with no
  `CUDA out of memory`, `Traceback`, `libpng`, `CRC`, or `NoneType` signatures.
  Latest observed progress at 2026-06-03 16:48 CST was
  `Epoch(train) [3][15300/78014]`, ETA about 1 day 9:17.

- 2026-06-03 09:29 CST: OpenRSD S0 DOTA2 S2ANet valid-PNG batch-size-1 run
  is running on physical GPU 1 in screen
  `s0_dota2_s2anet_validpng_bs1_20260603_gpu1`.
  Config:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_s2anet_validpng_bs1_20260603/G02_Baselines_Data1_DOTA2_M3_S2ANet_validpng_bs1_20260603.py`.
  Log:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_s2anet_validpng_bs1_20260603/launch_20260603.log`.
  Startup check reached `[1][1050/78014]`; latest observed progress at
  2026-06-03 16:48 CST was `Epoch(train) [3][50650/78014]`, ETA about
  1 day 0:48.

- 2026-06-03 09:29 CST: OpenRSD S0 DOTA2 R3Det-KFIoU valid-PNG
  batch-size-1 run is running on physical GPU 2 in screen
  `s0_dota2_r3det_kfiou_validpng_bs1_20260603_gpu2`.
  Config:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_r3det_kfiou_validpng_bs1_20260603/G02_Baselines_Data1_DOTA2_M4_R3Det_KFIoU_validpng_bs1_20260603.py`.
  Log:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_r3det_kfiou_validpng_bs1_20260603/launch_20260603.log`.
  Startup check reached `[1][600/78014]`; latest observed progress at
  2026-06-03 16:48 CST was `Epoch(train) [2][62600/78014]`, ETA about
  1 day 16:50.

- 2026-06-03 09:29 CST: OpenRSD S0 DOTA2 RTMDet-M valid-PNG batch-size-1 run
  is running on physical GPU 4 in screen
  `s0_dota2_rtmdet_m_validpng_bs1_20260603_gpu4`.
  Config:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_rtmdet_m_validpng_bs1_20260603/G02_Baselines_Data1_DOTA2_M9_RTMDet_M_validpng_bs1_20260603.py`.
  Log:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_rtmdet_m_validpng_bs1_20260603/launch_20260603.log`.
  The CSPNeXt-M checkpoint downloaded successfully. Startup check reached
  `[1][750/78014]`; latest observed progress at 2026-06-03 16:48 CST was
  `Epoch(train) [3][16250/78014]`, ETA about 1 day 7:58.

- 2026-06-03 16:51 CST: OpenRSD S0 DOTA2 RTMDet-L valid-PNG batch-size-1 run
  is running on physical GPU 0 in screen
  `s0_dota2_rtmdet_l_validpng_bs1_20260603_gpu0`; PID `3031320`.
  Config:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_rtmdet_l_validpng_bs1_20260603/G02_Baselines_Data1_DOTA2_M10_RTMDet_L_validpng_bs1_20260603.py`.
  Log:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_rtmdet_l_validpng_bs1_20260603/launch_20260603.log`.
  This run uses DOTA2 valid-PNG train annotations
  `train/annfiles_validpng_20260602/`, `ss_val/annfiles/` validation,
  batch size 1, 12 epochs, val/ckpt interval 4, and the cached CSPNeXt-L
  checkpoint `/home/zwl/.cache/torch/hub/checkpoints/cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth`.
  Dataset preparation completed over `170831/170831` annotations, with
  non-fatal too-many-instances cut warnings on dense tiles. Startup check
  reached `[1][1600/78014]` at 2026-06-03 17:07:55 CST with no
  `CUDA out of memory`, `Traceback`, `libpng`, `CRC`, or `NoneType`
  signatures.
