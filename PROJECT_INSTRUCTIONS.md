# Project Instructions

This file is the persistent project memory. Keep it updated when the research
scope, repository structure, server workflow, or experiment protocol changes.
Paper-first rule: if the research direction, claim, experiment sequence, or
submission target changes, update the canonical manuscript and this file before
changing code, configs, or secondary docs.

## Research Direction

Project name: GeoNexus-RSD.

Primary goal: hierarchy- and context-aware vision-language prompting for
DOTA2-centered oriented remote sensing object detection, with DIOR-R as the
required cross-dataset validation and FAIR1M as stretch fine-grained evidence.

Practical first target: IEEE JSTARS. Consider TGRS or ISPRS P&RS only if final
results are strong across at least two datasets. Consider GRSL, IGARSS, or a
workshop if results are modest or incomplete.

Main paper claim:

Hierarchy- and context-aware vision-language prompting improves fine-grained
oriented object detection and semi-supervised pseudo-label quality in remote
sensing imagery.

2026-06-06 research pivot:

- DOTA v1.5 GeoNexus runs are diagnostic/archive-only evidence. They remain
  useful for debugging hierarchy/context code paths, but they are no longer the
  formal benchmark route for the paper and must not be used as headline table
  evidence.
- The formal benchmark order is now DOTA2 first, DIOR-R second, and FAIR1M
  only after DOTA2 plus DIOR-R are stable.
- Stop extending the DOTA v1.5 S2/S3/S4 chain. The lower-LR DOTA v1.5 S2
  refinement in screen
  `geonexus_s2_hierarchy_refine_s2e4_lr5e5_20260606_gpu1` was stopped by
  research pivot on 2026-06-06 at about epoch 3, not classified as a failed
  experiment. Preserve its workdir, launch log, and any partial checkpoints:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_refine_s2e4_lr5e5_20260606`.
- Active priority order: finish DOTA2 baselines and archive their exact
  config/checkpoint/metric sources; run only the most defensible DOTA2
  GeoNexus S1/S2 module on the strongest stable detector; establish a DIOR-R
  baseline on `DIOR_R_dota/train_val` and `DIOR_R_dota/test`; then repeat the
  same minimal GeoNexus module on DIOR-R. FAIR1M is stretch evidence for
  fine-grained hierarchy claims, not the first cross-dataset proof.

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
- S1 real VLM embedding support passed the RemoteCLIP smoke test (`classes=16`,
  `embedding_shape=[16, 512]`), using the checkpoint symlink at
  `/data5/2025/ldh/OpenRSD/checkpoints/remoteclip/RemoteCLIP-ViT-B-32.pt`.
  Earlier S1 launches on 2026-06-03 failed with CUDA OOM before checkpointing,
  but the patched retry 2 completed 36 epochs on 2026-06-04.
- 2026-06-04 GPU pruning is archived in
  `docs/experiments/20260604_gpu_pruning_and_next_priority.md`: lower-priority
  `zwl` jobs on GPUs 0/1/2/4 were stopped after checkpoint confirmation, GPU 3
  was left untouched, GeoNexus S1 retry 2 stayed active on GPU 5 with current
  best epoch 25 `dota/mAP=0.376255`, and DOTA2 ORCNN stayed active on GPU 6
  with current best epoch 8 `dota/mAP=0.585885`. The next priority is to finish
  and archive S1, then launch the next S2 hierarchy-regularizer rerun from the
  best S1 checkpoint before restarting secondary DOTA2 baselines.
- 2026-06-05 recovery update: GeoNexus S1 retry 2 completed 36 epochs. Best
  epoch 32: `dota/mAP=0.3800`, `dota/AP50=0.3800`; final epoch 36:
  `dota/mAP=0.3793`, `dota/AP50=0.3790`; metric summary
  `docs/experiments/20260605_geonexus_s1_retry2_metrics.json`.
- 2026-06-05 recovery update: GeoNexus S2 hierarchy-regularizer rerun from S1
  epoch 32 completed 12 epochs. Best epoch 4: `dota/mAP=0.3858`,
  `dota/AP50=0.3860`; final epoch 12: `dota/mAP=0.3784`,
  `dota/AP50=0.3780`; metric summary
  `docs/experiments/20260605_geonexus_s2_rerun_s1e32_metrics.json`. Use
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_rerun_s1e32_20260604/epoch_4.pth`
  as the S3 initialization checkpoint.
- 2026-06-05 recovery update: GeoNexus S3 scene-adapter rerun from S2 epoch 4
  completed 12 epochs and released GPU 1. Best epoch 2: `dota/mAP=0.3827`,
  `dota/AP50=0.3830`; final epoch 12: `dota/mAP=0.3756`,
  `dota/AP50=0.3760`; metric summary
  `docs/experiments/20260605_geonexus_s3_rerun_s2e4_metrics.json`.
  Compared with S1 retry2 best epoch 32 `0.3800/0.3800` and S2 rerun best
  epoch 4 `0.3858/0.3860`, hierarchy is currently positive but the scene
  adapter is not yet stable enough to support the paper claim. Do not launch S4
  yet. The next diagnostic priority is a controlled S3 repair from S2 epoch 4
  using an identity-initialized scene adapter that honors `scene_adapter_dim`
  and a reduced residual scale.
- 2026-06-05 launch update: the controlled S3 repair run was launched on GPU 1
  in screen `geonexus_s3_identity_rerun_s2e4_20260605_gpu1`. Work dir:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_scene_adapter_identity_rerun_s2e4_20260605`;
  config:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_scene_adapter_identity_rerun_s2e4_20260605/roi-trans-le90_r50_fpn_remoteclip-s3-identity-rerun-s2e4-20260605_dota15.py`;
  launch log:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_scene_adapter_identity_rerun_s2e4_20260605/launch_20260605_gpu1.log`.
  Startup acceptance passed at `Epoch(train) [1][200/1410]` with no
  `Traceback`, CUDA OOM, `libpng`, `CRC`, or `NoneType` signature in the log.
- 2026-06-05 recovery update: the controlled S3 identity repair from S2 epoch
  4 completed 12 epochs. Best epoch 9: `dota/mAP=0.3806`,
  `dota/AP50=0.3810`; final epoch 12: `dota/mAP=0.3792`,
  `dota/AP50=0.3790`; metric summary
  `docs/experiments/20260605_geonexus_s3_identity_rerun_s2e4_metrics.json`.
  This did not recover the prior S3 rerun best `0.3827/0.3830` or the S2
  rerun best `0.3858/0.3860`. The final S3 diagnostic is an adapter-off rerun
  from the same S2 epoch-4 checkpoint to isolate whether degradation comes from
  scene modulation or the S3 head/config transition.
- 2026-06-05 launch update: the final S3 adapter-off diagnostic was launched
  on GPU 3 in screen `geonexus_s3_adapter_off_rerun_s2e4_20260605_gpu3`.
  Work dir:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_adapter_off_rerun_s2e4_20260605`;
  config:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_adapter_off_rerun_s2e4_20260605/roi-trans-le90_r50_fpn_remoteclip-s3-adapter-off-rerun-s2e4-20260605_dota15.py`;
  launch log:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_adapter_off_rerun_s2e4_20260605/launch_20260605_gpu3.log`.
  Preflight passed: GPU 3 stayed at `14 MiB` and `0%` over three polls, the
  config parsed with both cascade bbox heads reporting `use_scene_adapter=False`,
  and startup reached `Epoch(train) [1][200/1410]` with no `Traceback`, CUDA
  OOM, `libpng`, `CRC`, or `NoneType` signature in the launch log.
- 2026-06-06 recovery update: the final S3 adapter-off diagnostic from S2
  epoch 4 completed 12 epochs. Best epoch 3: `dota/mAP=0.3772`,
  `dota/AP50=0.3770`; final epoch 12: `dota/mAP=0.3758`,
  `dota/AP50=0.3760`; metric summary
  `docs/experiments/20260606_geonexus_s3_adapter_off_rerun_s2e4_metrics.json`.
  This stayed below the stop threshold `0.3827` and below the S2 rerun best
  `0.3858/0.3860`. Stop S3 diagnostics and do not launch S4 from S3. The next
  GeoNexus paper-path run is an S2 hierarchy-stabilization refinement from S2
  epoch 4 with LR `1e-4`, not S4.
- 2026-06-05 DOTA2 secondary status: ORCNN completed epoch 12 at
  `dota/mAP=0.5973`, `dota/AP50=0.5970`; S2ANet completed epoch 12 at
  `dota/mAP=0.5869`, `dota/AP50=0.5870`; R3Det, RTMDet-M, and RTMDet-L were
  interrupted by `KeyboardInterrupt`. Do not auto-restart those secondary runs
  before the next GeoNexus paper-path run. Status note:
  `docs/experiments/20260605_dota2_baseline_status.md`.
- 2026-06-05 GPU state before S3 launch planning: screens for our training runs
  are gone; GPUs 0, 1, 3, 5, and 6 are effectively free, while user `lyc`
  owns compute jobs on GPUs 2 and 4. Prefer GPU 1 for S3 if three consecutive
  `nvidia-smi` polls keep it at `memory.used <= 4000 MiB` and `util <= 10%`;
  otherwise use GPU 5 or GPU 6. Do not use GPU 3 for the next S3 launch.
- 2026-06-06 DOTA2 secondary update: RTMDet-M resumed and completed epoch 12
  with `dota/mAP=0.3312`, `dota/AP50=0.3310`. R3Det-KFIoU is active on GPU 5
  in screen `s0_dota2_r3det_kfiou_validpng_bs1_resume_20260605_gpu5`, currently
  epoch 10 with about nine hours remaining as of `2026-06-06 09:45 +0800`.
  RTMDet-L remains the only unfinished secondary baseline resume candidate;
  resume from
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_rtmdet_l_validpng_bs1_20260603/epoch_4.pth`
  only if GPU 6 passes the three-poll idle check.
- 2026-06-06 launch update: with R3Det still active on GPU 5, GPU 1 and GPU 6
  passed three idle polls (`14 MiB`, `0%` each). RTMDet-L was resumed on GPU 6
  in screen `s0_dota2_rtmdet_l_validpng_bs1_resume_20260606_gpu6`; launch log
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_rtmdet_l_validpng_bs1_20260603/launch_resume_20260606_gpu6.log`;
  startup passed at epoch 5 `[200/78014]` with no `Traceback`, CUDA OOM,
  `libpng`, `CRC`, `NoneType`, or immediate `KeyboardInterrupt`. GeoNexus S2
  hierarchy refinement from S2 epoch 4 was launched on GPU 1 in screen
  `geonexus_s2_hierarchy_refine_s2e4_lr1e4_20260606_gpu1`; work dir
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_refine_s2e4_lr1e4_20260606`;
  config
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_refine_s2e4_lr1e4_20260606/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-refine-s2e4-lr1e4-20260606_dota15.py`;
  launch log
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_refine_s2e4_lr1e4_20260606/launch_20260606_gpu1.log`;
  startup passed at epoch 1 `[200/1410]` with no listed failure signatures.
- 2026-06-06 recovery update: GeoNexus S2 hierarchy refinement from S2 epoch 4
  with LR `1e-4` completed 12 epochs and released GPU 1. Best epoch 1:
  `dota/mAP=0.3804`, `dota/AP50=0.3800`; final epoch 12:
  `dota/mAP=0.3765`, `dota/AP50=0.3760`; metric summary
  `docs/experiments/20260606_geonexus_s2_refine_s2e4_lr1e4_metrics.json`.
  This did not improve the S2 rerun best epoch 4 `0.3858/0.3860`. The later
  lower-LR S2 refinement is now archive-only under the 2026-06-06 DOTA2
  cross-dataset pivot.
- 2026-06-06 launch update: the lower-LR S2 refinement from S2 epoch 4 was
  launched on GPU 1 with LR `5e-5` in screen
  `geonexus_s2_hierarchy_refine_s2e4_lr5e5_20260606_gpu1`. Work dir:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_refine_s2e4_lr5e5_20260606`;
  config:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_refine_s2e4_lr5e5_20260606/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-refine-s2e4-lr5e5-20260606_dota15.py`;
  launch log:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_refine_s2e4_lr5e5_20260606/launch_20260606_gpu1.log`.
  Startup passed at epoch 1 `[200/1410]` with no `Traceback`, CUDA OOM,
  `libpng`, `CRC`, `NoneType`, `KeyboardInterrupt`, or early exit signature.
  It was stopped by research pivot on 2026-06-06 at about epoch 3; preserve the
  partial log/checkpoint artifacts and do not relaunch as paper-path evidence.

Paper-level claims require:

- A credible oriented detector baseline, preferably from MMRotate or an
  equivalent strong implementation.
- Real text/image embeddings such as CLIP, SkyCLIP, or RemoteCLIP.
- Verified tiling, class mapping, rotated IoU/NMS, and mAP.
- Complete ablations with real numbers.

## Experiment Sequence

Run experiments in this order:

1. DOTA2 S0: complete and archive strong closed-set baselines. Completed
   baselines: RoI Transformer `0.6088/0.6090`, Oriented R-CNN
   `0.5973/0.5970`, S2ANet `0.5869/0.5870`, RTMDet-M `0.3312/0.3310` on
   `DOTA2_1024_500/ss_val`. Let active R3Det finish because it is already
   near completion. Reassess RTMDet-L after its next validation; if it remains
   near `0.35`, stop it and free GPU 6.
2. DOTA2 GeoNexus S1/S2: port only the strongest defensible module first:
   hierarchy-aware prompt scoring or hierarchy regularization on the strongest
   stable DOTA2 detector. Do not run S3/S4 until DOTA2 S1/S2 beats or clearly
   complements the strongest closed-set baseline.
3. DIOR-R S0: run an Oriented R-CNN or RoI Transformer baseline using local
   `DIOR_R_dota/train_val` and `DIOR_R_dota/test`, with a smoke validation
   before full training.
4. DIOR-R GeoNexus S1/S2: repeat the same minimal GeoNexus module used on
   DOTA2, without changing the paper story between datasets.
5. FAIR1M: stretch evidence after DOTA2 and DIOR-R are stable. Use it for
   fine-grained hierarchy claims only.
6. S3/S4 and optional routing: run only after S1/S2 provide credible evidence
   on the DOTA2 and DIOR-R path.

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
- record the exact dataset version and split, especially distinguishing
  DOTA v1.0, DOTA v1.5, DOTA2, DIOR-R, and FAIR1M
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
unrelated user changes. Continue the GeoNexus-RSD DOTA2-first JSTARS path:
do not make unsupported performance claims, keep routing/compression secondary,
maintain the local/server GitHub workflow, treat DOTA v1.5 as archive-only
diagnostic evidence, make DOTA2 the primary benchmark, make DIOR-R the required
cross-dataset validation, and update the canonical manuscript before code/docs
when the research direction changes.

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
it; otherwise restart from epoch 0. Do not relaunch DOTA v1.5 GeoNexus
refinements unless the user explicitly asks for archive/debug work.

## Active Server Runs

- DOTA2 R3Det-KFIoU baseline remains active on GPU 5 in screen
  `s0_dota2_r3det_kfiou_validpng_bs1_resume_20260605_gpu5`; let it finish
  and archive the final metric source.
- DOTA2 RTMDet-L baseline remains active on GPU 6 in screen
  `s0_dota2_rtmdet_l_validpng_bs1_resume_20260606_gpu6`; reassess after the
  next validation and stop it if it stays near `0.35`.
- DIOR-R Oriented R-CNN R50 baseline is active on GPU 1 in screen
  `s0_dior_orcnn_r50_20260606_gpu1`. Workdir:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dior_r_orcnn_r50_20260606`; config:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dior_r_orcnn_r50_20260606/G02_Baselines_Data2_DIOR_R_M5_ORCNN_R50.py`;
  launch log:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dior_r_orcnn_r50_20260606/launch_20260606_gpu1.log`.
  Startup acceptance passed at epoch 1 `[100/5862]` with no `Traceback`, CUDA
  OOM, `libpng`, `CRC`, `NoneType`, or early-exit signature.
- Result monitor `s0_result_log_monitor_20260603` remains active.
- No active GeoNexus DOTA v1.5 training screen should exist after the
  2026-06-06 pivot. GPU 1 was freed after stopping
  `geonexus_s2_hierarchy_refine_s2e4_lr5e5_20260606_gpu1`.
