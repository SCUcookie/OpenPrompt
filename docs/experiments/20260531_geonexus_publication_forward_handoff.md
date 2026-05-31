# GeoNexus-RSD Publication Forward Handoff - 2026-05-31

This is the consolidated record for the current GeoNexus-RSD publication path.
It records completed evidence, active runs, implementation state, publication
plan, and unfinished work. Large checkpoints/logs remain under `OpenRSD/`; this
file is the durable index in `New/`.

## Publication Direction

Primary target: TGRS/JSTARS-style remote-sensing oriented detection paper.

Stretch target: AAAI-style method paper only if hierarchy/context prompting
shows clear algorithmic generality across detector families or datasets.

Working title:

`Hierarchy- and Scene-Aware Vision-Language Prompting for Oriented Object Detection in Remote Sensing Images`

Core claim:

GeoNexus-RSD improves prompt-based remote-sensing oriented detection by injecting
class hierarchy and scene context into a strong detector's classification head,
improving robustness to fine-grained class confusion, prompt wording, and
DOTA-style small/rotated objects.

Do not claim generic open-vocabulary detection unless held-out vocabulary or
cross-dataset transfer experiments prove it.

## Literature Positioning

- DOTA/DOTA2 are the main empirical anchor for oriented aerial detection.
- RoI Transformer, Oriented R-CNN, ReDet, and Oriented RepPoints are geometry
  baselines. GeoNexus should be framed as semantic hierarchy/context on top of
  these detectors, not as a new geometry representation.
- RemoteCLIP, SkyScript/SkyCLIP, and GeoRSCLIP motivate remote-sensing VLM
  alignment, but most evidence there is classification/retrieval/foundation
  oriented rather than fine-grained oriented detection.
- OpenRSD is the closest prompt-based remote-sensing detection prior. Frame this
  work as structured hierarchy/context prompting, not simply OpenRSD on DOTA2.

## Completed DOTA v1.5 Evidence

Canonical summary:

`/data5/2025/ldh/New/docs/experiments/20260526_paper_evidence_dota15_summary.md`

| Method | Checkpoint | Best epoch | DOTAMetric mAP / AP50 | Status |
|---|---|---:|---:|---|
| RoI Transformer S0 | `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/roi_trans_lr001_3x/epoch_34.pth` | 34 | 0.2644 / 0.2640 | Primary closed-set baseline |
| GeoNexus S1 frozen backbone | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_frozen_backbone/epoch_6.pth` | 6 | 0.2666 / 0.2670 | Current strongest S1 |
| GeoNexus S2 hierarchy offsets | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_frozen_offsets_12e/epoch_1.pth` | 1 | 0.2666 / 0.2670 | Parity/slight S1 improvement, not robust yet |
| Oriented R-CNN S0 | `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/oriented_rcnn_3x_loadfrom/epoch_33.pth` | 33 | 0.2620 / 0.2620 | Secondary detector baseline |
| ReDet pretrained | `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/redet_pretrained_rerun/epoch_12.pth` | 12 | 0.2382 / 0.2380 | Comparison baseline |

Important interpretation:

- S1 improves S0 by only about `+0.0022` mAP. This is promising but not a paper
  claim by itself.
- S2 hierarchy offsets currently match S1 within rounding. Treat it as parity,
  not a robust hierarchy gain.
- The paper-worthy path is now robustness, hierarchy regularization, scene
  context, confusion reduction, and repeatability.

## Completed S1/S2 Records

- S1 archive:
  `/data5/2025/ldh/New/docs/experiments/20260526_dota15_geonexus_s1_archive.md`
- S2 archive:
  `/data5/2025/ldh/New/docs/experiments/20260527_dota15_geonexus_s2_archive.md`
- Scaffold diagnostics:
  `/data5/2025/ldh/New/docs/experiments/20260526_dota15_geonexus_remoteclip_scaffold_metrics.json`

The `New/` scaffold remains diagnostic-only. Paper-facing evidence is the
MMRotate/OpenRSD strong-detector path.

## New S2 Regularizer Implementation

Implemented on 2026-05-31:

- Head:
  `/data5/2025/ldh/OpenRSD/geonexus_mmrotate/hierarchy_prompt_bbox_head.py`
- Config:
  `/data5/2025/ldh/OpenRSD/mmrotate_configs/geonexus_dota15/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-12e_dota15.py`

Behavior:

- Loads `relation_matrix` from
  `/data5/2025/ldh/New/artifacts/generated/remoteclip_vit_b32_dota15_s2_hierarchy_prompt_embeddings.pt`.
- Registers `HierarchyPromptShared2FCBBoxHead` into both MMDet and MMRotate
  registries.
- Adds supervised `loss_hierarchy` on positive RoIs. The target class remains
  dominant, while related classes receive small soft-target mass.
- Uses frozen-backbone S1 initialization via:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_frozen_offsets_12e/s1_epoch6_without_prompt_embeddings.pth`

Verification:

- `py_compile` passed.
- Head-level forward/loss smoke produced finite `loss_hierarchy`.
- Full model build passed under the repository bootstrap import semantics:
  model `CascadeRCNN`, two `HierarchyPromptShared2FCBBoxHead` cascade stages,
  hierarchy loss weights `[0.05, 0.05]`, relation matrices `(16, 16)`.

Not yet achieved:

- No training result exists yet for the S2 regularizer config.
- Need 12-epoch smoke/full run and then 36-epoch if it is non-negative.

## DOTA2 OpenRSD Status

Canonical launch/resume record:

`/data5/2025/ldh/New/docs/experiments/20260531_resume_opensrd_dota2_launch.md`

Smoke result:

- `OpenRSD/work_dirs/opensrd_step2_dota2_nozero_smoke/epoch_1.pth`
- Reached `Epoch(train) [1][200/200]`.

Full run:

- Config:
  `/data5/2025/ldh/OpenRSD/work_dirs/opensrd_step2_dota2_nozero_full_20260531/opensrd_step2_dota2_nozero_full_20260531.py`
- Work dir:
  `/data5/2025/ldh/OpenRSD/work_dirs/opensrd_step2_dota2_nozero_full_20260531`
- Resume session:
  `2396455.opensrd_dota2_full_20260531`
- Active process observed on 2026-05-31:
  `/data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py ... --resume auto`
- Latest observed log state:
  `Epoch(train) [9][10750/12000]` at `2026/05/31 16:04:41`
- Saved checkpoints observed: `epoch_1.pth` through `epoch_8.pth`.
- Validation is disabled. This is training/scalability evidence only until
  validation against `ss_val/annfiles` is adapted and run.

Known DOTA2 issue:

- Earlier full run crashed on corrupt PNG decode.
- Dataset guard now skips unreadable images with warning, e.g.
  `Skipping unreadable image: data/DOTA2_1024_500/train/images/P8461__682__1396___4188.png`.
- Future restarts should use:
  `/data5/2025/ldh/OpenRSD/data/Formatted_FederatedLabels/Data1_DOTA2_nozero_validpng_20260531`

## Required Paper Tables

Table 1: Main detection results

- RoI Transformer S0.
- Oriented R-CNN S0.
- ReDet reference.
- GeoNexus S1 flat prompt.
- GeoNexus S2 hierarchy prompt bank.
- GeoNexus S2 + hierarchy regularizer.
- GeoNexus S3 hierarchy + scene context.
- Optional S4 pseudo-label retraining.
- Report mAP, AP50, per-class AP, FPS, memory, checkpoint.

Table 2: Core ablation

- S0 closed-set detector.
- S1 flat RemoteCLIP class prompts.
- S2 hierarchy prompt bank.
- S2 + hierarchy regularizer.
- S3 scene-context modulation.
- S4 pseudo-label purification if stable.

Table 3: Prompt robustness

- Exact class names.
- Aliases only.
- Parent prompts only.
- Mixed hierarchy prompts.
- Hard-negative prompts.
- Report confusion pairs: small vehicle vs large vehicle, ship vs harbor,
  storage tank vs roundabout, bridge vs road/background, sports field subclasses.

Table 4: Pseudo-label quality

- Teacher only.
- Teacher + hierarchy consistency.
- Teacher + VLM crop agreement.
- Full purification.
- Report precision, recall, F1, accepted-label count, and class-wise quality on
  a labeled holdout treated as unlabeled.

Table 5: Generality

- Minimum TGRS: DOTA v1.5 plus DOTA2.
- Stronger TGRS: add DIOR-R or HRSC2016.
- AAAI stretch: add a non-remote-sensing/open-vocabulary benchmark or show
  transfer across detector families.

## Success Criteria

Minimum TGRS/JSTARS threshold:

- S2 or S3 beats RoI Transformer S0 by at least `+0.8` to `+1.0` mAP on DOTA
  v1.5, or shows smaller AP gain plus strong prompt robustness and confusion
  reduction.
- Gains reproduce on at least two seeds or two detector backbones.
- Per-class improvements align with the method story, especially
  fine-grained/context-sensitive classes.

Strong TGRS threshold:

- S3 improves `+1.5` mAP or more over S0/S1.
- Prompt robustness table shows clear degradation resistance under
  aliases/parent prompts.
- DOTA2 or cross-dataset result confirms the method is not overfit to DOTA v1.5.

AAAI threshold:

- Must show an algorithmic contribution beyond remote-sensing engineering:
  hierarchy-aware prompt alignment, graph-regularized prompt learning, or
  scene-conditioned semantic calibration.
- Must include multiple datasets, multiple detectors, and clean ablation proof.

## Unfinished Work

Immediate:

1. Let the active DOTA2 full run finish.
2. Archive final DOTA2 checkpoint, log, and note that validation remains disabled.
3. Adapt DOTA2 validation against `ss_val/annfiles` if safe.
4. Run S2 hierarchy regularizer 12e using the new config.
5. Compare S2 regularizer against S1 frozen and S2 hierarchy-offset epoch 1.

Next:

6. If S2 regularizer is non-negative, run a longer 36e or repeat-seed version.
7. Run S2 on Oriented R-CNN if RoI Transformer S2 improves.
8. Implement S3 scene-context adapter only after S2 is stable.
9. Run prompt robustness evaluation using frozen checkpoints.
10. Build confusion matrices focused on the named fine-grained class pairs.

Later:

11. Run pseudo-label quality study on a labeled holdout.
12. Only launch S4 retraining if pseudo-label F1 improves clearly.
13. Generate final qualitative figures: good/bad examples, confusion matrices,
    prompt failure cases, and rejected pseudo-label examples.
14. Add DIOR-R or HRSC2016 if aiming above minimum TGRS.
15. Draft TGRS manuscript first; derive AAAI version only if S2/S3 generality is
    strong.

## Do Not Claim Yet

- Do not claim robust hierarchy gain from current S2 offsets; the observed gain
  over S1 is only at the fourth decimal.
- Do not claim DOTA2 superiority; validation is not complete.
- Do not claim open-vocabulary generalization; no held-out vocabulary or
  cross-dataset vocabulary test is complete.
- Do not use the lightweight `New/` scaffold as paper-facing AP evidence.
- Do not present S4 pseudo-label purification as core contribution until
  holdout pseudo-label quality and retraining mAP improve clearly.

## Existing Source Records To Preserve

- Complete experiment plan:
  `/data5/2025/ldh/New/docs/setup/complete_experiment_plan.md`
- Strong detector sweep:
  `/data5/2025/ldh/New/docs/experiments/20260525_strong_detector_sweep.md`
- S1 archive:
  `/data5/2025/ldh/New/docs/experiments/20260526_dota15_geonexus_s1_archive.md`
- S2 archive:
  `/data5/2025/ldh/New/docs/experiments/20260527_dota15_geonexus_s2_archive.md`
- DOTA2 launch/resume:
  `/data5/2025/ldh/New/docs/experiments/20260531_resume_opensrd_dota2_launch.md`
- Method notes:
  `/data5/2025/ldh/New/docs/method/geonexus_rsd.md`
