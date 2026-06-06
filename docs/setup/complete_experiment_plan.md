# Complete Experiment Plan For Paper Indicators

Date: 2026-06-06

Goal: build paper-facing GeoNexus-RSD evidence around DOTA2 as the primary
benchmark, DIOR-R as required cross-dataset validation, and FAIR1M as optional
fine-grained stretch evidence. DOTA v1.5 GeoNexus results are now
diagnostic/archive-only and must not be used as headline results.

## Current DOTA2 S0 Status

Use `DOTA2_1024_500/ss_val` as the main benchmark split unless a later note
documents a stricter official evaluation.

Completed closed-set baselines to archive:

- RoI Transformer valid-PNG recovery: `dota/mAP=0.6088`, `AP50=0.6090`;
  metric source `docs/experiments/20260603_s0_dota2_roi_trans_validpng_metrics.json`.
- Oriented R-CNN valid-PNG baseline: `dota/mAP=0.5973`, `AP50=0.5970`;
  status source `docs/experiments/20260605_dota2_baseline_status.md`.
- S2ANet valid-PNG baseline: `dota/mAP=0.5869`, `AP50=0.5870`;
  status source `docs/experiments/20260605_dota2_baseline_status.md`.
- RTMDet-M valid-PNG baseline: `dota/mAP=0.3312`, `AP50=0.3310`.

Active decisions:

- Let R3Det-KFIoU finish because it is already late in training.
- Reassess RTMDet-L after its next validation; if it remains near `0.35`, stop
  it and free GPU 6.
- Do not spend more compute on DOTA v1.5 S2/S3/S4 refinements unless the user
  explicitly requests archive/debug work.

## Benchmark Matrix

| Dataset | Role | First baseline | GeoNexus module | Acceptance |
| --- | --- | --- | --- | --- |
| DOTA2 | Primary paper benchmark | RoI Transformer, Oriented R-CNN, S2ANet, R3Det, RTMDet variants | Minimal S1/S2 hierarchy-aware prompt scoring or hierarchy regularization on the strongest stable detector | S1/S2 must beat or clearly complement the strongest closed-set baseline before S3/S4 |
| DIOR-R | Required cross-dataset validation | Oriented R-CNN or RoI Transformer on `DIOR_R_dota/train_val` and `DIOR_R_dota/test` | Same minimal S1/S2 module used on DOTA2 | Loader/config smoke reaches validation before full training |
| FAIR1M | Optional fine-grained stretch | Use only after DOTA2 and DIOR-R are stable | Hierarchy-focused S1/S2 evidence | Supports fine-grained hierarchy claims, not first cross-dataset proof |
| DOTA v1.5 | Archive/debug only | Existing RoI Transformer/ORCNN/ReDet records | Existing S1/S2/S3 diagnostics only | Do not use as headline paper table evidence |

## Required Paper Tables

### Main Comparison

Rows:

- Strong closed-set oriented detectors on DOTA2.
- GeoNexus-RSD S1/S2 on the selected DOTA2 detector.
- DIOR-R closed-set baseline.
- DIOR-R GeoNexus-RSD S1/S2 using the same module.
- FAIR1M rows only if compute allows stable fine-grained evidence.

Columns:

- dataset version and split, detector, prompt/VLM setting, config, checkpoint,
  metric source, mAP/AP50, per-class AP summary, inference speed, and peak GPU
  memory.

### Core Ablation

Keep the detector, data split, schedule, evaluator, and VLM backend fixed.

- S0: closed-set detector.
- S1: flat or hierarchy-aware prompt scoring with real VLM text embeddings.
- S2: hierarchy regularization or hierarchical prompt bank.
- S3: hierarchy plus scene/context adapter, only after S1/S2 are credible.
- S4: hierarchy/context plus VLM-assisted pseudo-label purification, only after
  S3 is justified.
- S5: optional routing only if S2-S4 are stable.

### Prompt Robustness

Evaluate with frozen detector weights where possible:

- exact class names only.
- aliases only or class names plus aliases.
- parent-category prompts.
- full mixed prompts with hierarchy, scene, geometry, confusing, and negative
  cues.

Report overall mAP/AP50 plus fine-grained pairs relevant to each dataset.

### Pseudo-Label Quality

Use a labeled holdout subset as if it were unlabeled, then compare pseudo labels
against ground truth before retraining:

- teacher detector only.
- detector plus hierarchy consistency.
- detector plus VLM crop-text agreement.
- full purification score.

Report pseudo-label precision, recall, F1, accepted-label count, and class-wise
quality. Run S4 retraining only after the filtering quality is defensible.

### Efficiency

Measure on the same GPU type:

- training time per epoch.
- validation time.
- inference FPS or images/s.
- peak GPU memory.
- prompt embedding cache build time.
- VLM crop-filtering throughput.

## Run Order

1. Finish DOTA2 R3Det and make the RTMDet-L keep/stop decision.
2. Archive the DOTA2 baseline table with exact configs, checkpoints, logs, and
   metric JSON or markdown source for every row.
3. Choose the strongest stable DOTA2 detector, favoring RoI Transformer unless
   implementation stability clearly favors Oriented R-CNN.
4. Port only the minimal GeoNexus S1/S2 module to DOTA2.
5. Run a DOTA2 GeoNexus smoke and then full run only if the smoke reaches
   validation without data/config failures.
6. Stage and smoke DIOR-R via `DIOR_R_dota/train_val` and `DIOR_R_dota/test`.
7. Run the DIOR-R closed-set baseline.
8. Run the same minimal DIOR-R GeoNexus module.
9. Decide whether FAIR1M compute is justified after DOTA2 and DIOR-R are stable.

## Assets To Stage

Required:

- `DOTA2_1024_500` valid-PNG annotations and exact corrupt-file exclusion list.
- `DIOR_R_dota/train_val` and `DIOR_R_dota/test`.
- RemoteCLIP checkpoint already staged at
  `/data5/2025/ldh/OpenRSD/checkpoints/remoteclip/RemoteCLIP-ViT-B-32.pt`.
- The selected DOTA2 detector checkpoint and config for S1/S2 initialization.

Recommended:

- OpenAI CLIP or OpenCLIP weights as a natural-image VLM comparison.
- SkyCLIP/SkyScript weights for a second remote-sensing VLM comparison.
- FAIR1M converted annotations only after DOTA2 and DIOR-R are stable.

Public source anchors:

- RemoteCLIP: `https://arxiv.org/abs/2306.11029`
- SkyScript/SkyCLIP: `https://arxiv.org/abs/2312.12856`
- GeoRSCLIP: `https://arxiv.org/abs/2306.11300`
- OpenRSD: `https://arxiv.org/abs/2503.06146`
- DOTA: `https://arxiv.org/abs/1711.10398`
- DIOR: `https://arxiv.org/abs/1909.00133`
- FAIR1M: `https://arxiv.org/abs/2103.05569`
