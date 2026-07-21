# 2026-07-21 Next Training Route Plan (Full Stage Design)

## Project state this plan builds on

As of commit `5274d9b8` the evidence loop is closed end to end:

- **Manuscript: complete.** 8 pages, zero undefined references, zero
  `[TO FILL]` markers. Table V (per-class), Table VI (efficiency), and
  Fig. 4 (qualitative strip) all carry committed evidence. Headline:
  DIOR-R 69.79 mean / 69.92 best AP50 vs OrientedFormer 68.83 (our
  reproduction) and Strip R-CNN 68.70 (reported); DOTA-v2.0 60.88 -> 62.06.
- **Closed routes (respected throughout this plan, none reopened):**
  FAIR1M S2/GeoNexus, DIOR-R S4 pseudo-labeling, segmentation lane, and
  DOTA-v2.0 follow-up training, per the 2026-07-21 evidence-bundle manifest.
- **What remains before submission (non-training):** advisor review pass,
  author-list finalization, submission package. Optional evidence upgrades
  are listed in Section 4.

The purpose of this plan: define the **next training campaigns** — what to
train, why, and the full stage design — so GPU time strengthens the work
where a TGRS reviewer will actually push. The two open scientific exposures
of the current manuscript are (1) a single detector family and (2) a single
backbone. Both admit clean, non-closed, high-value training routes.

## Route R1 — Cross-Detector Generality Campaign (priority 1)

**Claim targeted:** "GeoNexus-RSD is a detector-compatible prompting
framework" — currently demonstrated on exactly one detector. Porting the
stack to Oriented R-CNN on DIOR-R converts the paper's central framing into
a measured result and pre-empts the most likely major-revision request.

**Why Oriented R-CNN:** its DIOR-R baseline already exists in our protocol
(best epoch 28, `63.41` AP50, from the 2026-06-13 sanitized long runs), it
is a single-head (non-cascade) two-stage detector so the port is the
smallest possible code delta, and it is universally known to reviewers.

### Stage R1-P0 — Code port and gates (no GPU training)

1. Implement `PromptRotatedShared2FCBBoxHead`: the existing
   `geonexus_mmrotate/prompt_bbox_head.py` targets the RoI Transformer
   cascade heads; Oriented R-CNN uses a single `RotatedShared2FCBBoxHead`.
   The port reuses the identical prompt-embedding buffer, cosine classifier,
   projections, and (for R1-S2) hierarchy loss hooks — only the parent class
   and the single-stage wiring change. Add a unit-level config gate that
   asserts prompt keys initialize and `num_classes=20` matches the
   embedding shape `[20, 512]`.
2. Verify the Oriented R-CNN DIOR-R baseline checkpoint still exists on the
   server (the 2026-06-13 note says the final epoch was not locally
   archived; epoch-28 best must be located or, failing that, the baseline is
   retrained once under R1-S0). Record its exact path + SHA-256 in the
   launch note.
3. Reuse unchanged: sanitized DIOR-R root, canonical DIOR-R prompt artifact
   (`remoteclip_vit_b32_dior_r_s2_hierarchy_prompt_embeddings.pt`, which
   already contains the 20-class `relation_matrix` — unlike FAIR1M, no new
   relation artifact is needed), evaluation protocol, and the standard gate
   chain (config/model/data gate -> real 1000-step train-step diagnostic ->
   three-poll GPU selection -> `launch_provenance.txt`).

### Stage R1-S0 — Baseline confirmation (conditional, 1 run)

Only if the epoch-28 checkpoint cannot be located: retrain Oriented R-CNN
R50 DIOR-R once under the exact existing protocol (52 epochs, val every 4)
and archive best/final. Otherwise skip.

### Stage R1-S1 — +TPC (taxonomy-prompt classifier)

- 3 replicas (seeds 3407/4407/5407), 12-epoch fine-tune from the baseline
  best checkpoint, val every 4, LR 2.5e-3 schedule as in the RoI
  Transformer stages.
- Acceptance: finite losses at iterations 200/1000, clean scoped failure
  scan; archive best and final means separately.

### Stage R1-S2 — +HRR (hierarchy relation regularization)

- 3 replicas, 12-epoch fine-tune from each replica's strongest R1-S1
  checkpoint; `hierarchy_loss_weight=0.05`,
  `hierarchy_target_self_weight=0.8` (unchanged from DIOR-R).

### Stage R1-S3 — +SCA (scene-context adapter)

- 3 replicas, 12-epoch fine-tune from the strongest R1-S2 replica
  checkpoint (mirroring the original DIOR-R S3 design).

### Pre-registered success criteria (decided before launch)

- **Generality confirmed:** every stage's best mean is >= the previous
  stage's best mean, and total best-mean gain over the ORCNN baseline is
  >= +2.0 points. Result enters the manuscript as a generality table/section
  (or the revision response).
- **Partial:** monotone ordering holds but total gain < +2.0 — report as
  supporting evidence with magnitudes stated honestly.
- **Negative:** any stage regresses below its predecessor — stop after that
  stage (no variant chasing), archive, and report the detector-dependence
  finding honestly; the RoI Transformer result stands on its own.

**Budget:** DIOR-R 12-epoch ORCNN run is roughly 8 GPU-hours on one
RTX 4090 (5,862 iters/epoch at batch 2). 9 runs across 3 GPUs ~= 3 days
wall-clock, plus 1-2 days for the port and gates.

## Route R2 — Backbone Generality Campaign (priority 2, after R1's port validates)

**Claim targeted:** removes the last comparator asymmetry — OrientedFormer's
best row uses Swin-T while ours uses R-50. GeoNexus on Swin-T plausibly
clears 70 AP50, which would be the headline of a revision or follow-up.

### Stages

- **R2-B0** RoI Transformer Swin-T DIOR-R baseline: 1 run, 52 epochs, val
  every 4 (identical protocol to the R50 baseline). Prerequisite: download
  and checksum the official Swin-T ImageNet-1k checkpoint; config gate on
  the backbone swap (mmdet Swin is available in the installed stack — the
  OrientedFormer reproduction already ran a Swin-T model in this
  environment).
- **R2-B1** +TPC, **R2-B2** +HRR, **R2-B3** +SCA: same 3-replica, 12-epoch
  fine-tune pattern and gates as R1.

**Success criterion:** best mean >= 70.0 AP50 is the headline outcome; any
result is publishable as a backbone-scaling row. Budget: ~4-5 days
wall-clock on 3 GPUs including the 52-epoch baseline.

## Sequencing

| # | Item | Depends on | GPU | Wall-clock |
|---|---|---|---|---|
| 1 | Submission finishers (Section 4, non-training) | — | none | days, human-paced |
| 2 | R1-P0 port + gates | — | none | 1-2 days |
| 3 | R1-S1..S3 campaign | R1-P0 | 3 GPUs | ~3 days |
| 4 | R2-B0 baseline | R1-P0 validated | 1 GPU | ~1.5 days |
| 5 | R2-B1..B3 campaign | R2-B0 | 3 GPUs | ~2.5 days |

R1/R2 do not block the TGRS submission — they are the revision arsenal and
the next paper cycle. Launch R1 whenever GPUs are free; the submission
proceeds in parallel.

## Section 4 — Non-training finishers and optional upgrades

1. **Advisor review pass** on the complete 8-page PDF, then author-list and
   affiliation finalization (single-author placeholder is still in place by
   explicit earlier decision).
2. **Submission package:** source + tables/figures + biography photo, cover
   letter, ORCID, arXiv decision.
3. *Optional figure upgrade:* re-render the qualitative strip focused on
   the confusion classes with the largest measured gains that the current
   scenes do not showcase (harbor +7.0, bridge +6.9, overpass +5.0) —
   `render_qualitative_detections_20260713.py` with harbor/bridge test
   scenes; the current airport/ESA strip is committed and already wired, so
   this is quality polish, not a blocker.
4. *Optional analyses (eval-only, non-blocking):* A4 prompt-template
   robustness (one Discussion sentence), A5 confusion matrix from the
   existing SCA preds.pkl, FAIR1M per-class S1-vs-S0 delta as supplementary
   material.
5. *Conditional only (currently closed, requires an explicit route
   reopening decision):* DOTA-v2.0 checkpoint-stabilization study (EMA /
   relation-aware LR) — named in the manuscript's Discussion as future
   work; run it only if a reviewer requests it, since DOTA-v2.0 follow-up
   training is closed by the 2026-07-21 manifest.

## Standing constraints

FAIR1M S2/GeoNexus, DIOR-R S4, segmentation, and DOTA-v2.0 follow-up
training remain closed. R1 and R2 are new route decisions on the DIOR-R
benchmark with the existing sanitized data and existing prompt artifacts;
they reuse the full gate discipline (config/model/data gate, 1000-step
finite diagnostic, three-poll GPU selection, per-workdir provenance, scoped
failure scans, 3-seed replicas, best/final reported separately) without
modification.
