# 2026-07-13 Paper Finalization Schedule And Forward Experiment Plan

## Purpose

The TGRS manuscript was rewritten to submission voice on 2026-07-13 (local
`_local_archive_20260601_pull_backup/docs/TGRS/geonexus_tgrs.tex`, 7 pages,
clean compile). Three evidence slots remain marked `[TO FILL]` and can only
be produced on the server. This note is the complete forward schedule:
Phase A closes the manuscript's remaining slots (analysis-only, no
training), Phase B is the already-approved FAIR1M S0 gate, Phase C is
submission logistics.

Standing rule: DOTA-v2.0 follow-up **training** stays paused. Phase A
contains no training at all; Phase B follows the 2026-07-10 handoff gate
exactly. The DIOR-R S4 pseudo-label route remains closed (best mean
`0.697272` below the S3 gate `0.6979`); it appears in the manuscript as the
honest negative ablation and must not be relaunched.

Component-name mapping used by the manuscript (never rename these back):
S0 = Baseline, S1 = TPC (taxonomy-prompt classifier), S2 = HRR (hierarchy
relation regularization), S3 = SCA (scene-context adapter), S4 = PLP
(pseudo-label purification, negative ablation). DOTA2 stability runs are
anonymized Run 1-7 in the paper; the seed mapping lives only in
`scripts/make_tgrs_result_assets_20260713.py`.

## Phase A - Paper-blocking analysis jobs (server, no training, ~half a day)

All jobs run from `/data5/2025/ldh/OpenRSD` in the usual MMRotate env
(`/data1/anaconda3/envs/zwl_mmrotate/bin/python` via `tools/bootstrap_run.py`
where a config is loaded). GPU jobs need one idle GPU (three-poll rule).

### A1. Per-class AP50 extraction -> manuscript Table V

- Script: `New/scripts/extract_perclass_ap_20260713.py` (stdlib-only; parses
  the last DOTAMetric per-class table from an evaluation runtime log).
- Inputs (three runs, all already evaluated - reuse the existing logs; only
  rerun `tools/test.py` if a log lacks the per-class table):
  1. DIOR-R Baseline: RoI Transformer epoch-52 paper-eval log under
     `/data5/2025/ldh/OpenRSD/work_dirs/paper_eval_20260617/` (S0 epoch52 run).
  2. GeoNexus-RSD best run: SCA rep0 epoch-8 paper-eval log under the same
     `paper_eval_20260617` tree (S3 rep0 epoch8 run).
  3. OrientedFormer reproduction: runtime log
     `/data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_eval_20260704_rerun/20260704_105235/20260704_105235.log`.
- Command per run:
  `python New/scripts/extract_perclass_ap_20260713.py --log <log> --tag <baseline_e52|geonexus_sca_rep0_e8|orientedformer_swint>`
- Acceptance: 20/20 classes parsed; the printed mAP must reproduce the known
  values (65.44 / 69.92 / 68.83) within rounding.
- Destination: paste the three LaTeX rows into
  `tables/table_perclass.tex` (or re-run the 20260713 generator after adding
  the JSONs to `docs/experiments/`), then update the qualitative sentence in
  Section IV-G against the real numbers.

### A2. Qualitative comparison strip -> manuscript Fig. 4

- Script: `New/scripts/render_qualitative_detections_20260713.py` (needs the
  MMRotate env and one idle GPU).
- Inputs: DIOR-R baseline config+epoch-52 ckpt; SCA rep0 config+epoch-8 ckpt
  (both under the `geonexus_dior_r` workdirs recorded in the 20260615 S3
  completion note); four curated DIOR-R test images (harbor, dense vehicles,
  overpass/bridge, storage tanks) - inspect candidates first and pick scenes
  with visible baseline confusions.
- Acceptance: stitched `geonexus_tgrs_qualitative.png` with two rows, shared
  class colors, readable labels at print size.
- Destination: copy to the local TGRS `figure/` dir, replace the placeholder
  `\fbox` block in Section IV-H with an `\includegraphics`, recompile.

### A3. Efficiency measurement -> manuscript Table VI

- Script: `New/scripts/measure_efficiency_20260713.py`, run twice
  (baseline and GeoNexus-RSD configs/ckpts as in A2) on one idle GPU.
- Acceptance: JSON with params_M, gflops (or a note that FLOPs analysis
  failed and a manual count is needed), fps_median, and the GPU name.
- Destination: fill `tables/table_efficiency.tex`; add the GPU model to the
  table caption.

### A4. Prompt-robustness check (optional, strengthens Discussion)

- Regenerate the DIOR-R prompt-embedding artifact with an alternative
  template set (swap templates in the generation script that produced
  `remoteclip_vit_b32_dior_r_s2_hierarchy_prompt_embeddings.pt`), then
  eval-only `tools/test.py` on the SCA rep0 epoch-8 checkpoint with the
  swapped artifact. No training.
- Acceptance: one AP50 number per template variant.
- Destination: one Discussion sentence ("results are robust to prompt
  template choice, varying by less than X points"); add a small table only
  if the spread is interesting.

### A5. Confusion matrix (optional Fig. 5)

- From the existing SCA rep0 `preds.pkl` (paper-eval tree) and the DIOR-R
  test annotations, compute a 20x20 confusion matrix at IoU 0.5 and render
  with the manuscript palette. Only add to the paper if it visibly supports
  the taxonomy-confusion claim; otherwise keep as reviewer-response material.

## Phase B - FAIR1M S0 gate (unchanged from the 2026-07-10 handoff)

Execute steps 1-6 of `docs/experiments/20260710_fair1m_sanitized_staging_handoff.md`
verbatim (summarized; the handoff note is authoritative):

1. Download + sha256 the official ResNet-50 init
   (`checkpoints/pretrained/resnet50-0676ba61.pth`).
2. Full active-split geometry/rbox scan, exact stem comparison, decode checks
   (533/800/1024), config load, dataloader construction, one full batch.
3. Three idle-GPU polls (<1 GB, 0% util) with dynamic selection.
4. Detached 1,000-step train diagnostic with
   `tools/diagnose_first_nonfinite_loss.py` on
   `G02_Baselines_Data3_FAIR1M_M2_RoITrans_S0_Sanitized_20260710.py`;
   proceed only on `checked_batches == 1000`, `finite_within_limit`, clean scan.
5. One-epoch smoke; accept startup only after iteration 200 + clean scan;
   stop after epoch-1 validation with a finite nonzero metric; do NOT
   auto-launch 12 epochs.
6. Dated provenance note with the full operation trail.

Gate outcome decides the stretch goal: a full FAIR1M baseline schedule and a
TPC (taxonomy-prompt) pass using the staged canonical
`remoteclip_vit_b32_fair1m_prompt_embeddings_canonical.pt` (`[37, 512]`).
FAIR1M results are **not** required for the current TGRS submission; the
manuscript cites FAIR1M only as future work.

## Phase C - Submission logistics (local, after A1-A3 land)

1. Re-run `python scripts/make_tgrs_result_assets_20260713.py` after any
   table change; never hand-edit generated tables.
2. Number-consistency audit: every value in the abstract, tables, and text
   cross-checked against `tables/README_tgrs_result_assets_20260713.md`.
3. Update the two qualitative-claim sentences (Sections IV-G and IV-H)
   against the real per-class numbers and figure content.
4. Advisor review pass (author list is still the single-author placeholder;
   finalize authors/affiliations/corresponding author before submission).
5. TGRS package: final PDF + source (.tex, .bib, tables/, figure/ PDFs,
   biography photo), cover letter, ORCID; decide on an arXiv preprint.

## Status Tracking

| Job | Blocking | Status |
| --- | --- | --- |
| A1 per-class AP | Table V | pending server |
| A2 qualitative strip | Fig. 4 | pending server |
| A3 efficiency | Table VI | pending server |
| A4 prompt robustness | optional | pending server |
| A5 confusion matrix | optional | pending server |
| B FAIR1M gate steps 1-6 | future work only | pending server |
| C submission package | final step | blocked on A1-A3 |
