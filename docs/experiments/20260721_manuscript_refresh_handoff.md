# Manuscript Refresh Evidence Handoff

Date: 2026-07-21

This is a portable evidence handoff for the local-only TGRS manuscript. The
canonical manuscript remains outside this repository; this note does not add
or replace that directory.

## A1 Per-Class Evidence

The three committed records are the source of truth for Table V:

| Record | Classes | mAP (evaluator) | Parsed-table mean |
| --- | ---: | ---: | ---: |
| `20260720_dior_r_perclass_baseline_e52.{json,csv}` | 20 | 65.44 | 65.44 |
| `20260720_dior_r_perclass_geonexus_sca_rep0_e8.{json,csv}` | 20 | 69.92 | 69.92 |
| `20260720_dior_r_perclass_orientedformer_swint.{json,csv}` | 20 | 68.83 | 68.84 |

Use the validated parsed-table value `68.84` for the current manuscript
refresh handoff when reproducing the per-class table. Retain the historical
schedule/evaluator value `68.83` in the dated schedule and provenance notes;
the 0.01 difference is caused by averaging rounded per-class values.

The current claim boundary is measured DIOR-R evidence only: GeoNexus SCA
improves over the baseline on airport (+9.5), airplane (+8.9), ESA (+7.6),
harbor (+7.0), and bridge (+6.9), while ship is unchanged (+0.0) and tennis
court changes by +0.4 percentage points.

## A2 Qualitative Figure

The remaining manuscript asset is the host-local file
`/data5/2025/ldh/OpenRSD/work_dirs/paper_analysis_20260713/qualitative/geonexus_tgrs_qualitative.png`
(reported as 2048x1024 in the route note). It must be transferred and
visually curated before replacing the single Fig. 4 placeholder. This path is
host-local evidence, not a tracked repository reference.

## A3 Efficiency Evidence

Table VI is populated in the local manuscript with the reported RTX 4090
measurements: baseline 55.39M parameters, 19.13 FPS, 52.27 ms; GeoNexus SCA
58.31M, 18.96 FPS, 52.75 ms. GFLOPs remain unreported because the analysis
was unavailable. The GPU model and missing-FLOPs boundary must remain in the
table caption.

## Consistency Checklist

- [x] All three A1 JSON/CSV pairs are committed and each contains 20 classes.
- [x] JSON and CSV mAP values agree within the record precision.
- [x] Table V uses 65.44 / 69.92 / 68.84 for the current parsed-table handoff.
- [x] Historical `68.83` remains preserved in the dated schedule/provenance.
- [x] Table VI retains the measured parameter, FPS, latency, and RTX 4090 values.
- [x] GFLOPs are explicitly unavailable rather than inferred.
- [x] Transfer and curate the host-local Fig. 4 strip (done 2026-07-21: the
      committed bundle copy was placed at the manuscript's
      `figure/geonexus_tgrs_qualitative.png`; visual curation found the
      airport and expressway-service-area scenes align with the two largest
      per-class gains, and the caption was rewritten to describe what the
      figure actually shows; an optional stronger re-render focused on
      harbor/bridge confusion scenes is noted in the 2026-07-21 route plan).
- [x] Compile the canonical local manuscript after the Fig. 4 update (done
      2026-07-21: 8 pages, zero undefined references, zero `[TO FILL]`
      markers; Table V uses 65.44/68.84/69.92 per this handoff with the
      68.83/68.84 rounding note in the caption).
- [x] No GPU training, checkpoint movement, FAIR1M route expansion, or TGRS
      replacement tree was started by this handoff.
