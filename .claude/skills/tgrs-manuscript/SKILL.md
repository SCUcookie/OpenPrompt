---
name: tgrs-manuscript
description: Workflow for editing, regenerating, and verifying the GeoNexus-RSD TGRS manuscript and its figures/tables. Use this skill whenever the user asks to polish the paper, update the manuscript, fix or add a table or figure, change reported numbers, recompile the PDF, respond to advisor/reviewer comments, or prepare the submission package — even if they only say "update the paper" or mention a specific table/figure without naming the manuscript. Also use it before touching anything under _local_archive_20260601_pull_backup/docs/TGRS/.
---

# TGRS Manuscript Workflow (GeoNexus-RSD)

The manuscript is the project's single most valuable artifact. Every rule
here exists because a violation has already caused rework once: hand-edited
tables drifted from the evidence, stale rounding survived three revisions,
and internal lab jargon leaked into a submitted-quality draft.

## File map

- Manuscript: `_local_archive_20260601_pull_backup/docs/TGRS/geonexus_tgrs.tex`
  — **local-only by policy, never committed to git** (excluded via
  `.git/info/exclude`). Do not add it to git or create a tracked copy.
- Bibliography: same dir, `geonexus_refs.bib`.
- Tables + figure fragments: same dir, `tables/` — generated files only.
- Figures: same dir, `figure/` (matplotlib PDFs/SVGs + TikZ architecture
  `geonexus_tgrs_architecture.tex`, compiled standalone).
- Generator: `scripts/make_tgrs_result_assets_<date>.py` (latest dated file
  wins; currently 20260713). Evidence sources it reads live in
  `docs/experiments/*.json` and are assert-guarded.

## Hard rules

1. **Tables come only from the generator.** Never hand-edit a file in
   `tables/`. To change a table, edit the generator and re-run it; the
   `assert_round` guards recompute every mean from the archived experiment
   JSONs and fail loudly if a number drifts.
2. **Every number traces to committed evidence.** Before writing any metric
   into prose, locate its JSON/note under `docs/experiments/`. If it is not
   recorded there, it does not go into the paper.
3. **Percent convention:** AP values as percentages with two decimals
   (69.79, not 0.6979). Known rounding trap: OrientedFormer evaluator value
   is 68.83; its per-class parsed-table mean is 68.84 — Table I uses 68.83,
   the per-class table uses 68.84 with a caption note. Keep that split.
4. **Banned vocabulary** (lab-log leakage): stage labels S0–S4 (use
   Baseline/TPC/HRR/SCA/PLP), "route", "gate", "closed", "paused",
   "negative-to-neutral", "claim boundary", seed names like rep3407
   (anonymize to Run 1–7), internal paths (`ss_val`, `DIOR_R_dota`,
   `work_dirs`), dates-as-status ("July 9 evidence state"), and
   contractions. Grep for these before every compile (word-boundary regex;
   "S2ANet" and "closed-set" are legitimate).
5. **Honest claims:** best-checkpoint and final-checkpoint means stay
   separate wherever selection matters; the pseudo-label negative result
   stays in the ablation; reproduced comparator rows carry a dagger and
   cited rows keep their fair-comparison footnote.

## Compile-and-verify chain (run after any edit)

```
cd _local_archive_20260601_pull_backup/docs/TGRS
pdflatex -interaction=nonstopmode geonexus_tgrs.tex
bibtex geonexus_tgrs
pdflatex -interaction=nonstopmode geonexus_tgrs.tex
pdflatex -interaction=nonstopmode geonexus_tgrs.tex
```

Then check, in order:
- zero "undefined" hits in the final `.log`;
- zero `[TO FILL]` markers in the `.tex` and `tables/*.tex` (unless a
  placeholder is intentionally open — say so explicitly);
- banned-token grep is clean;
- every canonical value appears consistently (abstract vs tables vs text);
- read the compiled PDF page by page with the Read tool for layout breaks —
  a clean compile does not mean a clean page.

The TikZ architecture figure compiles standalone the same way; always view
the produced PDF afterward, since TikZ overlaps never appear in the log.

## Figures

Follow the dataviz skill's validated palette already encoded in the
generator (CVD-checked hex values in the file header comment). New figures
go through the generator; the qualitative strip is rendered server-side
(`scripts/render_qualitative_detections_20260713.py`) and must be visually
curated before it replaces anything — check that the scenes actually show
the effect the caption claims.

## When numbers change upstream

New evidence arrives as dated JSON/notes in `docs/experiments/` via server
commits. The update path is always: evidence file → generator edit (new
dated copy if the change is structural) → regenerate → prose update →
compile chain → verification. Never shortcut from evidence directly to
prose.

## Pre-submission proofread checklist

<!-- Checks adapted 2026-07-22 from flonat/flonat-research skills/proofread
     (MIT, (c) 2026 Florian Burnat); reduced to the categories our other
     checks do not already cover. -->

Run these once the content is frozen, before the advisor/submission pass:

- **Notation consistency**: every math symbol defined once and used with
  one meaning ($\mathcal{C}$, $\bar{e}_c$, $\tilde{e}_{c,i}$, $R$, $\Omega$,
  $q_j$, $\lambda_{hier}$); no symbol collisions between Method equations.
- **Equation completeness**: each equation's variables all defined in the
  surrounding text; summation/index bounds stated; every equation
  referenced at least once from prose.
- **Causal-language audit**: "improves/causes/leads to" only where the
  ablation isolates the factor; correlational observations use
  "is associated with / coincides with". Our stability findings are
  observations, not mechanisms — keep them phrased that way.
- **Citation-voice balance**: no paragraph should be a bare citation list;
  each cited work gets a claim about its relation to ours.
- **Preprint staleness sweep**: for every `arXiv preprint` entry in the
  bib, check whether a published version now exists (this caught Strip
  R-CNN's AAAI 2026 upgrade once already; RTMDet and RiO-DETR are the
  current arXiv-only entries to re-check at submission time).
- **Report, don't edit**: proofread findings go into a dated review note
  first; edits are applied deliberately afterward, so the audit trail
  survives.
