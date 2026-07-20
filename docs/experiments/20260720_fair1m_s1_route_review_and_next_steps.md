# 2026-07-20 FAIR1M S1 Route Review And Detailed Next Steps

## Purpose

The FAIR1M TPC/S1 campaign completed (`20260720_fair1m_tpc_s1_completion.md`).
This note is the explicit route review it requested: it records the S1-vs-S0
comparison, defines the analysis jobs that produce the decision input, and
pre-registers the decision gate so the next launch (or non-launch) is
auditable. It also carries the unchanged paper-finalization jobs, which
remain the critical path. Execute the server jobs top to bottom; every job
before Phase D is analysis-only (no training, GPU only where stated).

## S1 vs S0 summary (the reason a gate is needed)

| Statistic | S0 baseline | S1 (TPC) | Delta |
|---|---:|---:|---:|
| Best-checkpoint mean mAP | `0.316867` (e8/e8/e8) | `0.318533` (e8/e4/e4) | `+0.0017` |
| Final epoch-12 mean mAP | `0.306567` | `0.306933` | `+0.0004` |
| Per-seed best deltas (3407/4407/5407) | -- | -- | `-0.0003 / +0.0014 / +0.0039` |

Reading: **flat at the mean level** — the gain is the same order as seed
noise and inconsistent in sign across seeds, in sharp contrast to DIOR-R S1
(`+0.0176`) and DOTA2 S1 (`+0.0089`). This is NOT yet a negative verdict:
the FAIR1M motivation is fine-grained sibling confusion (S0 class AP:
`c919=0.000`, `boeing777~0.02`, `arj21~0.04`, `tractor~0.02`), and a mean
over 37 classes can hide a real fine-grained effect in both directions. The
per-class delta analysis below is the decision input. Do not launch FAIR1M
S2/GeoNexus, and do not archive the FAIR1M route, before Phase D is applied.

Secondary observation to carry forward: S1 best checkpoints moved earlier
(e4/e4/e8 vs S0's e8/e8/e8) and final e12 regressed ~1 point below best in
both campaigns — the same early-peak/late-erosion pattern as DOTA2. This
motivates branch D-B below.

## Phase N — Analysis jobs (server, no training; N1 needs no GPU)

All commands run from `/data5/2025/ldh` unless stated. The extractor was
generalized on 2026-07-20: `New/scripts/extract_perclass_ap_20260713.py` now
takes `--dataset fair1m`, loading the 37 canonical classes and parent groups
from `New/assets/hierarchies/fair1m_remote_sensing_taxonomy.json` (digit-safe
name matching; tested locally on synthetic fixtures for both datasets).
Pull the latest `New` commit before running.

### N1. Per-class S1-vs-S0 extraction (CPU, ~minutes)

Six extractions. S1 best-checkpoint evaluator logs (paths from the
completion record; use the runtime log inside each timestamped dir):

```bash
cd /data5/2025/ldh
python New/scripts/extract_perclass_ap_20260713.py --dataset fair1m --tag s1_rep3407_e8 \
  --log OpenRSD/work_dirs/paper_eval_20260720/fair1m_tpc_s1_best_epoch8_rep3407/20260720_094824/20260720_094824.log
python New/scripts/extract_perclass_ap_20260713.py --dataset fair1m --tag s1_rep4407_e4 \
  --log OpenRSD/work_dirs/paper_eval_20260720/fair1m_tpc_s1_best_epoch4_rep4407/20260720_093654/20260720_093654.log
python New/scripts/extract_perclass_ap_20260713.py --dataset fair1m --tag s1_rep5407_e4 \
  --log OpenRSD/work_dirs/paper_eval_20260720/fair1m_tpc_s1_best_epoch4_rep5407/20260720_093654/20260720_093654.log
```

S0 epoch-8 best-checkpoint tables come from the three S0 training runtime
logs (the epoch-8 validation block). The extractor takes the LAST table in a
log, which for a full training log is epoch 12 — so either point it at a log
copy truncated after the epoch-8 block, or (simpler) run three eval-only
`tools/test.py` passes on the S0 `epoch_8.pth` checkpoints into
`OpenRSD/work_dirs/paper_eval_20260720/fair1m_s0_best_epoch8_rep{3407,4407,5407}`
(same bootstrap_run pattern as the S1 evals, one idle GPU each, ~15 min
total) and extract from those logs:

```bash
python New/scripts/extract_perclass_ap_20260713.py --dataset fair1m --tag s0_rep3407_e8 \
  --log OpenRSD/work_dirs/paper_eval_20260720/fair1m_s0_best_epoch8_rep3407/<ts>/<ts>.log
# ... likewise rep4407, rep5407
```

Then assemble the delta table (any Python; keep the JSONs):
per class, `delta = mean(S1 pct) - mean(S0 pct)`; report the six JSONs, the
per-class delta column, the parent-group deltas (airplane/ship/vehicle/
court/road, printed automatically by the extractor), and the delta grouped
by S0 difficulty tier (near-zero `<5`, low `5-20`, mid `20-50`, high `>50`).

Acceptance: extracted mAPs reproduce the known values at rounding —
S1 `31.75/31.79/32.02`; S0 e8 `31.78/31.65/31.63`.

### N2. Rare-class instance audit (CPU, ~minutes)

Count ss_val instances per class in the sanitized validation annotations:

```bash
cd /data2/2023/lcs/xyun/FAIR1M_2_800_400_sanitized_20260713/ss_val
awk '{print $(NF-1)}' <annotation-dir>/*.txt | sort | uniq -c | sort -rn
```

(adjust `<annotation-dir>` to the actual labelTxt directory name in that
root). Specifically resolve whether `c919` has ~0 validation instances —
`0.000` AP in all six runs suggests a structural absence, not a model
failure. If so, record that mean mAP over 37 classes is deflated by
empty/near-empty classes and that any future FAIR1M reporting should also
state mAP over classes with at least K (e.g., 20) validation instances.

### N3. Honest significance record

In the analysis note (see Bookkeeping), record the paired per-seed best
deltas `{-0.0003, +0.0014, +0.0039}` and state plainly that the mean-level
S1 gain on FAIR1M is not distinguishable from seed noise at n=3. No spin in
either direction; the per-class result is the substantive finding.

## Phase D — Pre-registered route decision gate

Apply exactly one branch after N1-N3; record the measured values next to the
thresholds in the analysis note.

**D-A. Fine-grained lift confirmed -> launch FAIR1M HRR/S2.**
Trigger: airplane-group mean best-checkpoint delta `>= +1.0` point, OR at
least 7 of the near-zero/low-tier classes improve by `>= +0.5` point each.
Action: 3-replica, 12-epoch HRR/S2 campaign from the best S1 checkpoints
(rep3407 e8, rep4407 e4, rep5407 e4), hierarchy relation matrix from the
canonical taxonomy, `hierarchy_loss_weight=0.05`,
`hierarchy_target_self_weight=0.8` (the DIOR-R settings), same gate protocol
as S1: config/model gate, real 1000-step train-step diagnostic, three-poll
GPU selection, `launch_provenance.txt` per workdir. Rationale: on DIOR-R the
relation loss added `+1.67` on top of prompts; FAIR1M's explicit 5-parent /
37-child taxonomy is the structure HRR was designed for.

**D-B. Flat everywhere -> one controlled variant pack, no blind S2.**
Trigger: neither D-A condition met and no parent group moved by `>= 0.5` in
either direction.
Action (exactly one pack, hard cap — lesson from the DOTA2 S2/S3 sweeps):
**S1-v2 low-LR fine-tune from the S0 epoch-8 best checkpoints** (not the
regressed e12 finals): LR `5e-4`, 8 epochs, validation every 2 epochs,
3 replicas, same gates. This tests the erosion hypothesis directly (both
campaigns peak at e4-e8 and lose ~1 point by e12). Optional pairing, before
or with the pack: upgrade the FAIR1M prompt bank's fine-grained descriptors
(the taxonomy JSON's synonyms/geometry cues are an unreviewed first draft
per its own `_provenance` note; aircraft subtypes need discriminative cues —
engine count, fuselage length class, wingtip shape), regenerate the
canonical embeddings, and note the artifact version in the launch record.
If S1-v2 is also flat, apply D-C without further variants.

**D-C. Negative/flat after D-B -> archive and refocus.**
Archive FAIR1M S1 (and S1-v2 if run) as neutral stretch evidence with best
and final means kept separate. FAIR1M stays out of the TGRS manuscript
(already future-work only — no manuscript change needed). All effort moves
to Phase P. This is a publishable end-state: the paper's claims rest on
DIOR-R and DOTA-v2.0, and the honest FAIR1M analysis strengthens the
stability-methodology story rather than weakening the paper.

## Phase P — Paper finalization (critical path, independent of Phase D)

P1. **Job A1 — DIOR-R per-class extraction (STILL the only paper-blocking
item; CPU-only).** Three runs with the same (default-dataset) extractor:

```bash
cd /data5/2025/ldh
python New/scripts/extract_perclass_ap_20260713.py --tag baseline_e52 \
  --log <paper_eval_20260617 S0-epoch52 runtime log>
python New/scripts/extract_perclass_ap_20260713.py --tag geonexus_sca_rep0_e8 \
  --log <paper_eval_20260617 S3-rep0-epoch8 runtime log>
python New/scripts/extract_perclass_ap_20260713.py --tag orientedformer_swint \
  --log /data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_eval_20260704_rerun/20260704_105235/20260704_105235.log
```

Acceptance: mAPs reproduce `65.44 / 69.92 / 68.83`. Push the three JSONs +
printed LaTeX rows; they drop into manuscript Table V.

P2. **Transfer A2/A3 artifacts into the repo:** copy
`OpenRSD/work_dirs/paper_analysis_20260713/efficiency_baseline.json`,
`efficiency_geonexus.json` (55.39M/19.13FPS vs 58.31M/18.96FPS), and the
qualitative strip into a Git-tracked location (e.g.,
`New/artifacts/paper_analysis_20260713/`) and push. Before the strip enters
the manuscript, eyeball the four scenes — the image IDs were schedule
placeholders; re-render with better scenes if no baseline confusions are
visible.

P3. **FLOPs cell:** both efficiency runs returned FLOPs `None` with a
warning. Either rerun with a fixed `get_model_complexity_info` input path
on the server, or drop the GFLOPs column from Table VI (params + FPS
already carry the efficiency claim: `+2.9M` params, `<1%` FPS cost).

P4. Local (after P1-P3 results are pulled): fill Tables V/VI, replace the
Fig. 4 placeholder, recompile (`pdflatex -> bibtex -> pdflatex x2`), re-run
the number audit, then the advisor review pass (author list is still the
single-author placeholder).

## Bookkeeping

- Record N1-N3 results and the applied D-branch in a new dated note
  `20260720_fair1m_s1_vs_s0_perclass_analysis.md` (thresholds quoted
  verbatim next to measured values so the decision is auditable), and add a
  short pointer entry to `PROJECT_INSTRUCTIONS.md`.
- Update the Status Tracking table in
  `20260713_paper_finalization_schedule.md` (A2/A3 complete, A1 pending,
  FAIR1M gate executed through S1).
- Standing pauses unchanged: DOTA-v2.0 follow-up training, DIOR-R S4
  pseudo-labeling, segmentation lane, routing — none of them are reopened
  by this note.

## Execution checklist (server, in order)

1. `git pull` in `New` (brings the generalized extractor and this note).
2. N1 S0-e8 eval passes (3 short GPU jobs) + 6 extractions + delta table.
3. N2 instance count; N3 noted.
4. Apply Phase D branch; if D-A/D-B, run the full launch-gate protocol
   before any training starts.
5. P1 (A1) extractions; P2 artifact copy; P3 FLOPs decision.
6. Commit results (JSONs, delta table, analysis note) and push.
