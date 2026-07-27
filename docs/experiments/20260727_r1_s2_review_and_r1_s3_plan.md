# 2026-07-27 R1-S2 Review And R1-S3 Plan (for the server agent)

## Verdict

R1-S2 (Oriented R-CNN + TPC + HRR) on DIOR-R **succeeded and clears its
pre-registered criterion**. The R1 cross-detector generality campaign is now
two-for-two, cumulative **+3.46** over the ORCNN baseline with monotone
stages — well past the pre-registered ≥ +2.0 bar.

| Stage | Best mean AP50 | Final mean AP50 | Step gain (best) | Cumulative |
|---|---:|---:|---:|---:|
| ORCNN baseline (e28 best) | 63.41 | — | — | — |
| R1-S1 (+TPC) | 65.96 (all E12) | 65.96 | +2.55 | +2.55 |
| R1-S2 (+HRR) | **66.87** (66.77 E8 / 66.91 E8 / 66.93 E12) | 66.81 | **+0.91** | **+3.46** |

Pre-registered criterion "R1-S2 best mean ≥ R1-S1 best mean":
66.87 ≥ 65.96 ✓. Per-seed best-vs-best deltas are +0.59 / +0.84 / +1.29 —
all positive, well above the ~0.1–0.3 seed-noise scale, so the HRR gain on
ORCNN is real, not noise. It is smaller than the RoI Transformer HRR step
(+1.67) but the qualitative story matches (see per-class below).

## Per-class story (from the committed final-epoch CSVs, S2 − S1, 3-seed means)

Gains concentrate exactly where HRR is supposed to act — semantically
confusable and context-dependent classes, with sign-consistent per-seed
deltas: CH +5.20, TC +5.00, DAM +3.07, APO +2.30, ESA +2.27, HA +1.97,
TS +1.90, GF +1.77, BR +1.30. This mirrors the RoI Transformer HRR
per-class pattern, which strengthens the generality claim beyond the mean.
One notable regression: STO −4.93, but per-seed signs are inconsistent
(−/+/−, dominated by rep3407's −7.5) — recorded as a final-epoch
observation, not actionable. Note these CSVs are final-epoch (epoch-12)
tables; the extractor takes the last validation block.

## A0 finding worth keeping

R1-S1 best checkpoints are the **final** epoch (E12) for all three
replicas, and R1-S2 still improves at E12 for one replica. The
"peaks early, erodes late" heuristic from RoI Transformer / DOTA-v2.0 /
FAIR1M does **not** transfer to Oriented R-CNN fine-tuning — checkpoint
heuristics are detector-specific. Do not extend the schedule to chase this;
12 epochs stays the protocol.

## Flags (record-keeping, none result-invalidating)

1. **Source-checkpoint provenance gap.** Neither the 0726 launch note nor
   the 0727 completion record states *which* R1-S1 checkpoints seeded
   R1-S2 (the plan said "best per A0, else epoch-8 default"; A0 later
   showed best = E12). Append to `20260727_r1_s2_completion.md`: the exact
   source checkpoint path + SHA-256 per replica, runtime log paths, and a
   scoped failure-scan statement ("clean" or hits), per the
   experiment-completion-record conventions.
2. **Rounding split.** The completion note's final values (66.82 / 66.93
   for reps 4407/5407) differ from the per-class CSV mAP rows (66.81 /
   66.92) by the known evaluator-vs-parsed-table ±0.01. Cite the evaluator
   values in prose; do not "correct" the CSVs.

## Next training: R1-S3 (+SCA, scene-context adapter) — blocked on the port

The SCA head for Oriented R-CNN **still does not exist** (flagged 0722 as
N4, re-flagged 0723; nothing in the 0726–0727 commits adds it). Order of
work:

1. **Port the head (CPU).** Implement
   `SceneContextPromptRotatedShared2FCBBoxHead` for Oriented R-CNN:
   tile-descriptor pooling + adapter MLP + adapted-prototype cosine
   scoring, mirroring the existing cascade SCA head used in the RoI
   Transformer S3 runs. Same SCA hyperparameters as DIOR-R S3 — no
   retuning.
2. **Stage 3 replica configs** (seeds 3407/4407/5407), 12 epochs,
   val/ckpt interval 4, initialized from each replica's **best R1-S2
   checkpoint**:
   - rep3407 → epoch-8 (66.77)
   - rep4407 → epoch-8 (66.91)
   - rep5407 → epoch-12 (66.93)
   Record source-checkpoint SHA-256 in the gate report AND the launch
   provenance (closing the gap flagged above at the source).
3. **Full gate chain, no shortcuts:** CPU config/model/data gate for the
   new head → exact 1,000-step `train-step` diagnostic with
   `custom_hooks=[]` (diagnostic only) → three consecutive idle-GPU polls
   under elevated host access (sandbox hides `/dev/nvidia*`) → detached
   screens `geonexus_r1_s3_rep<seed>_gpu<N>_<date>` with
   `launch_provenance.txt` → startup acceptance (finite losses at 200 and
   1,000 + clean scoped scan).
4. **Success criterion (pre-registered):** R1-S3 best mean ≥ R1-S2 best
   mean (66.87). A regression ends the campaign at S2 — which already
   clears the R1 bar at +3.46 — with **no variant chasing**.

## After R1-S3 (either outcome)

Write the R1 campaign summary: one completion record with the full
generality table (baseline / S1 / S2 / S3; best AND final per replica;
per-class final-epoch tables committed as CSV/JSON), reconciled provenance,
and a closing decision boundary. That table is the manuscript's
revision-arsenal exhibit for cross-detector generality. **R2 (Swin-T
backbone) is not authorized by this note** — it needs its own dated route
decision after the R1 summary lands locally.

## Standing constraints (unchanged)

FAIR1M S2, DIOR-R S4 pseudo-labeling, segmentation lane, and DOTA-v2.0
follow-up training remain closed. R1-S3 is the authorized continuation of
the approved R1 route on existing DIOR-R data and artifacts. The paper
track is complete and independent (advisor review pending).

## Order of execution

1. Append provenance reconciliation to `20260727_r1_s2_completion.md`
   (analysis-only, minutes).
2. SCA head port + stage 3 configs + CPU gate (CPU, the critical path).
3. 1,000-step diagnostic → 3-poll (elevated host access) → launch 3
   R1-S3 replicas from the best R1-S2 checkpoints listed above.
4. R1-S3 completion record vs the pre-registered criterion → R1 campaign
   summary with the full generality table.
