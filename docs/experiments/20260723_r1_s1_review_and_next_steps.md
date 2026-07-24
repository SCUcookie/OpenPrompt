# 2026-07-23 R1-S1 Review And Next Steps (for the server agent)

## Verdict

R1-S1 (Oriented R-CNN + taxonomy-prompt classifier, TPC) on DIOR-R
**succeeded** and is a strong cross-detector generality signal. This is the
first evidence that the prompt head transfers off the RoI Transformer.

| | ORCNN baseline (epoch-28 best) | R1-S1 final mean (3 reps) | Gain |
|---|---:|---:|---:|
| DIOR-R AP50 | 63.41 | **65.96** (66.18 / 66.07 / 65.64, std 0.23) | **+2.55** |

For reference, on RoI Transformer the same S0→S1 step was 65.44 → 67.20
(+1.76). The ORCNN gain is *larger*, and note it is measured final-vs-best
(the baseline value is a best checkpoint, the S1 values are final epoch 12),
so the true best-vs-best gain is very likely higher still. The
pre-registered R1 criterion (total best-mean gain over baseline ≥ +2.0,
monotone stages) is already cleared by stage 1 alone. Proceed to R1-S2.

Provenance reconciles: seeds 3407/4407/5407 on physical GPUs 2/3/4, screens
`geonexus_r1_s1_rep{3407,4407,5407}_gpu{2,3,4}_20260723`, epoch-28 source
checkpoint (SHA-verified in the 0722 gate), clean scoped scans, per-run
`launch_provenance.txt`. The 0722 GPU failure was sandbox device isolation
(no `/dev/nvidia*` nodes in the ordinary sandbox); elevated host access
exposes all seven RTX 4090s — record this so it is not rediscovered.

## One documentation gap to close (analysis-only, no GPU)

The completion record reports only the **final** epoch-12 metrics. The
project always reports best AND final checkpoints separately, and a future
R1 generality table (and any manuscript revision) needs both plus per-class
AP. All inputs already exist on disk.

**A0 — extract R1-S1 best-checkpoint values and per-class AP.** With
validation interval 4 the runs have epoch-4/8/12 validations; parse the best
epoch per replica from the runtime logs, and extract the per-class table:

```bash
cd /data5/2025/ldh
for s in 3407 4407 5407; do
  python New/scripts/extract_perclass_ap_20260713.py --tag r1_s1_rep${s} \
    --log OpenRSD/work_dirs/geonexus_dior_r/orcnn_tpc_s1_r1_rep${s}_20260722/<ts>/<ts>.log
done
```

(the extractor takes the LAST table = epoch 12; for the best epoch, point it
at a log truncated after that epoch's validation block, or read the epoch-4/8
val lines directly). Record best mean/std alongside the final mean/std in a
completion record `20260723_r1_s1_completion.md` following the standard
conventions. This is not a launch blocker for R1-S2 — do it in parallel.

## Next training: R1-S2 (+HRR, hierarchy relation regularization)

R1-S1 cleared the bar, so launch R1-S2 from each replica's **strongest**
R1-S1 checkpoint (use A0's best epochs; if A0 is not yet done, epoch-8
checkpoints are the safe default given the project's early-peak pattern).

- Configs: the R1-S2 replica configs are already staged (0722 port status).
- Head: `HierarchyPromptRotatedShared2FCBBoxHead` (already implemented).
- Settings: `hierarchy_loss_weight=0.05`, `hierarchy_target_self_weight=0.8`
  (unchanged from DIOR-R), 12 epochs, val/ckpt interval 4, 3 seeds.
- Artifact: the DIOR-R hierarchy prompt artifact with the `[20,20]`
  relation matrix (verified `[20,512]` embeddings + `[20,20]` matrix in the
  0722 CPU gate) — no new relation artifact needed.
- **Full gate chain, no shortcuts:** repeat the CPU config/model/data gate
  for the S2 head, then the exact 1,000-step `train-step` diagnostic with
  `custom_hooks=[]` (diagnostic only — training configs keep EMA), then
  three consecutive idle-GPU polls under elevated host access, then detached
  screens with `launch_provenance.txt`. Startup acceptance = finite losses at
  iterations 200 and 1000 + clean scoped scan.
- Success criterion (pre-registered): R1-S2 best mean ≥ R1-S1 best mean.
  A regression stops the campaign at S2 with no variant chasing.

## Parallel prerequisite for R1-S3: port the SCA head (CPU, do now)

R1-S3 (+SCA, scene-context adapter) **cannot launch yet** — only the S1 and
S2 heads exist. Port `SceneContextPromptRotatedShared2FCBBoxHead` for
Oriented R-CNN now (tile-descriptor pooling + adapter MLP + adapted-prototype
cosine scoring, mirroring the cascade SCA head), stage R1-S3 replica configs,
and run the CPU gate — so R1-S3 launches the day R1-S2 completes instead of
losing a porting day mid-campaign. This is the same N4 item flagged on 0722;
it was not addressed by the R1-S1 launch.

## Standing constraints (unchanged)

FAIR1M S2, DIOR-R S4 pseudo-labeling, segmentation lane, and DOTA-v2.0
follow-up training remain closed. R1-S2/S3 are the authorized continuation of
the already-approved R1 route on the existing DIOR-R data and artifacts. The
paper track is complete and independent (advisor review pending); R1 results
are the revision arsenal and do not block submission.

## Order of execution

1. R1-S2 CPU gate → 1000-step diagnostic (elevated host access) → 3-poll →
   launch 3 replicas from best R1-S1 checkpoints.
2. In parallel: A0 R1-S1 best/per-class extraction + completion record.
3. In parallel: SCA-head port + R1-S3 CPU gate.
4. R1-S2 completion record vs the pre-registered criterion → R1-S3 launch
   once both R1-S2 completes and the SCA head lands.
