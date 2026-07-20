# FAIR1M Route Review And Next Analysis - 2026-07-20

## Decision

FAIR1M S2/GeoNexus is closed. The completed FAIR1M TPC/S1 campaign gives a
best-checkpoint mean of `0.318533` mAP / `0.318333` AP50, versus the FAIR1M S0
epoch-8 mean of `0.316867` mAP. The gain is marginal (`+0.001666` mAP), while
the TPC/S1 epoch-12 mean falls to `0.306933` mAP / `0.306667` AP50. This does
not justify another FAIR1M training stage or a new route claim.

FAIR1M remains supplementary stretch evidence only. The paper route stays:

- DOTA2: S1 as the stable positive result, with S2 loss-0 early-checkpoint
  evidence reported separately.
- DIOR-R: S2 as the stable hierarchy result, with S3 best-checkpoint scene
  context evidence and its weaker final-checkpoint behavior reported
  separately.
- FAIR1M: archive the S0/TPC-S1 result as supplementary evidence; do not train
  S2/GeoNexus, relaunch DIOR-R S4, start segmentation, or run DOTA2 follow-up
  training from this review.

## FAIR1M Evidence

The source completion records are
`docs/experiments/20260717_fair1m_s0_completion.md` and
`docs/experiments/20260720_fair1m_tpc_s1_completion.md`. The S0
best-checkpoint mean is `0.316867` mAP at epoch 8. TPC/S1 selected checkpoints
and metrics are:

| Replica | Checkpoint | mAP / AP50 |
|---|---|---:|
| rep3407 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_fair1m/roi_trans_tpc_s1_rep3407_20260717/epoch_8.pth` | `0.3175 / 0.3170` |
| rep4407 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_fair1m/roi_trans_tpc_s1_rep4407_20260717/epoch_4.pth` | `0.3179 / 0.3180` |
| rep5407 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_fair1m/roi_trans_tpc_s1_rep5407_20260717/epoch_4.pth` | `0.3202 / 0.3200` |

The matching analysis outputs are under
`/data5/2025/ldh/OpenRSD/work_dirs/paper_eval_20260720/fair1m_tpc_s1_best_epoch8_rep3407`,
`.../fair1m_tpc_s1_best_epoch4_rep4407`, and
`.../fair1m_tpc_s1_best_epoch4_rep5407`. Their evaluator JSON values are
`0.3174866736/0.317`, `0.3178664148/0.318`, and `0.3202100992/0.320`,
respectively, and reconcile with the training logs at evaluator rounding.
All final evaluator logs reached `Epoch(test) [5485/5485]` and wrote
`preds.pkl`.

The S0 comparator is archived in
`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_fair1m/roi_trans_s0_rep{3407,4407,5407}_20260715`;
its epoch-8 values are `0.3178`, `0.3165`, and `0.3163` mAP. Training and
evaluator logs passed scoped traceback, CUDA OOM, decode/CRC, invalid-box,
NaN/Inf, and interruption scans. Obsolete-environment and mistyped-config
attempts exited before model load and are not experiment failures.

## Existing Paper Analysis

The approved analysis-only A1-A3 jobs from
`docs/experiments/20260713_paper_finalization_schedule.md` are complete; no
new GPU job is required by this review.

### A1: Per-Class AP50

The stdlib parser `scripts/extract_perclass_ap_20260713.py` parsed all 20
classes from these exact runtime logs:

| Method | Runtime log | Parsed mAP/AP50 |
|---|---|---:|
| DIOR-R baseline epoch 52 | `/data5/2025/ldh/OpenRSD/work_dirs/paper_eval_20260617/dior_r_s0_roi_trans_epoch52/20260617_092751/20260617_092751.log` | `65.44` |
| GeoNexus SCA rep0 epoch 8 | `/data5/2025/ldh/OpenRSD/work_dirs/paper_eval_20260617/dior_r_s3_rep0_epoch8/20260617_092800/20260617_092800.log` | `69.92` |
| OrientedFormer Swin-T | `/data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_eval_20260704_rerun/20260704_105235/20260704_105235.log` | `68.84` table mean; runtime `0.6883/0.6880` |

The extracted JSON/CSV records are
`20260720_dior_r_perclass_baseline_e52.{json,csv}`,
`20260720_dior_r_perclass_geonexus_sca_rep0_e8.{json,csv}`, and
`20260720_dior_r_perclass_orientedformer_swint.{json,csv}` in this directory.
The old schedule target `68.83` for OrientedFormer is stale by one hundredth;
use the exact runtime metric and parsed table consistently in the manuscript.

### A2: Qualitative Strip

The completed output is
`/data5/2025/ldh/OpenRSD/work_dirs/paper_analysis_20260713/qualitative/geonexus_tgrs_qualitative.png`.
It is a readable RGB PNG of `2048x1024`, with baseline and GeoNexus rows for
curated images `11726`, `12003`, `14830`, and `17650`. The strip uses shared
class colors and the successful launch log is
`/data5/2025/ldh/OpenRSD/work_dirs/paper_analysis_20260713/logs/qualitative_gpu4.launch.log`.

### A3: Efficiency

Both measurements used an NVIDIA GeForce RTX 4090 and 200 timing runs:

| Method | Parameters | Median FPS | Median latency | GFLOPs |
|---|---:|---:|---:|---|
| Baseline | `55.39M` | `19.13` | `52.27 ms` | unavailable; JSON records warning/null |
| GeoNexus SCA | `58.31M` | `18.96` | `52.75 ms` | unavailable; JSON records warning/null |

Sources are `/data5/2025/ldh/OpenRSD/work_dirs/paper_analysis_20260713/efficiency_baseline.json`
and `efficiency_geonexus.json`. The launch record is
`docs/experiments/20260713_phase_a_three_gpu_launch.md`; its scoped logs are
free of traceback, OOM, decode/CRC, and NaN/Inf signatures. Report the GFLOPs
fallback explicitly rather than inventing a value.

## Manuscript Boundary

The TGRS directory is local-only and remains untracked by project policy. A
manuscript asset update is justified for A1-A3, but this sandbox exposes no
writable local TGRS copy. Do not broaden the edit to another paper directory.
The source analysis outputs, exact evidence paths, and this route decision are
archived here for the next host-side manuscript refresh. Before compilation,
replace the A1/A3 placeholders, include the A2 strip, use `68.84` or the
runtime-consistent rounded value for OrientedFormer, then run the manuscript
number-consistency audit and verify there are no undefined references or
remaining `[TO FILL]` entries.
