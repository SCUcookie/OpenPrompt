# 2026-07-01 DIOR-R S4 LR5e-6 Complete

## Scope

Archive the completed DIOR-R S4 LR5e-6 micro-sweep launched on
`2026-06-30` from each S4 short-pack epoch-1 checkpoint.

Launch provenance:
`New/docs/experiments/20260630_dior_r_sota_audit_s4_micro_sweep_launch.md`.

Dataset/protocol: pseudo-label train root
`data/DIOR_R_dota_s4_pseudo_agreement_20260627/`, sanitized
`DIOR_R_dota/test`, MMRotate `DOTAMetric` mAP at IoU 0.5.

## Completion Status

All three accepted bootstrap runs completed through epoch 6 on
`2026-06-30 CST`. Each workdir contains `epoch_6.pth`, and each
`last_checkpoint` points to the matching final checkpoint.

Accepted training used `tools/bootstrap_run.py tools/train.py`, with launch
logs:

- `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr5e6_rep23407_20260630/launch_20260630_gpu0_bootstrap.log`
- `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr5e6_rep24407_20260630/launch_20260630_gpu4_bootstrap.log`
- `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr5e6_rep25407_20260630/launch_20260630_gpu5_bootstrap.log`

Archive verification on `2026-07-01 09:27 CST` found only
`s0_result_log_monitor_20260603` remaining in `screen`. GPUs 0, 4, and 5
were idle for paper-eval follow-up.

## Metrics

| Replica | Best epoch | Best `dota/mAP` | Best `dota/AP50` | Final epoch 6 `dota/mAP` | Final epoch 6 `dota/AP50` |
| --- | ---: | ---: | ---: | ---: | ---: |
| rep23407 | 6 | 0.696492 | 0.6960 | 0.696492 | 0.6960 |
| rep24407 | 2 | 0.696987 | 0.6970 | 0.692690 | 0.6930 |
| rep25407 | 2 | 0.698336 | 0.6980 | 0.692781 | 0.6930 |

Aggregate:

- Best mean mAP: `0.697272`.
- Final mean mAP: `0.693988`.

## Decision

Classify this as stabilization / negative-to-neutral evidence only, not
paper-facing S4 superiority.

The best mean `0.697272` remains below the S3 gate `0.6979`, and the best
single checkpoint `0.698336` remains below the single-run gate `0.7000`.
Pause further S4 training. Use `2026-07-01` for evaluation-only paper
artifacts on the best checkpoint from each replica.
