# FAIR1M S0 Campaign Partial Results (2026-07-17)

This is a partial results record aligned with the launch provenance in
`20260715_fair1m_s0_campaign_launch.md`. Replicas 3407 and 4407 completed
epoch-12 validation cleanly. Replica 5407 was still training epoch 12 when
this record was written; its final checkpoint and final validation result are
not yet available.

## Verified validation results

| Seed | GPU / screen | Epoch 4 mAP / AP50 | Epoch 8 mAP / AP50 | Epoch 12 mAP / AP50 | Status |
|---:|---|---|---|---|---|
| 3407 | GPU 0 / `fair1m_s0_rep3407_20260715_gpu0` | `0.3106 / 0.3110` | `0.3178 / 0.3180` | `0.3045 / 0.3040` | complete |
| 4407 | GPU 1 / `fair1m_s0_rep4407_20260715_gpu1` | `0.3058 / 0.3060` | `0.3165 / 0.3160` | `0.3043 / 0.3040` | complete |
| 5407 | GPU 2 / `fair1m_s0_rep5407_20260715_gpu2` | `0.3143 / 0.3140` | `0.3163 / 0.3160` | pending | training epoch 12 |

Across the three available checkpoints, epoch-4 mAP mean/std is
`0.310233 / 0.003480`; epoch-8 mAP mean/std is `0.316867 / 0.000665`.
These are interim statistics only because rep5407 epoch-12 validation is
missing. Among the two completed replicas, epoch-12 mAP is `0.3045` and
`0.3043`.

## Provenance and checks

- Launch record: `20260715_fair1m_s0_campaign_launch.md`.
- Configs, workdirs, source checkpoint, and launch commands are unchanged from
  that record; each workdir retains `launch_provenance.txt`.
- Completed checkpoint paths: `OpenRSD/work_dirs/geonexus_fair1m/roi_trans_s0_rep3407_20260715/epoch_{4,8,12}.pth` and the corresponding `rep4407` paths.
- Rep5407 latest marker at `2026-07-17T09:03:28+08:00`: `Epoch(train) [12][1700/66467]`; only `epoch_4.pth` and `epoch_8.pth` exist.
- Scoped scans of the three logs found no traceback, CUDA OOM, decode/CRC,
  invalid-box, NaN/Inf, or keyboard-interruption signatures.

## Decision state

Do not promote the FAIR1M S0 campaign to a final result or launch FAIR1M
TPC/GeoNexus from this partial record. First wait for rep5407 epoch-12
validation, complete the three-replica mean/std and best-versus-final
comparison, and make an explicit route decision with the exact next config.
