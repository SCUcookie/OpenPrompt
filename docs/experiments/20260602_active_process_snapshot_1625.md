# 2026-06-02 Active Process Snapshot 16:25

## Scope

This is an unarchived live-status record for the active GeoNexus/OpenRSD
experiment processes on the server. It records process state, latest observed
log progress, and rough finish estimates. It is not a metric archive except
where a final validation line is explicitly noted.

## Screen And GPU State

Checked at `2026-06-02 16:25 +0800`.

Active experiment screens:

- `1598562.s0_dota2_orcnn_r50_validpng_20260602_gpu1`
- `1077110.s0_dota2_roi_trans_rebuild_validpng_20260602_gpu0`
- `425331.geonexus_s3_scene_adapter_144e`
- `3891792.geonexus_s2_hierarchy_reg_144e`

Our active experiment PIDs:

| Experiment | GPU | PID | Latest status | Finish estimate |
| --- | ---: | ---: | --- | --- |
| S2 hierarchy regularizer 144e | 2 | `3891957` | Final validation line logged at epoch 144: `dota/mAP=0.3723`, `dota/AP50=0.3720` | Effectively finished; allow a few minutes for process/screen cleanup and metric parsing |
| S3 scene adapter 144e | 6 | `425496` | `Epoch(train) [75][40/1410]`, latest log ETA `12:47:44` | Around `2026-06-03 05:14 +0800` |
| S0 DOTA2 RoITrans valid-PNG | 0 | `1077281` | `Epoch(train) [2][1550/39007]`, latest log ETA `1 day, 2:31:43` | Around `2026-06-03 19:57 +0800` |
| S0 DOTA2 ORCNN R50 valid-PNG | 1 | `1598732` | Filtered annotation preparation around `161530/170831`; not yet in `Epoch(train)` | Annotation pass likely under 15 minutes left; full 12e run roughly `2026-06-03 21:30-23:30 +0800` once training timing is confirmed |

Non-project GPU activity observed during this check:

- GPU 3 has a separate `python` process using about `21932 MiB`.
- GPUs 4 and 5 remain occupied by other users.
- Small `cmnext` Python processes appeared on GPUs 1 and 2; they were not part
  of the four screens listed above.

## Latest Log Evidence

S2 hierarchy regularizer 144e:

```text
06/02 16:25:54 - mmengine - INFO - Epoch(val) [144][458/458]    dota/mAP: 0.3723  dota/AP50: 0.3720
```

S3 scene adapter 144e:

```text
06/02 16:26:10 - mmengine - INFO - Epoch(train)  [75][  40/1410] ... eta: 12:47:44
```

S0 DOTA2 RoITrans valid-PNG:

```text
06/02 16:26:05 - mmengine - INFO - Epoch(train)  [2][ 1550/39007] ... eta: 1 day, 2:31:43
```

S0 DOTA2 ORCNN R50 valid-PNG:

```text
95%|█████████▍| 161530/170831 [1:28:14<04:08, 37.44it/s]
```

The ORCNN run still needs training-iteration verification after the annotation
pass finishes. Do not cite it as recovered iteration-level evidence until it
enters `Epoch(train)` and reaches at least `[1][1600/39007]` or the equivalent
denominator without PNG-related crash signatures.

## Follow-Up

- Parse and record the final S2 144e metrics now that epoch-144 validation is
  logged.
- Keep S3 144e running; next checkpoint/validation milestones are the regular
  epoch intervals in its config.
- Keep S0 RoITrans and ORCNN valid-PNG runs separate from S1/S2/S3 claims.
- Recheck ORCNN shortly after the annotation pass finishes to verify epoch-1
  training reaches the `[1600]` acceptance threshold without `libpng`,
  `NoneType`, `CRC`, or `Traceback` signatures.
