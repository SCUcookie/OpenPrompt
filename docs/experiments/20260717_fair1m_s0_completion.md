# FAIR1M S0 Campaign Completion (2026-07-17)

This completion record reconciles the three replicas with the launch record
in `20260715_fair1m_s0_campaign_launch.md`. All replicas completed the full
12-epoch schedule and epoch-12 validation.

## Results

| Seed | GPU / screen | Epoch 4 mAP / AP50 | Epoch 8 mAP / AP50 | Epoch 12 mAP / AP50 |
|---:|---|---|---|---|
| 3407 | GPU 0 / `fair1m_s0_rep3407_20260715_gpu0` | `0.3106 / 0.3110` | `0.3178 / 0.3180` | `0.3045 / 0.3040` |
| 4407 | GPU 1 / `fair1m_s0_rep4407_20260715_gpu1` | `0.3058 / 0.3060` | `0.3165 / 0.3160` | `0.3043 / 0.3040` |
| 5407 | GPU 2 / `fair1m_s0_rep5407_20260715_gpu2` | `0.3143 / 0.3140` | `0.3163 / 0.3160` | `0.3109 / 0.3110` |

Epoch-12 final mAP mean/std is `0.306567 / 0.003065`. Epoch-8 best mAP
mean/std is `0.316867 / 0.000665`; epoch 8 is the best checkpoint for each
replica. The final result is therefore reported separately from the selected
best-checkpoint result.

## Final Class AP

The following values are the exact epoch-12 AP rows from the three runtime
logs, in canonical FAIR1M class order:

| Class | 3407 | 4407 | 5407 |
|---|---:|---:|---:|
| a220 | .103 | .092 | .121 |
| a321 | .084 | .103 | .133 |
| a330 | .106 | .106 | .123 |
| a350 | .124 | .134 | .163 |
| arj21 | .035 | .061 | .041 |
| baseball-field | .855 | .861 | .855 |
| basketball-court | .445 | .455 | .457 |
| boeing737 | .089 | .095 | .062 |
| boeing747 | .452 | .405 | .364 |
| boeing777 | .023 | .018 | .052 |
| boeing787 | .115 | .126 | .185 |
| bridge | .394 | .410 | .411 |
| bus | .226 | .175 | .224 |
| c919 | .000 | .000 | .000 |
| cargo-truck | .370 | .362 | .367 |
| dry-cargo-ship | .584 | .581 | .590 |
| dump-truck | .243 | .244 | .245 |
| engineering-ship | .416 | .417 | .419 |
| excavator | .228 | .212 | .228 |
| fishing-boat | .237 | .227 | .236 |
| football-field | .513 | .513 | .494 |
| intersection | .553 | .551 | .555 |
| liquid-cargo-ship | .338 | .333 | .346 |
| motorboat | .421 | .433 | .432 |
| other-airplane | .455 | .392 | .458 |
| other-ship | .092 | .096 | .099 |
| other-vehicle | .034 | .032 | .094 |
| passenger-ship | .300 | .314 | .288 |
| roundabout | .650 | .705 | .706 |
| small-car | .397 | .398 | .397 |
| tennis-court | .804 | .807 | .800 |
| tractor | .020 | .007 | .014 |
| trailer | .173 | .188 | .187 |
| truck-tractor | .281 | .304 | .294 |
| tugboat | .247 | .260 | .261 |
| van | .372 | .375 | .375 |
| warship | .482 | .467 | .426 |

## Provenance And Scans

- Launch record: `20260715_fair1m_s0_campaign_launch.md`.
- Workdirs: `OpenRSD/work_dirs/geonexus_fair1m/roi_trans_s0_rep{3407,4407,5407}_20260715`.
- Source: local ResNet-50 checkpoint SHA-256
  `0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a`.
- Configs and exact commands are retained in each workdir's
  `launch_provenance.txt`; physical GPUs and screen names are unchanged from
  the launch record.
- Epoch-4/8/12 checkpoints exist for all three replicas. Runtime logs are
  `20260715_103147/20260715_103147.log` in each workdir.
- Scoped scans of all three logs found no traceback, CUDA OOM, decode/CRC,
  invalid-box, NaN/Inf, or keyboard-interruption signatures.
- The precision-v2 data/runtime gate remains the provenance source for the
  `208927/10970` train/validation pairs, `[37,512]` canonical embeddings,
  and finite 1000-step S0 diagnostic.

FAIR1M TPC/S1 is the next controlled route. No FAIR1M S2/GeoNexus or paused
DOTA2, DIOR-R S4, pseudo-labeling, or segmentation work is launched by this
completion record.
