# 2026-06-30 DIOR-R SOTA Audit And S4 Micro-Sweep Launch

## Scope

Make the DIOR-R SOTA comparison defensible before claiming superiority, while
using currently free GPUs for a controlled S4 stabilization micro-sweep.

Decision: do not claim DIOR-R SOTA from the current GeoNexus numbers alone.
The strongest local DIOR-R S3 result remains best mean `0.6979` under the
local sanitized `DIOR_R_dota/test` MMRotate `DOTAMetric` evaluator. Public
comparators must be matched on split, annotation layout, class names, rotated
box convention, evaluator semantics, and checkpoint-selection policy first.

## Comparator Clone Audit

Local clone commits were read from `.git/HEAD` and `.git/refs/heads/main`
because normal `git rev-parse` is blocked by Git safe-directory ownership
checks in this workspace.

| Repo | Local path | Commit | DIOR config inspected |
| --- | --- | --- | --- |
| OrientedFormer | `/data5/2025/ldh/OrientedFormer` | `e6e42f9` | `projects/OrientedFormer/configs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dior.py` and `configs/_base_/datasets/dior.py` |
| Strip R-CNN | `/data5/2025/ldh/Strip-R-CNN` | `3774f97` | `configs/strip_rcnn/strip_rcnn_s_fpn_1x_dior_le90.py` and `configs/_base_/datasets/dior.py` |
| LSKNet | `/data5/2025/ldh/LSKNet` | `386cbef` | clone present; DIOR config path available through Strip R-CNN family context |

Checkpoint scan:

```bash
find OrientedFormer Strip-R-CNN LSKNet -type f \( -name '*.pth' -o -name '*.pt' -o -name '*.pkl' \)
```

Result: no local third-party `.pth`, `.pt`, or `.pkl` checkpoints were found
inside the cloned comparator repos.

## Protocol Mismatch

OrientedFormer DIOR config expects XML-style DIOR under `data/DIOR/`, with
`ImageSets/Main/{train,val,test}.txt`, `JPEGImages-trainval`,
`JPEGImages-test`, and annotations loaded by `DIORDataset` from
`Annotations/Oriented Bounding Boxes`.

Strip R-CNN DIOR config expects XML-style DIOR under
`/defaultShare/pubdata/remote_sensing/DIOR/`, with `ImageSets/train.txt`,
`ImageSets/val.txt`, `ImageSets/test.txt`, `JPEGImages/trainval`,
`JPEGImages/test`, and `Annotations/Oriented Bounding Boxes/`.

GeoNexus DIOR-R results use DOTA-style local data roots such as
`data/DIOR_R_dota/` and
`data/DIOR_R_dota_s4_pseudo_agreement_20260627/`, with `train_val/labelTxt`,
`train_val/images`, `test/labelTxt`, and `test/images`.

This is a direct dataset-layout and loader mismatch. Released comparator
checkpoint evaluation should not be run until the official checkpoints are
available and the DIOR-R protocol is matched or an explicit conversion bridge
is documented.

## Comparator Lane

1. Archive exact OrientedFormer Swin-T and Strip R-CNN-S DIOR configs and
   dataset loaders before any metric claim.
2. Do not run third-party evaluation until official checkpoints are downloaded
   and the dataset layout/protocol is matched.
3. If downloads are approved later, evaluate released checkpoints before any
   retraining.
4. Record checkpoint URL, hash, repo commit, config, conversion rules, command,
   log, metrics JSON, and failure scan for every comparator run.

## S4 Micro-Sweep Configs

The S4 micro-sweep is separate metric-improvement work. It is not SOTA evidence
unless it beats S3 under the same evaluator and selection policy.

All configs initialize from the best S4 short-pack epoch-1 checkpoints, keep
the pseudo-label root `data/DIOR_R_dota_s4_pseudo_agreement_20260627/`, keep
`resume=False`, keep `val_interval=1`, and keep the original replica seeds.

| Replica | GPU | Config | Source checkpoint | Workdir |
| --- | ---: | --- | --- | --- |
| rep23407 | 0 | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr5e6_rep23407_20260630/roi-trans-le90_r50_fpn_remoteclip-s4-pseudo-agreement-e1-lr5e6-rep23407-20260630_dior_r.py` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_rep23407_20260627/epoch_1.pth` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr5e6_rep23407_20260630` |
| rep24407 | 4 | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr5e6_rep24407_20260630/roi-trans-le90_r50_fpn_remoteclip-s4-pseudo-agreement-e1-lr5e6-rep24407-20260630_dior_r.py` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_rep24407_20260627/epoch_1.pth` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr5e6_rep24407_20260630` |
| rep25407 | 5 | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr5e6_rep25407_20260630/roi-trans-le90_r50_fpn_remoteclip-s4-pseudo-agreement-e1-lr5e6-rep25407-20260630_dior_r.py` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_rep25407_20260627/epoch_1.pth` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr5e6_rep25407_20260630` |

Controlled changes from the accepted 2026-06-28 S4 low-LR configs:

- optimizer LR: `1e-5` to `5e-6`
- `max_epochs`: `8` to `6`
- `work_dir`: new 2026-06-30 workdirs

Config parse/print checks passed through:

```bash
PYTHONNOUSERSITE=1 /data1/anaconda3/envs/zwl_mmrotate/bin/python \
  tools/bootstrap_run.py tools/misc/print_config.py <config> \
  --save-path /tmp/dior_r_s4_e1_lr5e6_<rep>_20260630_print.py
```

Saved printouts confirmed `load_from`, pseudo-label data root, `lr=5e-06`,
`max_epochs=6`, `val_interval=1`, `resume=False`, seed, and workdir.

## Preflight

Source checkpoint existence checks passed for all three replicas.

Initial GPU check before config generation showed GPUs `0`, `4`, and `5` idle;
GPUs `1`, `2`, `3`, and `6` were occupied and were not targeted. Final GPU
state immediately before launch again showed GPUs `0`, `4`, and `5` idle.

Launch shape:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONNOUSERSITE=1 \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python \
  tools/bootstrap_run.py tools/train.py <config> \
  > <workdir>/launch_20260630_gpu<gpu>_bootstrap.log 2>&1
```

## Startup Acceptance

Accepted launch at `2026-06-30 11:09 CST`:

| Replica | GPU | PID | Screen | Launch log | Runtime log |
| --- | ---: | ---: | --- | --- | --- |
| rep23407 | 0 | `2325827` | `dior_r_s4_e1_lr5e6_rep23407_20260630_gpu0` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr5e6_rep23407_20260630/launch_20260630_gpu0_bootstrap.log` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr5e6_rep23407_20260630/20260630_110918/20260630_110918.log` |
| rep24407 | 4 | `2325835` | `dior_r_s4_e1_lr5e6_rep24407_20260630_gpu4` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr5e6_rep24407_20260630/launch_20260630_gpu4_bootstrap.log` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr5e6_rep24407_20260630/20260630_110918/20260630_110918.log` |
| rep25407 | 5 | `2325831` | `dior_r_s4_e1_lr5e6_rep25407_20260630_gpu5` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr5e6_rep25407_20260630/launch_20260630_gpu5_bootstrap.log` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr5e6_rep25407_20260630/20260630_110918/20260630_110918.log` |

Startup verification at `2026-06-30 11:10 CST`:

- each screen is detached and alive;
- each process resides on the intended GPU;
- each accepted bootstrap log reached at least
  `Epoch(train) [1][ 200/5847]`;
- logs confirm intended `load_from`, LR `5.0000e-06`, and checkpoint save
  workdir;
- scoped failure scan was clean for `Traceback`, CUDA OOM, `out-of-memory`,
  `out of memory`, `libpng`, `CRC`, `NoneType`, `ValueError`,
  `KeyboardInterrupt`, `loss: nan`, `loss: inf`, `grad_norm: nan`, and
  `grad_norm: inf`.

## Completion Criteria

Archive best and final `dota/mAP` and `dota/AP50` per replica. Compute best
mean and final mean separately.

Strong S4 evidence requires best mean above DIOR-R S3 best mean `0.6979`, or
any single checkpoint cleanly reaching at least `0.7000` and being reproduced
or validated by paper-eval. Otherwise classify the run as stabilization or
negative evidence.
