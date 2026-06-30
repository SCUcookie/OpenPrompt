# 2026-06-28 DIOR-R S4 Low-LR From Epoch-1 Launch

## Scope

Archive the completed 2026-06-27 DIOR-R S4 pseudo-label short-pack, then run a
controlled stabilization test from each replica's S4 epoch-1 checkpoint.

This is not a new superiority claim. The purpose is to test whether a lower LR
continuation from the early best checkpoints stabilizes final-epoch behavior.

## S4 Short-Pack Archive

Source launch note:
`New/docs/experiments/20260627_dior_r_s4_pseudolabel_pilot_launch.md`.

All three 2026-06-27 S4 pseudo-label short-pack best checkpoints occurred at
epoch 1. Final epoch-12 mean degraded relative to the epoch-1 best mean.

| Replica | Best epoch | Best `dota/mAP` | Best `dota/AP50` | Final epoch | Final `dota/mAP` | Final `dota/AP50` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rep23407 | 1 | 0.697574 | 0.698 | 12 | 0.693460 | 0.693 |
| rep24407 | 1 | 0.696419 | 0.696 | 12 | 0.690285 | 0.690 |
| rep25407 | 1 | 0.696716 | 0.697 | 12 | 0.690267 | 0.690 |

Aggregate:

- best mean `dota/mAP`: `0.696903`
- final mean `dota/mAP`: `0.691337`
- failure scan: clean on the completed short-pack logs

Decision: continue S4 only as a low-LR stabilization test.

## Configs

Generated clean workdirs under `OpenRSD/work_dirs/geonexus_dior_r/`:

| Replica | Config | Source checkpoint | Workdir |
| --- | --- | --- | --- |
| rep23407 | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr1e5_rep23407_20260628/roi-trans-le90_r50_fpn_remoteclip-s4-pseudo-agreement-e1-lr1e5-rep23407-20260628_dior_r.py` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_rep23407_20260627/epoch_1.pth` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr1e5_rep23407_20260628` |
| rep24407 | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr1e5_rep24407_20260628/roi-trans-le90_r50_fpn_remoteclip-s4-pseudo-agreement-e1-lr1e5-rep24407-20260628_dior_r.py` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_rep24407_20260627/epoch_1.pth` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr1e5_rep24407_20260628` |
| rep25407 | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr1e5_rep25407_20260628/roi-trans-le90_r50_fpn_remoteclip-s4-pseudo-agreement-e1-lr1e5-rep25407-20260628_dior_r.py` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_rep25407_20260627/epoch_1.pth` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr1e5_rep25407_20260628` |

Controlled changes from the 2026-06-27 S4 configs:

- `load_from`: matching replica `epoch_1.pth`
- `work_dir`: new 2026-06-28 low-LR workdir
- optimizer LR: `2.5e-5` to `1e-5`
- `max_epochs`: `12` to `8`

Kept unchanged:

- pseudo-label data root: `data/DIOR_R_dota_s4_pseudo_agreement_20260627/`
- `val_interval=1`
- `resume=False`
- seeds `23407`, `24407`, `25407`

Config parse/print checks passed through:

```bash
PYTHONNOUSERSITE=1 /data1/anaconda3/envs/zwl_mmrotate/bin/python \
  tools/bootstrap_run.py tools/misc/print_config.py <config> \
  --save-path /tmp/dior_r_s4_e1_lr1e5_<rep>_20260628_print.py
```

Saved printouts confirmed `load_from`, `work_dir`, `lr=1e-05`,
`max_epochs=8`, `val_interval=1`, `resume=False`, seed, and data root.

## Preflight

Preflight at `2026-06-28 10:15 CST`:

- `screen -ls` showed only `s0_result_log_monitor_20260603` before launch.
- Checkpoint existence checks passed for all three source `epoch_1.pth` files.
- GPU state:
  - GPU 0: idle
  - GPU 1: occupied by PID `616621` owned by another user/process
  - GPU 2: idle
  - GPU 3: idle
  - GPUs 4-6: idle
- Default mapping was used with no remap:
  - rep23407 -> GPU 0
  - rep24407 -> GPU 2
  - rep25407 -> GPU 3

## Launch

Initial direct `tools/train.py` launches exited immediately because
`geonexus_mmrotate` was not on `sys.path`. Those failed logs are preserved as
`launch_20260628_gpu*.log`; they are not the accepted training logs.

Accepted bootstrap relaunch at `2026-06-28 10:15 CST`:

| Replica | GPU | PID | Screen | Launch log | Runtime log |
| --- | ---: | ---: | --- | --- | --- |
| rep23407 | 0 | `743669` | `dior_r_s4_e1_lr1e5_rep23407_20260628_gpu0` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr1e5_rep23407_20260628/launch_20260628_gpu0_bootstrap.log` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr1e5_rep23407_20260628/20260628_101540/20260628_101540.log` |
| rep24407 | 2 | `743673` | `dior_r_s4_e1_lr1e5_rep24407_20260628_gpu2` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr1e5_rep24407_20260628/launch_20260628_gpu2_bootstrap.log` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr1e5_rep24407_20260628/20260628_101540/20260628_101540.log` |
| rep25407 | 3 | `743672` | `dior_r_s4_e1_lr1e5_rep25407_20260628_gpu3` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr1e5_rep25407_20260628/launch_20260628_gpu3_bootstrap.log` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr1e5_rep25407_20260628/20260628_101540/20260628_101540.log` |

Launch command shape:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONNOUSERSITE=1 \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python \
  tools/bootstrap_run.py tools/train.py <config> \
  > <workdir>/launch_20260628_gpu<gpu>_bootstrap.log 2>&1
```

The bootstrap wrapper is required for this repo because it preloads installed
OpenMMLab packages before appending the local OpenRSD repo, while still making
`geonexus_mmrotate` importable.

## Startup Acceptance

Acceptance at `2026-06-28 10:18 CST`:

- screens detached and alive for all three replicas.
- GPU residency confirmed on GPUs 0, 2, and 3; GPU 1 remained occupied by PID
  `616621`.
- `ps -p 743669,743673,743672 -o pid,ppid,user,cmd --forest` confirmed each
  Python process and matching config.
- each bootstrap log confirmed loading the matching local
  `roi_trans_remoteclip_s4_pseudo_agreement_rep*_20260627/epoch_1.pth`.
- each bootstrap log reached `Epoch(train) [1][ 450/5847]`.
- scoped failure scan across the three accepted bootstrap launch logs was clean
  for `Traceback`, CUDA OOM, `out-of-memory`, `out of memory`, `libpng`, `CRC`,
  `NoneType`, `ValueError`, `KeyboardInterrupt`, `loss: nan`, `loss: inf`,
  `grad_norm: nan`, and `grad_norm: inf`.

## Completion Criteria

Completion record must report best and final `dota/mAP` and `dota/AP50`
separately for every replica, plus aggregate best mean and final mean.

Interpretation thresholds:

- strong S4 evidence only if best mean exceeds original DIOR-R S3 best mean
  `0.6979` or final mean clearly exceeds long60 final mean `0.693014`.
- stabilization evidence if final mean improves over S4 short-pack final mean
  `0.691337` without failure signatures.
- archive as neutral/negative if final mean stays below `0.691337` or best
  mean does not improve over `0.696903`.
