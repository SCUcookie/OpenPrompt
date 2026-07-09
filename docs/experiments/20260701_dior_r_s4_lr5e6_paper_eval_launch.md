# 2026-07-01 DIOR-R S4 LR5e-6 Paper-Eval Launch

## Scope

Launch paper-facing evaluation artifacts for the best DIOR-R S4 LR5e-6
checkpoint from each replica. This is evaluation-only: no S4 training, no
route change, and no new superiority claim.

Source completion record:
`New/docs/experiments/20260701_dior_r_s4_lr5e6_complete.md`.

Evaluation command shape: `tools/bootstrap_run.py tools/test.py` with
`--out preds.pkl`.

## Preflight

Preflight at `2026-07-01 09:27 CST`:

- `screen -ls` showed only `s0_result_log_monitor_20260603`.
- GPU state: GPUs 0, 1, 3, 4, and 5 were idle; GPUs 2 and 6 were occupied.
- Preferred GPUs 0, 4, and 5 were idle, so no remap was needed.
- Checkpoints and configs existed for all three best LR5e-6 checkpoints.

Mapping:

| Replica | Checkpoint | GPU | Workdir |
| --- | --- | ---: | --- |
| rep23407 | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr5e6_rep23407_20260630/epoch_6.pth` | 0 | `work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep23407_epoch6` |
| rep24407 | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr5e6_rep24407_20260630/epoch_2.pth` | 4 | `work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep24407_epoch2` |
| rep25407 | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr5e6_rep25407_20260630/epoch_2.pth` | 5 | `work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep25407_epoch2` |

## Launch Trail

Launched detached screens from `/data5/2025/ldh/OpenRSD` at
`2026-07-01 09:29 CST`.

| Replica | GPU | PID | Screen | Launch log | Prediction output |
| --- | ---: | ---: | --- | --- | --- |
| rep23407 epoch 6 | 0 | `3912259` | `paper_eval_dior_r_s4_lr5e6_rep23407_e6_20260701_gpu0` | `work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep23407_epoch6/launch_20260701_gpu0.log` | `work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep23407_epoch6/preds.pkl` |
| rep24407 epoch 2 | 4 | `3912260` | `paper_eval_dior_r_s4_lr5e6_rep24407_e2_20260701_gpu4` | `work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep24407_epoch2/launch_20260701_gpu4.log` | `work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep24407_epoch2/preds.pkl` |
| rep25407 epoch 2 | 5 | `3912262` | `paper_eval_dior_r_s4_lr5e6_rep25407_e2_20260701_gpu5` | `work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep25407_epoch2/launch_20260701_gpu5.log` | `work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep25407_epoch2/preds.pkl` |

Commands:

```bash
screen -dmS paper_eval_dior_r_s4_lr5e6_rep23407_e6_20260701_gpu0 bash -lc 'cd /data5/2025/ldh/OpenRSD && mkdir -p work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep23407_epoch6 && CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib_dior_r_s4_lr5e6_rep23407_20260701 PYTHONNOUSERSITE=1 /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/test.py work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr5e6_rep23407_20260630/roi-trans-le90_r50_fpn_remoteclip-s4-pseudo-agreement-e1-lr5e6-rep23407-20260630_dior_r.py work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr5e6_rep23407_20260630/epoch_6.pth --work-dir work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep23407_epoch6 --out work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep23407_epoch6/preds.pkl > work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep23407_epoch6/launch_20260701_gpu0.log 2>&1'
```

```bash
screen -dmS paper_eval_dior_r_s4_lr5e6_rep24407_e2_20260701_gpu4 bash -lc 'cd /data5/2025/ldh/OpenRSD && mkdir -p work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep24407_epoch2 && CUDA_VISIBLE_DEVICES=4 MPLCONFIGDIR=/tmp/matplotlib_dior_r_s4_lr5e6_rep24407_20260701 PYTHONNOUSERSITE=1 /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/test.py work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr5e6_rep24407_20260630/roi-trans-le90_r50_fpn_remoteclip-s4-pseudo-agreement-e1-lr5e6-rep24407-20260630_dior_r.py work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr5e6_rep24407_20260630/epoch_2.pth --work-dir work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep24407_epoch2 --out work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep24407_epoch2/preds.pkl > work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep24407_epoch2/launch_20260701_gpu4.log 2>&1'
```

```bash
screen -dmS paper_eval_dior_r_s4_lr5e6_rep25407_e2_20260701_gpu5 bash -lc 'cd /data5/2025/ldh/OpenRSD && mkdir -p work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep25407_epoch2 && CUDA_VISIBLE_DEVICES=5 MPLCONFIGDIR=/tmp/matplotlib_dior_r_s4_lr5e6_rep25407_20260701 PYTHONNOUSERSITE=1 /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/test.py work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr5e6_rep25407_20260630/roi-trans-le90_r50_fpn_remoteclip-s4-pseudo-agreement-e1-lr5e6-rep25407-20260630_dior_r.py work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr5e6_rep25407_20260630/epoch_2.pth --work-dir work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep25407_epoch2 --out work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep25407_epoch2/preds.pkl > work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep25407_epoch2/launch_20260701_gpu5.log 2>&1'
```

## Startup Acceptance

Startup acceptance at `2026-07-01 09:32 CST`:

- Screens detached and alive for all three eval jobs.
- GPU residency confirmed:
  - GPU 0: PID `3912259`, 4384 MiB.
  - GPU 4: PID `3912260`, 4214 MiB.
  - GPU 5: PID `3912262`, 4214 MiB.
- Launch logs show intended checkpoint loading:
  - rep23407: `epoch_6.pth`.
  - rep24407: `epoch_2.pth`.
  - rep25407: `epoch_2.pth`.
- All three logs reached at least `Epoch(test) [ 350/5869]`.
- Scoped failure scan across the three launch logs was clean for `Traceback`,
  CUDA OOM, `out-of-memory`, `out of memory`, `libpng`, `CRC`, `NoneType`,
  `ValueError`, `KeyboardInterrupt`, `loss: nan`, `loss: inf`,
  `grad_norm: nan`, and `grad_norm: inf`.

## Completion Criteria

Each workdir should contain `preds.pkl`, the launch/runtime log, copied config,
and JSON metric output. Final metrics should be recorded separately from
training-log metrics. S4 remains classified as stabilization /
negative-to-neutral unless the paper eval unexpectedly crosses the existing
gates.

At completion, the screen state should return to only
`s0_result_log_monitor_20260603`.

## Completion Verification

All three paper-eval jobs completed on `2026-07-01 09:40 CST`.

Artifacts:

| Replica | Metrics JSON | Runtime log | `preds.pkl` |
| --- | --- | --- | --- |
| rep23407 epoch 6 | `work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep23407_epoch6/20260701_092953/20260701_092953.json` | `work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep23407_epoch6/20260701_092953/20260701_092953.log` | present |
| rep24407 epoch 2 | `work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep24407_epoch2/20260701_092953/20260701_092953.json` | `work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep24407_epoch2/20260701_092953/20260701_092953.log` | present |
| rep25407 epoch 2 | `work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep25407_epoch2/20260701_092953/20260701_092953.json` | `work_dirs/paper_eval_20260701/dior_r_s4_e1_lr5e6_rep25407_epoch2/20260701_092953/20260701_092953.log` | present |

Final paper-eval metrics:

| Replica | Paper-eval `dota/mAP` | Paper-eval `dota/AP50` | Expected training-log best |
| --- | ---: | ---: | --- |
| rep23407 epoch 6 | 0.696492 | 0.6960 | matched `0.696492/0.6960` |
| rep24407 epoch 2 | 0.696987 | 0.6970 | matched `0.696987/0.6970` |
| rep25407 epoch 2 | 0.698336 | 0.6980 | matched `0.698336/0.6980` |

Aggregate paper-eval best mean mAP: `0.697272`.

Final state:

- `screen -ls` returned to only `s0_result_log_monitor_20260603`.
- GPUs 0, 4, and 5 returned to idle.
- Scoped failure scan across `work_dirs/paper_eval_20260701` was clean for
  `Traceback`, CUDA OOM, `out-of-memory`, `out of memory`, `libpng`, `CRC`,
  `NoneType`, `ValueError`, `KeyboardInterrupt`, `loss: nan`, `loss: inf`,
  `grad_norm: nan`, and `grad_norm: inf`.
- S4 remains stabilization / negative-to-neutral: mean `0.697272` is below
  the S3 gate `0.6979`, and best single `0.698336` is below the single-run
  gate `0.7000`.
