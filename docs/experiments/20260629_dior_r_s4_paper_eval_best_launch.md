# 2026-06-29 DIOR-R S4 Paper-Eval Best Launch

## Scope

Launch paper-facing evaluation artifacts for the best low-LR S4 checkpoint from
each replica. This is evaluation-only: no S4 training, no route change, and no
new superiority claim.

Source completion record:
`New/docs/experiments/20260629_dior_r_s4_low_lr_complete.md`.

Evaluation command shape: `tools/bootstrap_run.py tools/test.py` with
`--out preds.pkl`.

## Preflight

Preflight at `2026-06-29 09:06 CST`:

- `screen -ls` showed only `s0_result_log_monitor_20260603`.
- Checkpoint and config existence checks passed for all three best low-LR
  checkpoints.
- GPU state: GPUs 0, 2, and 3 were idle; GPU 1 was occupied by PID `616621`;
  GPU 6 was occupied by another process. No remap was needed.

Mapping:

| Replica | Checkpoint | GPU | Workdir |
| --- | --- | ---: | --- |
| rep23407 | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr1e5_rep23407_20260628/epoch_2.pth` | 0 | `work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep23407_epoch2` |
| rep24407 | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr1e5_rep24407_20260628/epoch_6.pth` | 2 | `work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep24407_epoch6` |
| rep25407 | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr1e5_rep25407_20260628/epoch_2.pth` | 3 | `work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep25407_epoch2` |

## Launch Trail

Launched detached screens from `/data5/2025/ldh/OpenRSD` at
`2026-06-29 09:07 CST`.

| Replica | GPU | PID | Screen | Launch log | Prediction output |
| --- | ---: | ---: | --- | --- | --- |
| rep23407 epoch 2 | 0 | `1279562` | `paper_eval_dior_r_s4_rep23407_e2_20260629_gpu0` | `work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep23407_epoch2/launch_20260629_gpu0.log` | `work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep23407_epoch2/preds.pkl` |
| rep24407 epoch 6 | 2 | `1280263` | `paper_eval_dior_r_s4_rep24407_e6_20260629_gpu2` | `work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep24407_epoch6/launch_20260629_gpu2.log` | `work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep24407_epoch6/preds.pkl` |
| rep25407 epoch 2 | 3 | `1281166` | `paper_eval_dior_r_s4_rep25407_e2_20260629_gpu3` | `work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep25407_epoch2/launch_20260629_gpu3.log` | `work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep25407_epoch2/preds.pkl` |

Commands:

```bash
screen -dmS paper_eval_dior_r_s4_rep23407_e2_20260629_gpu0 bash -lc 'cd /data5/2025/ldh/OpenRSD && mkdir -p work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep23407_epoch2 && CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib_dior_r_s4_rep23407_e2_20260629 PYTHONNOUSERSITE=1 /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/test.py work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr1e5_rep23407_20260628/roi-trans-le90_r50_fpn_remoteclip-s4-pseudo-agreement-e1-lr1e5-rep23407-20260628_dior_r.py work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr1e5_rep23407_20260628/epoch_2.pth --work-dir work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep23407_epoch2 --out work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep23407_epoch2/preds.pkl > work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep23407_epoch2/launch_20260629_gpu0.log 2>&1'
```

```bash
screen -dmS paper_eval_dior_r_s4_rep24407_e6_20260629_gpu2 bash -lc 'cd /data5/2025/ldh/OpenRSD && mkdir -p work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep24407_epoch6 && CUDA_VISIBLE_DEVICES=2 MPLCONFIGDIR=/tmp/matplotlib_dior_r_s4_rep24407_e6_20260629 PYTHONNOUSERSITE=1 /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/test.py work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr1e5_rep24407_20260628/roi-trans-le90_r50_fpn_remoteclip-s4-pseudo-agreement-e1-lr1e5-rep24407-20260628_dior_r.py work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr1e5_rep24407_20260628/epoch_6.pth --work-dir work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep24407_epoch6 --out work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep24407_epoch6/preds.pkl > work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep24407_epoch6/launch_20260629_gpu2.log 2>&1'
```

```bash
screen -dmS paper_eval_dior_r_s4_rep25407_e2_20260629_gpu3 bash -lc 'cd /data5/2025/ldh/OpenRSD && mkdir -p work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep25407_epoch2 && CUDA_VISIBLE_DEVICES=3 MPLCONFIGDIR=/tmp/matplotlib_dior_r_s4_rep25407_e2_20260629 PYTHONNOUSERSITE=1 /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/test.py work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr1e5_rep25407_20260628/roi-trans-le90_r50_fpn_remoteclip-s4-pseudo-agreement-e1-lr1e5-rep25407-20260628_dior_r.py work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_e1_lr1e5_rep25407_20260628/epoch_2.pth --work-dir work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep25407_epoch2 --out work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep25407_epoch2/preds.pkl > work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep25407_epoch2/launch_20260629_gpu3.log 2>&1'
```

## Startup Acceptance

Startup acceptance at `2026-06-29 09:09 CST`:

- Screens detached and alive for all three eval jobs.
- GPU residency confirmed:
  - GPU 0: PID `1279562`, 4550 MiB
  - GPU 2: PID `1280263`, 4392 MiB
  - GPU 3: PID `1281166`, 4386 MiB
- Launch logs show intended checkpoint loading:
  - rep23407: `epoch_2.pth`
  - rep24407: `epoch_6.pth`
  - rep25407: `epoch_2.pth`
- All three logs reached at least `Epoch(test) [ 350/5869]`.
- Scoped failure scan across the three launch logs was clean for `Traceback`,
  CUDA OOM, `out-of-memory`, `out of memory`, `libpng`, `CRC`, `NoneType`,
  `ValueError`, `KeyboardInterrupt`, `loss: nan`, `loss: inf`,
  `grad_norm: nan`, and `grad_norm: inf`.

## Completion Criteria

Each workdir should contain `preds.pkl`, the launch/runtime log, copied config,
and JSON metric output. Final metrics should match the corresponding
training-log best-checkpoint values within expected evaluator rounding:

- rep23407 epoch 2: about `0.6935/0.6930`.
- rep24407 epoch 6: about `0.6966/0.6970`.
- rep25407 epoch 2: about `0.6967/0.6970`.

At completion, the screen state should return to only
`s0_result_log_monitor_20260603`.

## Completion Verification

All three paper-eval jobs completed on `2026-06-29 09:17 CST`.

Artifacts:

| Replica | Metrics JSON | Runtime log | `preds.pkl` |
| --- | --- | --- | --- |
| rep23407 epoch 2 | `work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep23407_epoch2/20260629_090734/20260629_090734.json` | `work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep23407_epoch2/20260629_090734/20260629_090734.log` | present |
| rep24407 epoch 6 | `work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep24407_epoch6/20260629_090747/20260629_090747.json` | `work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep24407_epoch6/20260629_090747/20260629_090747.log` | present |
| rep25407 epoch 2 | `work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep25407_epoch2/20260629_090801/20260629_090801.json` | `work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep25407_epoch2/20260629_090801/20260629_090801.log` | present |

Final metrics:

| Replica | Paper-eval `dota/mAP` | Paper-eval `dota/AP50` | Expected training-log best |
| --- | ---: | ---: | --- |
| rep23407 epoch 2 | 0.6935 | 0.6930 | matched `0.6935/0.6930` |
| rep24407 epoch 6 | 0.6966 | 0.6970 | matched `0.6966/0.6970` |
| rep25407 epoch 2 | 0.6967 | 0.6970 | matched `0.6967/0.6970` |

Final state:

- `screen -ls` returned to only `s0_result_log_monitor_20260603`.
- GPUs 0, 2, and 3 returned to idle.
- Scoped failure scan across the three launch logs and runtime logs was clean
  for `Traceback`, CUDA OOM, `out-of-memory`, `out of memory`, `libpng`,
  `CRC`, `NoneType`, `ValueError`, `KeyboardInterrupt`, `loss: nan`,
  `loss: inf`, `grad_norm: nan`, and `grad_norm: inf`.
