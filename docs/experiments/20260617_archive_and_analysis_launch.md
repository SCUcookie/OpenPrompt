# 2026-06-17 Archive And Analysis Launch

## Scope

Archive hygiene and paper-facing evaluation only. No S4, pseudo-labeling,
FAIR1M, routing, new training, or other route-changing follow-up is launched
by this note.

## Repository State Before Launch

Command: `git -C New status --short`

```text
 M PROJECT_INSTRUCTIONS.md
 M artifacts/result_assets_20260614/all_experiment_results_20260614.csv
 M docs/experiments/20260614_dior_r_geonexus_s2_replicas_complete.json
 M docs/experiments/20260614_dior_r_geonexus_s2_replicas_complete.md
 M docs/geonexus_short_paper.tex
 M scripts/make_result_assets_20260614.py
?? docs/experiments/20260614_dior_r_geonexus_s3_scene_adapter_replicas_launch.md
?? docs/experiments/20260615_dior_r_geonexus_s3_scene_adapter_replicas_complete.json
?? docs/experiments/20260615_dior_r_geonexus_s3_scene_adapter_replicas_complete.md
?? docs/experiments/20260615_dior_r_s3_epoch8_lr5e5_stability_complete.json
?? docs/experiments/20260615_dior_r_s3_epoch8_lr5e5_stability_complete.md
?? docs/experiments/20260615_dior_r_s3_epoch8_lr5e5_stability_launch.md
?? docs/experiments/20260615_dota2_s3_scene_adapter_loss0_best_launch.md
?? docs/experiments/20260616_dior_r_s3_stability_e4_lr2p5e5_complete.json
?? docs/experiments/20260616_dior_r_s3_stability_e4_lr2p5e5_complete.md
?? docs/experiments/20260616_dota2_s3_scene_adapter_loss0_best_complete.json
?? docs/experiments/20260616_dota2_s3_scene_adapter_loss0_best_complete.md
?? docs/experiments/20260616_result_analysis_and_claim_boundaries.md
```

The untracked package records DIOR-R S3 completion, DIOR-R S3 stability,
DOTA2 S3 completion, and claim-boundary analysis from 2026-06-14 through
2026-06-16. Existing modified assets and manuscript/support files are preserved
as-is; this archive action does not commit or revert them.

## Screen State Before Launch

Command: `screen -ls`

```text
There is a screen on:
        3470174.s0_result_log_monitor_20260603     (06/03/26 19:55:38)     (Detached)
1 Socket in /run/screen/S-zwl.
```

Only the long-running result-log monitor was active before this launch.

## GPU State Before Launch

Poll time: `2026-06-17 09:26:02 CST`

```text
0, 14, 0
1, 14, 0
2, 14, 0
3, 14, 0
4, 14, 0
5, 22607, 23
6, 14, 0
```

Follow-up polls:

```text
0, 14, 0
1, 14, 0
2, 14, 0
3, 14, 0
4, 14, 0
5, 22607, 13
6, 14, 0
```

```text
0, 14, 0
1, 14, 0
2, 14, 0
3, 14, 0
4, 14, 0
5, 22607, 72
6, 14, 0
```

GPUs 0, 1, and 2 were idle by memory/utilization. GPU 5 remained excluded.

## Config And Checkpoint Preflight

All requested paths existed before launch:

- DIOR-R S0 RoI Transformer config:
  `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_roi_trans_sanitized_long_20260612_gpu1/dior_r_s0_roi_trans_sanitized_long_20260612_gpu1.py`
- DIOR-R S0 RoI Transformer checkpoint:
  `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_roi_trans_sanitized_long_20260612_gpu1/epoch_52.pth`
- DIOR-R S2 hierarchy config:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep4_20260614/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-s1e12-rep4-20260614_dior_r.py`
- DIOR-R S2 hierarchy checkpoint:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep4_20260614/epoch_12.pth`
- DIOR-R S3 scene-adapter config:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_s2e12_rep0_20260614/roi-trans-le90_r50_fpn_remoteclip-s3-scene-adapter-s2e12-rep0-20260614_dior_r.py`
- DIOR-R S3 scene-adapter checkpoint:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_s2e12_rep0_20260614/epoch_8.pth`

## Analysis Jobs

Planned screen jobs under `/data5/2025/ldh/OpenRSD`:

| GPU | Screen | Workdir | Output |
| --- | --- | --- | --- |
| 0 | `paper_eval_dior_r_s0_epoch52_20260617_gpu0` | `work_dirs/paper_eval_20260617/dior_r_s0_roi_trans_epoch52` | `preds.pkl` |
| 1 | `paper_eval_dior_r_s2_rep4_epoch12_20260617_gpu1` | `work_dirs/paper_eval_20260617/dior_r_s2_rep4_epoch12` | `preds.pkl` |
| 2 | `paper_eval_dior_r_s3_rep0_epoch8_20260617_gpu2` | `work_dirs/paper_eval_20260617/dior_r_s3_rep0_epoch8` | `preds.pkl` |

## Startup Acceptance

Passed at `2026-06-17 09:29 CST`.

New screens:

```text
3344421.paper_eval_dior_r_s0_epoch52_20260617_gpu0
3344812.paper_eval_dior_r_s2_rep4_epoch12_20260617_gpu1
3345362.paper_eval_dior_r_s3_rep0_epoch8_20260617_gpu2
```

GPU residency after launch:

```text
0, 5065, 34
1, 4257, 29
2, 4707, 16
3, 14, 0
4, 14, 0
5, 22607, 43
6, 14, 0
```

Each launch log showed checkpoint loading and test progress:

- S0 loaded
  `work_dirs/dior_r_s0_roi_trans_sanitized_long_20260612_gpu1/epoch_52.pth`
  and reached `Epoch(test) [ 350/5869]`.
- S2 loaded
  `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_rep4_20260614/epoch_12.pth`
  and reached `Epoch(test) [ 750/5869]`.
- S3 loaded
  `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_s2e12_rep0_20260614/epoch_8.pth`
  and reached `Epoch(test) [ 700/5869]`.

Scoped startup failure scan was clean for `Traceback`, CUDA OOM,
`out of memory`, `libpng`, `CRC`, `NoneType`, `ValueError`, true `nan`, and
true `inf`.

## Completion Acceptance

Completed. Each workdir contains `preds.pkl`, a runtime `.log`, and a JSON
metric file under `/data5/2025/ldh/OpenRSD/work_dirs/paper_eval_20260617/`.

| Run | Workdir | JSON metrics |
| --- | --- | --- |
| S0 RoI Transformer epoch 52 | `dior_r_s0_roi_trans_epoch52` | `dota/mAP=0.654401421546936`, `dota/AP50=0.654` |
| S2 hierarchy rep4 epoch 12 | `dior_r_s2_rep4_epoch12` | `dota/mAP=0.6914003491401672`, `dota/AP50=0.691` |
| S3 scene-adapter rep0 epoch 8 | `dior_r_s3_rep0_epoch8` | `dota/mAP=0.6991876363754272`, `dota/AP50=0.699` |

Final artifacts:

- S0: `work_dirs/paper_eval_20260617/dior_r_s0_roi_trans_epoch52/preds.pkl`,
  `launch_20260617_gpu0.log`,
  `20260617_092751/20260617_092751.log`, and
  `20260617_092751/20260617_092751.json`.
- S2: `work_dirs/paper_eval_20260617/dior_r_s2_rep4_epoch12/preds.pkl`,
  `launch_20260617_gpu1.log`,
  `20260617_092757/20260617_092757.log`, and
  `20260617_092757/20260617_092757.json`.
- S3: `work_dirs/paper_eval_20260617/dior_r_s3_rep0_epoch8/preds.pkl`,
  `launch_20260617_gpu2.log`,
  `20260617_092800/20260617_092800.log`, and
  `20260617_092800/20260617_092800.json`.

Scoped completion failure scan across all launch and runtime logs was clean for
`Traceback`, CUDA OOM, `out-of-memory`, `out of memory`, `libpng`, `CRC`,
`NoneType`, `ValueError`, `KeyboardInterrupt`, `loss: nan`, `loss: inf`,
`grad_norm: nan`, and `grad_norm: inf`.
