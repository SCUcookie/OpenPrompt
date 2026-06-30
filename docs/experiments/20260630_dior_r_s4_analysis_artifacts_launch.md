# 2026-06-30 DIOR-R S4 Analysis Artifacts Launch

## Scope

Create qualitative, confidence-ranked paper artifacts from the completed
DIOR-R S4 low-LR paper-eval outputs. This is analysis-only: no new training,
no S4 continuation, no FAIR1M work, and no route-change claim.

Source record:
`New/docs/experiments/20260629_dior_r_s4_paper_eval_best_launch.md`.

Interpretation boundary: these artifacts provide qualitative support only. The
June 29 metrics keep S4 classified as weak stabilization, not superiority:

| Replica | Paper-eval `dota/mAP` | Paper-eval `dota/AP50` |
| --- | ---: | ---: |
| rep23407 epoch 2 | 0.6935 | 0.6930 |
| rep24407 epoch 6 | 0.6966 | 0.6970 |
| rep25407 epoch 2 | 0.6967 | 0.6970 |

## Preflight

Preflight at `2026-06-30 09:36 CST`:

- OpenRSD git hash: `12d3fd8b75e8b64ec53fded9cf035a2306d58874`.
- OpenRSD tree was dirty before this run; unrelated modified and untracked
  files were left untouched.
- `screen -ls` showed only `s0_result_log_monitor_20260603`.
- GPU state was recorded for provenance only; no GPU allocation is required.
  Rendering is forced to CPU with `CUDA_VISIBLE_DEVICES=`.
- Source configs and `preds.pkl` files exist for all three selected paper-eval
  outputs.
- Each `preds.pkl` loaded as a list with 11,738 samples; sample records include
  both `pred_instances` and `gt_instances`.

GPU state:

| GPU | Name | Memory | Utilization |
| ---: | --- | ---: | ---: |
| 0 | NVIDIA GeForce RTX 4090 | 14 MiB | 0% |
| 1 | NVIDIA GeForce RTX 4090 | 5355 MiB | 23% |
| 2 | NVIDIA GeForce RTX 4090 | 14 MiB | 0% |
| 3 | NVIDIA GeForce RTX 4090 | 14 MiB | 0% |
| 4 | NVIDIA GeForce RTX 4090 | 14 MiB | 0% |
| 5 | NVIDIA GeForce RTX 4090 | 14 MiB | 0% |
| 6 | NVIDIA GeForce RTX 4090 | 22619 MiB | 100% |

## Planned Render Jobs

Renderer:
`New/scripts/render_mmrotate_qualitative_confidence.py`.

Common options: `--topk 5 --score-thr 0.3`.

| Replica | Config | Predictions | Output |
| --- | --- | --- | --- |
| rep23407 epoch 2 | `work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep23407_epoch2/roi-trans-le90_r50_fpn_remoteclip-s4-pseudo-agreement-e1-lr1e5-rep23407-20260628_dior_r.py` | `work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep23407_epoch2/preds.pkl` | `work_dirs/paper_artifacts_20260630/dior_r_s4_e1_lr1e5_rep23407_epoch2/qualitative_confidence` |
| rep24407 epoch 6 | `work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep24407_epoch6/roi-trans-le90_r50_fpn_remoteclip-s4-pseudo-agreement-e1-lr1e5-rep24407-20260628_dior_r.py` | `work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep24407_epoch6/preds.pkl` | `work_dirs/paper_artifacts_20260630/dior_r_s4_e1_lr1e5_rep24407_epoch6/qualitative_confidence` |
| rep25407 epoch 2 | `work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep25407_epoch2/roi-trans-le90_r50_fpn_remoteclip-s4-pseudo-agreement-e1-lr1e5-rep25407-20260628_dior_r.py` | `work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep25407_epoch2/preds.pkl` | `work_dirs/paper_artifacts_20260630/dior_r_s4_e1_lr1e5_rep25407_epoch2/qualitative_confidence` |

## Completion Criteria

- Detached render screens either exist at startup or complete quickly with log
  files written.
- Logs are clean for `Traceback`, CUDA OOM, `out-of-memory`, `out of memory`,
  `libpng`, `CRC`, `NoneType`, `ValueError`, and `KeyboardInterrupt`.
- Each output directory contains `ranking_summary.txt`.
- Each replica renders exactly 5 PNGs under `good/` and 5 PNGs under `bad/`.
- Final record preserves the June 29 metrics and the qualitative-only claim
  boundary.

## Launch Trail

Launched detached CPU render screens from `/data5/2025/ldh/OpenRSD` at
`2026-06-30 09:37 CST`.

| Replica | Screen | Render log | Output |
| --- | --- | --- | --- |
| rep23407 epoch 2 | `paper_artifacts_dior_r_s4_rep23407_e2_20260630_cpu` | `work_dirs/paper_artifacts_20260630/dior_r_s4_e1_lr1e5_rep23407_epoch2/render_20260630.log` | `work_dirs/paper_artifacts_20260630/dior_r_s4_e1_lr1e5_rep23407_epoch2/qualitative_confidence` |
| rep24407 epoch 6 | `paper_artifacts_dior_r_s4_rep24407_e6_20260630_cpu` | `work_dirs/paper_artifacts_20260630/dior_r_s4_e1_lr1e5_rep24407_epoch6/render_20260630.log` | `work_dirs/paper_artifacts_20260630/dior_r_s4_e1_lr1e5_rep24407_epoch6/qualitative_confidence` |
| rep25407 epoch 2 | `paper_artifacts_dior_r_s4_rep25407_e2_20260630_cpu` | `work_dirs/paper_artifacts_20260630/dior_r_s4_e1_lr1e5_rep25407_epoch2/render_20260630.log` | `work_dirs/paper_artifacts_20260630/dior_r_s4_e1_lr1e5_rep25407_epoch2/qualitative_confidence` |

Commands:

```bash
screen -dmS paper_artifacts_dior_r_s4_rep23407_e2_20260630_cpu bash -lc 'cd /data5/2025/ldh/OpenRSD && mkdir -p work_dirs/paper_artifacts_20260630/dior_r_s4_e1_lr1e5_rep23407_epoch2 && CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py /data5/2025/ldh/New/scripts/render_mmrotate_qualitative_confidence.py work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep23407_epoch2/roi-trans-le90_r50_fpn_remoteclip-s4-pseudo-agreement-e1-lr1e5-rep23407-20260628_dior_r.py work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep23407_epoch2/preds.pkl work_dirs/paper_artifacts_20260630/dior_r_s4_e1_lr1e5_rep23407_epoch2/qualitative_confidence --topk 5 --score-thr 0.3 > work_dirs/paper_artifacts_20260630/dior_r_s4_e1_lr1e5_rep23407_epoch2/render_20260630.log 2>&1'
```

```bash
screen -dmS paper_artifacts_dior_r_s4_rep24407_e6_20260630_cpu bash -lc 'cd /data5/2025/ldh/OpenRSD && mkdir -p work_dirs/paper_artifacts_20260630/dior_r_s4_e1_lr1e5_rep24407_epoch6 && CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py /data5/2025/ldh/New/scripts/render_mmrotate_qualitative_confidence.py work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep24407_epoch6/roi-trans-le90_r50_fpn_remoteclip-s4-pseudo-agreement-e1-lr1e5-rep24407-20260628_dior_r.py work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep24407_epoch6/preds.pkl work_dirs/paper_artifacts_20260630/dior_r_s4_e1_lr1e5_rep24407_epoch6/qualitative_confidence --topk 5 --score-thr 0.3 > work_dirs/paper_artifacts_20260630/dior_r_s4_e1_lr1e5_rep24407_epoch6/render_20260630.log 2>&1'
```

```bash
screen -dmS paper_artifacts_dior_r_s4_rep25407_e2_20260630_cpu bash -lc 'cd /data5/2025/ldh/OpenRSD && mkdir -p work_dirs/paper_artifacts_20260630/dior_r_s4_e1_lr1e5_rep25407_epoch2 && CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py /data5/2025/ldh/New/scripts/render_mmrotate_qualitative_confidence.py work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep25407_epoch2/roi-trans-le90_r50_fpn_remoteclip-s4-pseudo-agreement-e1-lr1e5-rep25407-20260628_dior_r.py work_dirs/paper_eval_20260629/dior_r_s4_e1_lr1e5_rep25407_epoch2/preds.pkl work_dirs/paper_artifacts_20260630/dior_r_s4_e1_lr1e5_rep25407_epoch2/qualitative_confidence --topk 5 --score-thr 0.3 > work_dirs/paper_artifacts_20260630/dior_r_s4_e1_lr1e5_rep25407_epoch2/render_20260630.log 2>&1'
```

Startup acceptance at `2026-06-30 09:37 CST`:

- Screens detached and alive for all three render jobs.
- Log files were created for all three render jobs.
- Initial logs were empty while startup/import/render work was still in
  progress.

## Rerun Note

The first render attempt exited quickly on all three replicas with the same
visualizer assertion:

- `AssertionError: The length of palette should not be less than num_classes.`

The renderer was patched to normalize and extend the dataset palette to cover
the configured classes and observed label range. No prediction files, configs,
or metric artifacts were changed. The same three CPU-only render commands were
relaunched at `2026-06-30 09:40 CST`, overwriting the failed render logs.

## Completion Verification

All three analysis-artifact render jobs completed on
`2026-06-30 09:41 CST`.

Artifacts:

| Replica | Render log | Ranking summary | Rendered PNGs |
| --- | --- | --- | --- |
| rep23407 epoch 2 | `work_dirs/paper_artifacts_20260630/dior_r_s4_e1_lr1e5_rep23407_epoch2/render_20260630.log` | `work_dirs/paper_artifacts_20260630/dior_r_s4_e1_lr1e5_rep23407_epoch2/qualitative_confidence/ranking_summary.txt` | 5 `good` + 5 `bad` |
| rep24407 epoch 6 | `work_dirs/paper_artifacts_20260630/dior_r_s4_e1_lr1e5_rep24407_epoch6/render_20260630.log` | `work_dirs/paper_artifacts_20260630/dior_r_s4_e1_lr1e5_rep24407_epoch6/qualitative_confidence/ranking_summary.txt` | 5 `good` + 5 `bad` |
| rep25407 epoch 2 | `work_dirs/paper_artifacts_20260630/dior_r_s4_e1_lr1e5_rep25407_epoch2/render_20260630.log` | `work_dirs/paper_artifacts_20260630/dior_r_s4_e1_lr1e5_rep25407_epoch2/qualitative_confidence/ranking_summary.txt` | 5 `good` + 5 `bad` |

Final screen state:

- `screen -ls` returned to only `s0_result_log_monitor_20260603`.

Scoped final failure scan across the three render logs was clean for:

- `Traceback`
- CUDA OOM
- `out-of-memory`
- `out of memory`
- `libpng`
- `CRC`
- `NoneType`
- `ValueError`
- `KeyboardInterrupt`

Final interpretation:

- These outputs are qualitative S4 paper artifacts only.
- The preserved June 29 metrics are `0.6935`, `0.6966`, and `0.6967` mAP.
- S4 remains weak stabilization, not a superiority result.
