# FAIR1M TPC/S1 Campaign Completion (2026-07-20)

This record closes the three-replica FAIR1M TPC/S1 campaign launched in
`20260717_fair1m_tpc_s1_campaign_launch.md` and archives analysis-only
evaluation of the selected checkpoints. No FAIR1M S2/GeoNexus training was
launched.

## Results

| Replica | Best checkpoint | Best mAP / AP50 | Final epoch-12 mAP / AP50 |
|---|---|---|---|
| rep3407 | `epoch_8.pth` | `0.3175 / 0.3170` | `0.3052 / 0.3050` |
| rep4407 | `epoch_4.pth` | `0.3179 / 0.3180` | `0.3095 / 0.3090` |
| rep5407 | `epoch_4.pth` | `0.3202 / 0.3200` | `0.3061 / 0.3060` |

The best-checkpoint mean is `0.318533` mAP and `0.318333` AP50. The
epoch-12 final mean is `0.306933` mAP and `0.306667` AP50. All three
campaigns completed cleanly through epoch 12.

## Checkpoints And Training Provenance

Training workdirs are:

- `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_fair1m/roi_trans_tpc_s1_rep3407_20260717`
- `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_fair1m/roi_trans_tpc_s1_rep4407_20260717`
- `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_fair1m/roi_trans_tpc_s1_rep5407_20260717`

Each retains `epoch_4.pth`, `epoch_8.pth`, `epoch_12.pth`, the replica
config, `launch_provenance.txt`, and the runtime log. The exact original
GPU/screen mapping was 3407/GPU 0/
`fair1m_tpc_s1_rep3407_20260717_gpu0`, 4407/GPU 1/
`fair1m_tpc_s1_rep4407_20260717_gpu1`, and 5407/GPU 2/
`fair1m_tpc_s1_rep5407_20260717_gpu2`.

## Analysis Evaluation

The selected checkpoints were evaluated with `tools/bootstrap_run.py
tools/test.py`, using the matching S1 config, `--work-dir`, and `--out
<workdir>/preds.pkl`. Host GPUs were 0, 1, and 2, with screens
`fair1m_s1_eval_rep3407_epoch8_20260720_gpu0`,
`fair1m_s1_eval_rep4407_final_20260720_gpu1`, and
`fair1m_s1_eval_rep5407_final_20260720_gpu2`.

Outputs are under
`/data5/2025/ldh/OpenRSD/work_dirs/paper_eval_20260720/`:

- `fair1m_tpc_s1_best_epoch8_rep3407`: `preds.pkl`, copied config,
  `20260720_094824/20260720_094824.json`, and evaluator log.
- `fair1m_tpc_s1_best_epoch4_rep4407`: `preds.pkl`, copied config,
  `20260720_093654/20260720_093654.json`, and evaluator log.
- `fair1m_tpc_s1_best_epoch4_rep5407`: `preds.pkl`, copied config,
  `20260720_093654/20260720_093654.json`, and evaluator log.

The evaluator JSON values are respectively `0.3174866736/0.317`,
`0.3178664148/0.318`, and `0.3202100992/0.320`; they match the training
logs at evaluator rounding. An additional epoch-4 rep3407 diagnostic is
preserved under `fair1m_tpc_s1_best_epoch4_rep3407` and correctly reports
`0.3121/0.3120`; it is not the selected best checkpoint.

## Failure Scan And Gate

Scoped scans of all three training and final evaluator logs found no
traceback, CUDA OOM, decode/CRC, invalid-box, NaN/Inf, or interruption
signatures. All final evaluator logs reached `Epoch(test) [5485/5485]` and
saved `preds.pkl`. Earlier pre-Python launch attempts with an obsolete
environment path and two mistyped config commands exited before model load;
their logs remain in the analysis workdirs and are not experiment failures.

FAIR1M S2/GeoNexus remains closed pending explicit route review. This record
does not modify the manuscript or open any unrelated route.
