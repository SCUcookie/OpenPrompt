# Experiment: dota_v15_anchor_repair_quick_test

Date: 2026-05-24

Status: completed

Purpose: run a very short anchor-repair smoke test on top of the current DOTA
v1.5 scaffold by feeding query centers into box regression, then evaluate the
first checkpoint with the same validation settings used for
`dota_v15_baseline_repro`.

Config:
`configs/experiments/dota_v15_anchor_repair.yaml`

Training command:

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/experiments/dota_v15_anchor_repair.yaml
```

Validation command:

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/experiments/dota_v15_anchor_repair.yaml \
  --checkpoint outputs/dota_v15_anchor_repair/epoch_001.pt \
  --split val \
  --metric-set both \
  --score-threshold 0.05 \
  --nms-iou-threshold 0.3 \
  --max-detections 100
```

Diagnosis command:

```bash
PYTHONPATH=src python scripts/diagnose_baseline.py \
  --config configs/experiments/dota_v15_anchor_repair.yaml \
  --checkpoint outputs/dota_v15_anchor_repair/epoch_001.pt \
  --split val \
  --score-thresholds 0.05,0.01,0.001 \
  --nms-iou-threshold 0.3 \
  --max-detections 100 \
  --output outputs/dota_v15_anchor_repair/diagnostics_epoch_001_val.json
```

Result:

- The 1-epoch smoke test completed and wrote `outputs/dota_v15_anchor_repair/epoch_001.pt` and `outputs/dota_v15_anchor_repair/last.pt`.
- Final training metrics from `outputs/dota_v15_anchor_repair/metrics.json` were `loss=0.07363908355801901`, `loss_cls=0.001671954903589549`, `loss_box=0.035983564312892485`, `loss_hier=0.0`, `loss_margin=0.0`, `positive_cls_acc=0.5529336195676059`, and `positive_box_l1=0.10294117139314753`.
- The run is archived as a smoke test only; no follow-up validation or diagnosis was run after the request to stop further compute.

Continue only if a later session explicitly resumes validation and the diagnostics show less center bias, better best-IoU summaries, and nontrivial recall/mAP improvement over the current scaffold baseline.

Parallel checklist: `docs/setup/strong_baseline_checklist.md`

Detached screen session: `openprompt_dota_v15_anchor_repair`

Current log: `outputs/openprompt_dota_v15_anchor_repair/train.log`