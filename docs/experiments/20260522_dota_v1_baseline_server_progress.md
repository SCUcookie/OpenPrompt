# Experiment: dota_v1_baseline_server_progress

Date: 2026-05-22

Last updated: 2026-05-23

Git commit: 663439d

Machine: nuosen (server)

GPU: 1 x NVIDIA GeForce RTX 4090

Dataset: /data2/2023/lcs/xyun/DOTA (train/val; DOTA v1.0 labels for the first baseline run)

Config: `configs/experiments/dota_v1_baseline_repro.yaml`

Command:

```bash
source /data1/anaconda3/etc/profile.d/conda.sh && \
conda activate zwl_oneformer_ViT_P && \
cd /data5/2025/ldh/OpenPrompt && \
PYTHONPATH=src python scripts/train.py --config configs/experiments/dota_v1_baseline_repro.yaml
```

External checkpoint path: `outputs/dota_v1_baseline_repro/epoch_012.pt`

Detached screen session: `openprompt_dota_v1_baseline` (completed)

External log path: `outputs/openprompt_dota_v1_baseline/train.log`

Resume command:

```bash
screen -r openprompt_dota_v1_baseline
```

## Purpose

Start the first server-side baseline reproduction run on DOTA v1.0 and verify that the training pipeline is stable on the available GPU environment.

## Result

- Training completed successfully in `zwl_oneformer_ViT_P` inside a detached `screen` session.
- The run progressed past the first batches after disabling cuDNN in the shared runtime hook.
- All 12 epochs finished and checkpoints were written through `outputs/dota_v1_baseline_repro/epoch_012.pt`.
- Final epoch metrics: `loss=0.18908667655497577`, `loss_cls=0.0010149941903454095`, `loss_box=0.09403584113557859`, `positive_cls_acc=0.3241933747263396`, `positive_box_l1=0.18381485000583453`.
- The detached session successfully protected the run from SSH disconnects.
- Validation on the saved checkpoint completed on 4055 images with `map50=3.326794065590851e-06`, `mean_precision=0.00015695091957847277`, and `mean_recall=0.00037193994697493814`.

## Notes

- `dlp` is not suitable for training this repo on this server because it does not have `torch` installed.
- The RTX 4090 on this host segfaults on the cuDNN convolution path, so the repo now disables cuDNN in `seed_everything()`.
- Long runs should be started with `scripts/run_train_in_screen.sh` or an equivalent detached `screen` command.
- If the SSH session drops, reconnect with `screen -r openprompt_dota_v1_baseline`.
- Earlier dataset configuration issues were fixed before this run.

## Next Action

Inspect the scaffold localization path and decide whether the next repair should
feed `query_centers` into box regression or move to a stronger baseline before
any prompt/context ablations.

## Evaluation Gate

Status: completed on server execution.

Evaluation command:

```bash
source /data1/anaconda3/etc/profile.d/conda.sh && \
conda activate zwl_oneformer_ViT_P && \
cd /data5/2025/ldh/OpenPrompt && \
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/experiments/dota_v1_baseline_repro.yaml \
  --checkpoint outputs/dota_v1_baseline_repro/epoch_012.pt \
  --split val \
  --metric-set both \
  --score-threshold 0.05 \
  --nms-iou-threshold 0.3 \
  --max-detections 100 \
  | tee outputs/dota_v1_baseline_repro/eval_epoch_012_val.json
```

Evaluation artifact path: `outputs/dota_v1_baseline_repro/eval_epoch_012_val.json`

Required fields to verify:

- `split`: must be `val`
- `detection_metrics.num_eval_images`: must be greater than zero
- class AP keys such as `ap50_plane`: must be present
- aggregate `detection_metrics.map50`, `detection_metrics.mean_precision`, and `detection_metrics.mean_recall`: must be present

Record after evaluation:

- `map50`: `3.326794065590851e-06`
- `mean_precision`: `0.00015695091957847277`
- `mean_recall`: `0.00037193994697493814`
- notable per-class AP/recall failures: almost all classes are zero; only plane, ship, and tennis-court have tiny nonzero values
- nonzero-mAP gate result: passed

Gate decision:

- Proceed to DOTA v1.5 only if DOTA v1.0 validation `map50 > 0` and class-level AP/recall values are interpretable.
- The gate passed, but the baseline is still extremely weak and should be treated as a sanity-check baseline rather than a strong detector.
- If `map50 == 0` or nearly all recall is zero, do not launch DOTA v1.5. Diagnose validation tile/object counts, decoded prediction score distribution, predicted box scale/coordinates after tiling, lower score thresholds such as `0.01`, and DOTA v1.0 label/class mapping first.

If the gate passes, launch the matched DOTA v1.5 baseline:

```bash
bash scripts/run_train_in_screen.sh \
  openprompt_dota_v15_baseline \
  configs/experiments/dota_v15_baseline_repro.yaml \
  outputs/openprompt_dota_v15_baseline
```

## v1.5 Follow-up

Detailed follow-up record:
`docs/experiments/20260523_dota_v15_baseline_server_followup.md`

Detached screen session: `openprompt_dota_v15_baseline` (completed)

External log path: `outputs/openprompt_dota_v15_baseline/train.log`

Latest completed epoch: `12/12`

Training status: completed successfully.

Validation evaluation completed on `2026-05-24 11:26` for `outputs/dota_v15_baseline_repro/epoch_012.pt`.

Evaluation artifact path: `outputs/dota_v15_baseline_repro/eval_epoch_012_val.json`

Validation metrics: `map50=1.0926445202230628e-05`, `mean_precision=0.0006667361585641629`, `mean_recall=0.0011823561703749874`

Notable nonzero class values: harbor, plane, ship, small-vehicle, and tennis-court; all remain very weak.

Overall assessment: the v1.5 run is a valid sanity-check baseline, but it is still far below paper-quality detector performance.

Resume command:

```bash
screen -r openprompt_dota_v15_baseline
```

## Wrap-up

Current baseline flows are complete:

1. DOTA v1.0 baseline training completed.
2. DOTA v1.0 validation evaluation completed.
3. DOTA v1.5 baseline training completed.
4. DOTA v1.5 validation evaluation completed.

The next research step is not another baseline rerun; it is to repair the
localization scaffold first, then decide the first defensible ablation or
methodology improvement on top of the verified detector path.
