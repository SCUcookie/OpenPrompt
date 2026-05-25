# Experiment: strong_detector_sweep

Date: 2026-05-25

Status: running (Oriented R-CNN completed; RoI Transformer and ReDet pending rerun)

Purpose: start the ordered strong-detector sweep for the current DOTA v1.5 path.
The sweep order is Oriented R-CNN -> RoI Transformer -> ReDet, with the first
wave launched in parallel according to the available GPU count.

This record is the S0 gate for later prompt/VLM ablations. Do not start S1-S5
paper-facing ablations from the local TinyBackbone/hash-embedding scaffold. The
completed `dota_v15_anchor_repair` run remains archived as smoke-test evidence
only until a later validation shows reduced center bias, improved best-IoU, and
nontrivial recall/mAP.

Available GPUs on this host:
- 7 x NVIDIA GeForce RTX 4090

Current launch split:
- Oriented R-CNN: 1 GPU
- RoI Transformer: 1 GPU
- ReDet: 2 GPUs, distributed launch from the bootstrap wrapper
- remaining GPUs: validation, retries, or second seed

Configs:
- `OpenRSD/mmrotate_configs/strong_baseline_dota15/oriented-rcnn-le90_r50_fpn_amp-1x_dota15.py`
- `OpenRSD/mmrotate_configs/strong_baseline_dota15/roi-trans-le90_r50_fpn_amp-1x_dota15.py`
- `OpenRSD/mmrotate_configs/strong_baseline_dota15/redet-le90_re50_refpn_amp-1x_dota15.py`

Data root:
- `/data5/2025/ldh/OpenPrompt/DOTA/`

Dataset version and split:
- DOTA v1.5
- use the same train/val split as the recorded `dota_v15_baseline_repro`
  scaffold baseline unless a later server note documents a forced switch

Class mapping:
- 16 DOTA v1.5 classes, including `container-crane`

Metric implementation:
- MMRotate / DOTA-style oriented detection validation for the strong detector
  path
- keep this separate from the local scaffold reduced-tile evaluator

Embedding backend:
- none for S0 closed-set detector baselines
- hash fallback and real VLM embeddings must be recorded separately for later
  prompt/VLM ablations

Notes:
- The first Oriented R-CNN baseline completed 12 epochs and produced a usable S0 result.
- The DOTA v1.5 wrappers use `PackDetInputs` pipelines and a 640x640 resize
  to keep memory within the available GPU budget.
- Validation/test pipelines must resize the image before `LoadAnnotations`; loading annotations before resize produced near-zero AP by resizing GT boxes into the wrong evaluator coordinates.
- `PackDetInputs` in val/test uses explicit `meta_keys=(img_id, img_path, ori_shape, img_shape, scale_factor)` so no missing `flip` metadata is required.
- ReDet is initialized from scratch by clearing the backbone `init_cfg` in the
  wrapper instead of loading the missing ReResNet checkpoint, and the scratch
  run uses a lower AMP learning rate to avoid NaN divergence.

## Common Validation Gate

Each detector record must be completed before it can support paper-facing S0
claims:

- training completes without NaN or divergence
- checkpoint path is recorded outside Git
- validation runs on the same DOTA v1.5 split
- overall mAP and per-class AP are copied into this record or a linked small
  summary file
- failures are fixed in the detector environment/config before changing prompt
  modules

## Oriented R-CNN

Status: completed baseline run

GPU allocation: 1 GPU (`CUDA_VISIBLE_DEVICES=1`)

Config:
- `OpenRSD/mmrotate_configs/strong_baseline_dota15/oriented-rcnn-le90_r50_fpn_amp-1x_dota15.py`

Training command:

```bash
CUDA_VISIBLE_DEVICES=1 /data1/anaconda3/envs/zwl_mmrotate/bin/python   tools/bootstrap_run.py tools/train.py   mmrotate_configs/strong_baseline_dota15/oriented-rcnn-le90_r50_fpn_amp-1x_dota15.py   --work-dir work_dirs/strong_baseline_dota15/oriented_rcnn   --resume work_dirs/strong_baseline_dota15/oriented_rcnn/epoch_3.pth
```

Validation command used for the corrected epoch-3 spot check:

```bash
CUDA_VISIBLE_DEVICES=1 /data1/anaconda3/envs/zwl_mmrotate/bin/python   tools/bootstrap_run.py tools/test.py   mmrotate_configs/strong_baseline_dota15/oriented-rcnn-le90_r50_fpn_amp-1x_dota15.py   work_dirs/strong_baseline_dota15/oriented_rcnn/epoch_3.pth   --work-dir work_dirs/strong_baseline_dota15/oriented_rcnn_eval_epoch3_fixedval
```

Final checkpoint path:
- `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/oriented_rcnn/epoch_12.pth`

External log path:
- `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/oriented_rcnn/20260525_163043/20260525_163043.log`

Small metric summary:
- `docs/experiments/20260525_oriented_rcnn_dota15_epoch12_metrics.json`

Full run summary and OpenRSD change record:
- `docs/experiments/20260525_strong_detector_run_summary.md`

Training status:
- Completed 12 epochs without NaN/divergence after fixing the validation pipeline.
- Corrected epoch-3 spot validation: `dota/mAP=0.1725`, `dota/AP50=0.1720`.
- Epoch 4: `dota/mAP=0.1898`, `dota/AP50=0.1900`.
- Epoch 5: `dota/mAP=0.2029`, `dota/AP50=0.2030`.
- Epoch 6: `dota/mAP=0.2066`, `dota/AP50=0.2070`.
- Epoch 7: `dota/mAP=0.2238`, `dota/AP50=0.2240`.
- Epoch 8: `dota/mAP=0.2298`, `dota/AP50=0.2300`.
- Epoch 9: `dota/mAP=0.2511`, `dota/AP50=0.2510`.
- Epoch 10: `dota/mAP=0.2548`, `dota/AP50=0.2550`.
- Epoch 11: `dota/mAP=0.2525`, `dota/AP50=0.2530`.
- Epoch 12: `dota/mAP=0.2561`, `dota/AP50=0.2560`.

Final class-wise AP at epoch 12:
- plane `0.353`, baseball-diamond `0.337`, bridge `0.091`, ground-track-field `0.361`, small-vehicle `0.091`, large-vehicle `0.421`, ship `0.091`, tennis-court `0.727`, basketball-court `0.310`, storage-tank `0.091`, soccer-ball-field `0.363`, roundabout `0.107`, harbor `0.422`, swimming-pool `0.160`, helicopter `0.176`, container-crane `0.000`.

Failure notes:
- Initial near-zero validation was a validation-pipeline bug: annotations were loaded before resize, so GT boxes were resized into the wrong evaluator coordinates. The corrected val/test order is `LoadImageFromFile -> Resize -> LoadAnnotations -> ConvertBoxType -> RandomFlip(prob=0) -> PackDetInputs`.
- `tools/test.py` was made compatible with this installed `mmdet` by guarding the unavailable `DumpDetResults` import; this is only needed when running standalone validation.

## RoI Transformer

Status: failed preliminary run; pending stable rerun

GPU allocation: 1 GPU

Config:
- `OpenRSD/mmrotate_configs/strong_baseline_dota15/roi-trans-le90_r50_fpn_amp-1x_dota15.py`

Training command:
- pending exact server command paste

Validation command:
- pending exact server command paste

External checkpoint path:
- pending

External log path:
- pending

Training status:
- Previous 1-epoch run reached `epoch_1.pth` but losses became NaN by epoch 1 with the earlier `lr=0.005` setting.
- Wrapper LR has been lowered to `0.001`, but the detector has not been rerun after the validation-pipeline fix.

Validation result:
- Not counted. The previous checkpoint is not usable paper evidence because training diverged.

Class-wise AP:
- Not counted.

Failure notes:
- Fix order: rerun with lowered LR or disable AMP; only validate checkpoints whose training logs stay finite.

## ReDet

Status: preliminary scratch checkpoint only; pending rerun/revalidation

GPU allocation: 2 GPUs, distributed launch from the bootstrap wrapper

Config:
- `OpenRSD/mmrotate_configs/strong_baseline_dota15/redet-le90_re50_refpn_amp-1x_dota15.py`

Training command:
- pending exact server command paste

Validation command:
- pending exact server command paste

External checkpoint path:
- pending

External log path:
- pending

Training status:
- Previous 1-epoch scratch run reached `epoch_1.pth` with finite losses, but it used the earlier validation pipeline and is not final S0 evidence.

Validation result:
- Previous corrected-metadata-only spot validation before the pipeline-order fix was `dota/mAP=0.0001`, `dota/AP50=0.0000`; treat this as invalid/diagnostic because annotations were loaded before resize.

Class-wise AP:
- Not counted until rerun or revalidated with the corrected val/test order.

Failure notes:
- ReDet is a scratch run because the expected ReResNet checkpoint was missing; compare it cautiously against ImageNet-pretrained R50 baselines.
- Rerun/revalidate with `LoadImageFromFile -> Resize -> LoadAnnotations -> ConvertBoxType -> RandomFlip(prob=0) -> PackDetInputs` before using any ReDet number.

## Next Action

1. Rerun RoI Transformer with the lowered LR wrapper or another stable setting; the earlier run diverged to NaN.
2. Rerun or revalidate ReDet with the corrected val/test pipeline, noting it is scratch unless the ReResNet checkpoint is restored.
3. Decide whether Oriented R-CNN epoch 12 is sufficient as the first S0 gate for S1 flat-prompt ablations, or whether to wait for a second detector.
4. Keep prompt/VLM ablations separate from this closed-set detector result and do not claim open-vocabulary behavior from S0.
