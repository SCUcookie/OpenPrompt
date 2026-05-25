# Experiment: strong_detector_sweep

Date: 2026-05-25

Status: mostly complete (Oriented R-CNN and ReDet completed; RoI Transformer
stable through epoch 11, epoch 12 not written at latest check)

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
- ReDet: 1 GPU for the current scratch rerun because only GPUs 2 and 4 were
  free at relaunch time; keep the original 2-GPU distributed plan for a later
  throughput rerun if resources free up
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
- RoI Transformer low-LR rerun is usable through epoch 11, with best observed
  validation at epoch 10.
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

Mid-run/final records:
- `docs/experiments/20260525_strong_detector_midrun_records.md`

## RoI Transformer

Status: stable rerun usable through epoch 11; epoch 12 pending/stalled at latest check

GPU allocation: 1 GPU (`CUDA_VISIBLE_DEVICES=2`)

Config:
- `OpenRSD/mmrotate_configs/strong_baseline_dota15/roi-trans-le90_r50_fpn_amp-1x_dota15.py`

Training command:

```bash
CUDA_VISIBLE_DEVICES=2 /data1/anaconda3/envs/zwl_mmrotate/bin/python \
  tools/bootstrap_run.py tools/train.py \
  mmrotate_configs/strong_baseline_dota15/roi-trans-le90_r50_fpn_amp-1x_dota15.py \
  --work-dir work_dirs/strong_baseline_dota15/roi_trans_lr001_rerun
```

Validation command:
- pending exact server command paste

External checkpoint path:
- best observed checkpoint:
  `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/roi_trans_lr001_rerun/epoch_10.pth`
- active work dir:
  `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/roi_trans_lr001_rerun/`

External log path:
- `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/roi_trans_lr001_rerun/train.log`

Training status:
- Previous 1-epoch run reached `epoch_1.pth` but losses became NaN by epoch 1 with the earlier `lr=0.005` setting.
- Wrapper LR has been lowered to `0.001`.
- The DOTA v1.5 wrapper now overrides both cascade bbox heads to
  `num_classes=16`; a shallow list override was rejected because MMEngine
  replaces list entries instead of deep-merging them.
- Relaunched at 2026-05-25 20:07 server time in screen session
  `geonexus_roi_trans_lr001`.
- Training and validation were stable through epoch 11. Best observed validation
  was epoch 10 with `dota/mAP=0.2485`, `dota/AP50=0.2480`.
- Latest visible log reached epoch 12 iter 1280/1410, but `epoch_12.pth` had
  not been written and the screen hardcopy was blank. Treat epoch 12 as
  pending/stalled until manually verified.

Validation result:
- Usable through epoch 11. Best observed checkpoint is epoch 10.

Class-wise AP:
- Epoch 11 class-wise AP is recorded in
  `docs/experiments/20260525_strong_detector_run_summary.md`; epoch 10
  class-wise AP still needs to be extracted if it is used as the table row.

Failure notes:
- The earlier LR `0.005` run diverged. The LR `0.001` rerun did not show NaN
  in the visible log. If epoch 12 remains stalled, use epoch 10 as the best
  observed RoI Transformer checkpoint or relaunch from epoch 11.

## ReDet

Status: completed scratch rerun

GPU allocation: 1 GPU (`CUDA_VISIBLE_DEVICES=4`) for the current rerun

Config:
- `OpenRSD/mmrotate_configs/strong_baseline_dota15/redet-le90_re50_refpn_amp-1x_dota15.py`

Training command:

```bash
CUDA_VISIBLE_DEVICES=4 /data1/anaconda3/envs/zwl_mmrotate/bin/python \
  tools/bootstrap_run.py tools/train.py \
  mmrotate_configs/strong_baseline_dota15/redet-le90_re50_refpn_amp-1x_dota15.py \
  --work-dir work_dirs/strong_baseline_dota15/redet_scratch_rerun
```

Validation command:
- pending exact server command paste

External checkpoint path:
- `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/redet_scratch_rerun/epoch_12.pth`

External log path:
- `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/redet_scratch_rerun/train.log`

Training status:
- Previous 1-epoch scratch run reached `epoch_1.pth` with finite losses, but it used the earlier validation pipeline and is not final S0 evidence.
- Relaunched at 2026-05-25 20:07 server time in screen session
  `geonexus_redet_scratch`.
- Startup check: training reached epoch 1 iter 300 with finite losses after an
  initial `grad_norm: nan` at iter 50; monitor before counting any checkpoint.
- Completed 12 epochs. Final checkpoint:
  `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/redet_scratch_rerun/epoch_12.pth`.
- Final validation at epoch 12: `dota/mAP=0.1221`, `dota/AP50=0.1220`.
- Best validation in this scratch rerun is epoch 12 so far, but compare
  cautiously because ReDet was initialized from scratch.

Validation result:
- Final scratch rerun validation at epoch 12: `dota/mAP=0.1221`,
  `dota/AP50=0.1220`.
- Previous corrected-metadata-only spot validation before the pipeline-order fix
  was `dota/mAP=0.0001`, `dota/AP50=0.0000`; treat this as
  invalid/diagnostic because annotations were loaded before resize.

Class-wise AP:
- Epoch 12 class-wise AP is recorded in
  `docs/experiments/20260525_strong_detector_run_summary.md`.

Failure notes:
- ReDet is a scratch run because the expected ReResNet checkpoint was missing; compare it cautiously against ImageNet-pretrained R50 baselines.
- Rerun/revalidate with `LoadImageFromFile -> Resize -> LoadAnnotations -> ConvertBoxType -> RandomFlip(prob=0) -> PackDetInputs` before using any ReDet number.

## Next Action

1. Verify whether the RoI Transformer screen session is stalled. If stalled,
   use epoch 10 as the best observed RoI Transformer baseline or relaunch from
   epoch 11.
2. Download/stage ReResNet pretraining before rerunning ReDet as a fair
   baseline.
3. Use Oriented R-CNN epoch 12 as the first S0 gate for S1-S4 prompt/VLM
   ablations.
4. Follow `docs/setup/complete_experiment_plan.md` for paper table closure.
5. Keep prompt/VLM ablations separate from closed-set detector results and do
   not claim open-vocabulary behavior from S0.
