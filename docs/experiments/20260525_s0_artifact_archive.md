# S0 Artifact Archive: DOTA v1.5 Strong Baselines

Date: 2026-05-26

Purpose: index S0 detector artifacts that exist on disk but were only partially
captured in the main run summaries. Large checkpoints and logs remain in
`/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/`; this file records
what each directory means and whether it should be used for paper-facing
numbers.

## Completed 3x Runs

### RoI Transformer 3x

- Status: completed
- GPU: 2
- Config: `/data5/2025/ldh/OpenRSD/mmrotate_configs/strong_baseline_dota15/roi-trans-le90_r50_fpn_amp-3x_dota15.py`
- Work dir: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/roi_trans_lr001_3x`
- Log: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/roi_trans_lr001_3x/train.log`
- Best checkpoint: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/roi_trans_lr001_3x/epoch_34.pth`
- Metric summary: `docs/experiments/20260526_roi_transformer_3x_dota15_metrics.json`
- Best validation: epoch 34, `dota/mAP=0.2644`, `dota/AP50=0.2640`
- Final validation: epoch 36, `dota/mAP=0.2612`, `dota/AP50=0.2610`

Paper-facing status: primary S0 detector baseline and preferred fixed detector
checkpoint for S1, unless simplicity/stability is prioritized over the small
mAP lead.

### Oriented R-CNN 3x

- Status: completed
- GPU: 4
- Config: `/data5/2025/ldh/OpenRSD/mmrotate_configs/strong_baseline_dota15/oriented-rcnn-le90_r50_fpn_amp-3x_dota15.py`
- Work dir: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/oriented_rcnn_3x_loadfrom`
- Log: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/oriented_rcnn_3x_loadfrom/train.log`
- Initialization: loaded weights from `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/oriented_rcnn/epoch_12.pth`
- Best checkpoint recorded: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/oriented_rcnn_3x_loadfrom/epoch_33.pth`
- Metric summary: `docs/experiments/20260526_oriented_rcnn_3x_dota15_metrics.json`
- Best validation: epochs 33 and 34, `dota/mAP=0.2620`, `dota/AP50=0.2620`
- Final validation: epoch 36, `dota/mAP=0.2607`, `dota/AP50=0.2610`
- Launch command:

```bash
CUDA_VISIBLE_DEVICES=4 ./run_bootstrap.sh tools/train.py \
  mmrotate_configs/strong_baseline_dota15/oriented-rcnn-le90_r50_fpn_amp-3x_dota15.py \
  --work-dir work_dirs/strong_baseline_dota15/oriented_rcnn_3x_loadfrom \
  --cfg-options load_from=work_dirs/strong_baseline_dota15/oriented_rcnn/epoch_12.pth
```

Paper-facing status: close secondary S0 detector baseline.

### ReDet pretrained

- Status: completed
- GPU: 2
- Config: `/data5/2025/ldh/OpenRSD/mmrotate_configs/strong_baseline_dota15/redet-le90_re50_refpn_amp-1x_dota15_pretrained.py`
- Work dir: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/redet_pretrained_rerun`
- Log: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/redet_pretrained_rerun/train.log`
- Pretrained checkpoint: `/data5/2025/temp/Supplements/re_resnet50_c8_batch256-25b16846.pth`
- Final checkpoint: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/redet_pretrained_rerun/epoch_12.pth`
- Metric summary: `docs/experiments/20260526_redet_pretrained_dota15_metrics.json`
- Best/final validation: epoch 12, `dota/mAP=0.2382`, `dota/AP50=0.2380`

Paper-facing status: completed comparison baseline.

## Aborted Or Superseded Runs

### Oriented R-CNN 3x, stopped `--resume` attempt

- Status: stopped intentionally
- Work dir: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/oriented_rcnn_3x`
- Log: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/oriented_rcnn_3x/train.log`
- Reason: launched with `--resume work_dirs/strong_baseline_dota15/oriented_rcnn/epoch_12.pth`, which restored the old 12-epoch scheduler state.
- Evidence: run resumed at epoch 12 / iter 16920 and started epoch 13 with
  `lr: 5.0000e-05`, not the intended 3x schedule.
- Stopped after roughly epoch 13 iteration 450.

Paper-facing status: do not use. Kept only as a launch audit trail.

### ReDet scratch early smoke

- Status: failed/superseded
- Work dir: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/redet`
- Checkpoint: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/redet/epoch_1.pth`
- Log: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/redet/train.log`
- Failure mode: validation dataloader failed because `PackDetInputs` expected
  `flip`, but the validation pipeline did not create it.
- Superseded by: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/redet_scratch_rerun`

Paper-facing status: do not use. This was a pipeline repair smoke run.

### RoI Transformer early smoke

- Status: failed/superseded
- Work dir: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/roi_trans`
- Checkpoint: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/roi_trans/epoch_1.pth`
- Log: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/roi_trans/train.log`
- Failure mode: training produced `nan` losses during epoch 1, then validation
  hit the same missing-`flip` pipeline assertion.
- Superseded by: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/roi_trans_lr001_rerun`

Paper-facing status: do not use. This was a failed high-LR/pipeline smoke run.

## Completed Baseline Reference Points

These are already covered by the main run summary and metric JSON files, but
are listed here to anchor the archive:

- Oriented R-CNN 12e: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/oriented_rcnn/epoch_12.pth`, mAP `0.2561`
- RoI Transformer 3x: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/roi_trans_lr001_3x/epoch_34.pth`, mAP `0.2644`
- Oriented R-CNN 3x: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/oriented_rcnn_3x_loadfrom/epoch_33.pth`, mAP `0.2620`
- ReDet pretrained 12e: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/redet_pretrained_rerun/epoch_12.pth`, mAP `0.2382`
- ReDet scratch 12e, superseded by pretrained: `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/redet_scratch_rerun/epoch_12.pth`, mAP `0.1221`

## Operational Notes

- Keep using idle GPUs only.
- Do not start S1/S2 VLM runs until real VLM embedding support passes the
  16-class smoke test. `/data1/anaconda3/envs/zwl_mmrotate/bin/python` has
  `torch`, but `open_clip` and `clip` are currently missing.
- For long-run continuation configs, prefer `load_from=<checkpoint>` over
  `--resume <checkpoint>` when the schedule changes. `--resume` restores the
  old optimizer and scheduler state.
- If `screen -ls` reports stale or dead sockets from inside the sandbox, verify
  liveness with fresh log timestamps and `nvidia-smi` before assuming the job
  stopped.
- After any active run finishes, export a small metric JSON under
  `/data5/2025/ldh/New/docs/experiments/` before promoting it to a paper-facing
  comparison.
