# Experiment: strong_detector_sweep

Date: 2026-05-25

Status: running

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
- All three baselines are now launched in detached screen sessions.
- The DOTA v1.5 wrappers use `PackDetInputs` pipelines and a 640x640 resize
  to keep memory within the available GPU budget.
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

Status: running

GPU allocation: 1 GPU

Config:
- `OpenRSD/mmrotate_configs/strong_baseline_dota15/oriented-rcnn-le90_r50_fpn_amp-1x_dota15.py`

Training command:
- pending exact server command paste

Validation command:
- pending exact server command paste

External checkpoint path:
- pending

External log path:
- pending

Training status:
- pending server update

Validation result:
- pending

Class-wise AP:
- pending

Failure notes:
- none recorded yet

## RoI Transformer

Status: running

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
- pending server update

Validation result:
- pending

Class-wise AP:
- pending

Failure notes:
- none recorded yet

## ReDet

Status: running

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
- pending server update

Validation result:
- pending

Class-wise AP:
- pending

Failure notes:
- ReDet is a scratch run because the expected ReResNet checkpoint was missing;
  compare it cautiously against ImageNet-pretrained R50 baselines.

## Next Action

1. Update the three detector sections with exact server commands, screen names,
   logs, checkpoint paths, and training status.
2. Run validation for any completed checkpoint on the same DOTA v1.5 split.
3. Record mAP and class-wise AP before launching S1 flat prompt ablations.
4. If all three strong baselines fail, fix the MMRotate environment, dataset
   wrapper, class count, or resize/memory setup before touching prompt modules.
