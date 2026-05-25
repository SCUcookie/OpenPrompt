# Experiment: strong_detector_sweep

Date: 2026-05-25

Status: running

Purpose: start the ordered strong-detector sweep for the current DOTA v1.5 path.
The sweep order is Oriented R-CNN -> RoI Transformer -> ReDet, with the first
wave launched in parallel according to the available GPU count.

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

Notes:
- All three baselines are now launched in detached screen sessions.
- The DOTA v1.5 wrappers use `PackDetInputs` pipelines and a 640x640 resize
  to keep memory within the available GPU budget.
- ReDet is initialized from scratch by clearing the backbone `init_cfg` in the
  wrapper instead of loading the missing ReResNet checkpoint, and the scratch
  run uses a lower AMP learning rate to avoid NaN divergence.
