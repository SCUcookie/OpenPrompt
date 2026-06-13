# DIOR-R S0 Sanitized Long Interim - 2026-06-13

This record archives the current DIOR-R sanitized S0 long-run evidence as of
`2026-06-13 09:53 CST`. It uses the sanitized DIOR-R root:

`/data5/2025/ldh/OpenRSD/data/DIOR_R_dota_sanitized_invalidsize_20260612/`

The runs are finite diagnostic S0 evidence while final metrics are still
pending for at least ORCNN, RoI Transformer, and RetinaNet.

## Active Runs

| GPU | Model | Screen | Schedule | Current state |
| --- | --- | --- | --- | --- |
| 0 | ORCNN R50 | `dior_r_s0_orcnn_extend_after_original_20260612` | `max_epochs=36`, `val_interval=4` | active in epoch 34/36 |
| 1 | RoI Transformer R50 | `dior_r_s0_roi_trans_extend_after_original_20260612` | `max_epochs=52`, `val_interval=4` | active in epoch 49/52 after epoch-48 validation |
| 2 | Rotated RetinaNet R50 | `dior_r_s0_retinanet_extend_after_original_20260612` | `max_epochs=96`, `val_interval=4` | active in epoch 87/96 after epoch-84 validation |

`screen -ls` at this pass showed all three continuation screens detached and
active. `nvidia-smi` showed GPUs 0, 1, and 2 occupied by these runs, with GPUs
3-6 idle except low Xorg memory.

## Artifacts

| Model | Work dir | Runtime config | Launch logs | Runtime logs |
| --- | --- | --- | --- | --- |
| ORCNN R50 | `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_orcnn_sanitized_long_20260612_gpu0` | `dior_r_s0_orcnn_sanitized_long_20260612_gpu0.py` | `launch_20260612_gpu0.log`, `launch_resume_to_36e_20260612_gpu0.log`, `continuation_watcher_20260612.log` | `20260612_181155/20260612_181155.log`, `20260612_235635/20260612_235635.log` |
| RoI Transformer R50 | `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_roi_trans_sanitized_long_20260612_gpu1` | `dior_r_s0_roi_trans_sanitized_long_20260612_gpu1.py` | `launch_20260612_gpu1.log`, `launch_resume_to_52e_20260612_gpu1.log`, `continuation_watcher_20260612.log` | `20260612_181202/20260612_181202.log`, `20260612_232047/20260612_232047.log` |
| Rotated RetinaNet R50 | `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_retinanet_sanitized_long_20260612_gpu2` | `dior_r_s0_retinanet_sanitized_long_20260612_gpu2.py` | `launch_20260612_gpu2.log`, `launch_resume_to_96e_20260612_gpu2.log`, `continuation_watcher_20260612.log` | `20260612_181213/20260612_181213.log`, `20260612_235311/20260612_235311.log` |

All three configs use `num_classes=20`, `data_root='data/DIOR_R_dota_sanitized_invalidsize_20260612/'`,
`train_val/labelTxt/` and `test/labelTxt/` under that sanitized root, and
`CheckpointHook(interval=4)`. The extension phase uses `resume=True`.

## Interim Metrics

| Model | Best checkpoint so far | Best metric so far | Latest validation |
| --- | --- | --- | --- |
| ORCNN R50 | `epoch_28.pth` | epoch 28: `dota/mAP=0.6341`, `dota/AP50=0.6340` | epoch 32: `dota/mAP=0.6331`, `dota/AP50=0.6330`; epoch 36 final pending |
| RoI Transformer R50 | `epoch_48.pth` | epoch 48: `dota/mAP=0.6531`, `dota/AP50=0.6530` | epoch 48 is latest; epoch 52 final pending |
| Rotated RetinaNet R50 | `epoch_80.pth` | epoch 80: `dota/mAP=0.5686`, `dota/AP50=0.5690` | epoch 84: `dota/mAP=0.5667`, `dota/AP50=0.5670`; final pending |

RoI Transformer is the current DIOR-R S0 leader. ORCNN is secondary and remains
competitive, while RetinaNet trails both two-stage detectors.

## Failure Scan

Scoped scan target:

`/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_orcnn_sanitized_long_20260612_gpu0`
`/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_roi_trans_sanitized_long_20260612_gpu1`
`/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_retinanet_sanitized_long_20260612_gpu2`

The scan pattern covered `Traceback`, CUDA OOM, `out of memory`, `libpng`,
`CRC`, `NoneType`, `ValueError`, `KeyboardInterrupt`, `loss: nan/inf`, and
`grad_norm: nan/inf`. No matches were found in the checked logs.

Latest tails show finite losses and finite gradient norms:

- ORCNN: epoch 34 training with finite losses.
- RoI Transformer: epoch 49 training after epoch-48 validation, finite losses.
- RetinaNet: epoch 87 training after epoch-84 validation, finite losses.

## Next Actions

Do not launch DIOR-R GeoNexus S1 until ORCNN and RoI Transformer finish and the
best and final metrics are archived separately. If RoI Transformer remains the
best S0 detector, prepare DIOR-R GeoNexus S1 from the best RoI Transformer
sanitized checkpoint using the same minimal RemoteCLIP S1 route used for DOTA2.

Keep DOTA2 S3/S4, pseudo-labeling, FAIR1M, and new prompt modules paused today.
RetinaNet may continue in the background unless GPU pressure requires stopping
after a validation checkpoint.
