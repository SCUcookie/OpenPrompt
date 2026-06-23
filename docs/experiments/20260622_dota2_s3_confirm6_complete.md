# 2026-06-22 DOTA2 S3 Confirm6 Completion

## Scope

Archive the completed DOTA2 S3 confirmation pack launched on 2026-06-21 from
DOTA2 S2 loss-0 replicas `6407/7407/8407`. This is confirmation/archive work
only. S4, pseudo-labeling, FAIR1M, and new route claims remain paused.

Launch record:
`docs/experiments/20260621_dota2_s3_confirm6_rep6407_7407_8407_launch.md`.

## Host State

Completion/archive checks on `2026-06-22`:

- All three confirm6 screens had exited.
- `screen -ls` showed only `s0_result_log_monitor_20260603`.
- GPU 4 was occupied by another process.
- GPUs 1, 2, and 3 were idle before continuation planning.

## Workdirs

| Replica | GPU | Workdir | Launch log |
| --- | ---: | --- | --- |
| rep6407 | 1 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep6407_confirm6_20260621` | `launch_20260621_gpu1.log` |
| rep7407 | 2 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep7407_confirm6_20260621` | `launch_20260621_gpu2.log` |
| rep8407 | 3 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep8407_confirm6_20260621` | `launch_20260621_gpu3.log` |

All three workdirs contain `epoch_1.pth` through `epoch_6.pth`.

## Metrics

| Replica | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 | Epoch 6 final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rep6407 | `0.6183/0.6180` | `0.6144/0.6140` | `0.6138/0.6140` | `0.6153/0.6150` | `0.6183/0.6180` | `0.6162/0.6160` |
| rep7407 | `0.6182/0.6180` | `0.6168/0.6170` | `0.6111/0.6110` | `0.6143/0.6140` | `0.6158/0.6160` | `0.6151/0.6150` |
| rep8407 | `0.6162/0.6160` | `0.6172/0.6170` | `0.6214/0.6210` | `0.6134/0.6130` | `0.6135/0.6130` | `0.6165/0.6170` |

Best rounded mAP:

- rep6407: `0.6183`
- rep7407: `0.6182`
- rep8407: `0.6214`
- Best mean rounded mAP: `0.6193`
- Final mean rounded mAP: `0.6159`

## Failure Scan

Scoped scan across the three launch logs found no matches for:
`Traceback`, CUDA OOM, `out-of-memory`, `out of memory`, `libpng`, `CRC`,
`NoneType`, `ValueError`, `KeyboardInterrupt`, `loss: nan`, `loss: inf`,
`grad_norm: nan`, or `grad_norm: inf`.

## Interpretation

This confirmation pack is clean but remains negative-to-neutral DOTA2 S3
evidence at final checkpoints. The useful signal, if any, is still
best-checkpoint/early-checkpoint behavior rather than final epoch behavior.
