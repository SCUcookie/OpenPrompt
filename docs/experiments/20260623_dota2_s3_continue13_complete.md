# 2026-06-23 DOTA2 S3 Continue13 Complete

## Scope

Archive the completed DOTA2 S3 continue13 pack launched on 2026-06-22 from
confirm6 replicas `6407/7407/8407`. This record closes the missing completion
note for `20260622_dota2_s3_continue13_launch.md`.

Epoch 12 is the primary comparable checkpoint. Epoch 13 is an occupancy and
stability tail. This continuation does not reopen DOTA2 S3/S4 claims,
pseudo-labeling, FAIR1M, or a route change.

## Workdirs

| Replica | GPU | Resume checkpoint | Workdir | Launch log |
| --- | ---: | --- | --- | --- |
| rep6407 | 1 | `...rep6407_confirm6_20260621/epoch_6.pth` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep6407_continue13_20260622` | `launch_20260622_gpu1.log` |
| rep7407 | 2 | `...rep7407_confirm6_20260621/epoch_6.pth` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep7407_continue13_20260622` | `launch_20260622_gpu2.log` |
| rep8407 | 3 | `...rep8407_confirm6_20260621/epoch_6.pth` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep8407_continue13_20260622` | `launch_20260622_gpu3.log` |

All three continuation workdirs contain checkpoints from `epoch_7.pth` through
`epoch_13.pth`.

## Metrics

Metrics are `dota/mAP/dota/AP50` from the continuation `vis_data` files.

| Replica | E7 | E8 | E9 | E10 | E11 | E12 primary | E13 final | Continuation best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rep6407 | `0.6173/0.6170` | `0.6184/0.6180` | `0.6171/0.6170` | `0.6145/0.6150` | `0.6149/0.6150` | `0.6149/0.6150` | `0.6117/0.6120` | `0.6184` |
| rep7407 | `0.6085/0.6090` | `0.6121/0.6120` | `0.6149/0.6150` | `0.6108/0.6110` | `0.6154/0.6150` | `0.6108/0.6110` | `0.6112/0.6110` | `0.6154` |
| rep8407 | `0.6133/0.6130` | `0.6118/0.6120` | `0.6148/0.6150` | `0.6120/0.6120` | `0.6119/0.6120` | `0.6153/0.6150` | `0.6154/0.6150` | `0.6154` |

Aggregate:

- Primary epoch-12 mean mAP: `0.6137`.
- Final epoch-13 mean mAP: `0.6128`.
- Continuation best mean mAP: `0.6164`.
- Full confirm6-plus-continuation best mean remains the confirm6-era rounded
  `0.6193`.

## Failure Scan

Scoped scan across the three continuation workdirs found no matches for:
`Traceback`, CUDA OOM, `out-of-memory`, `out of memory`, `libpng`, `CRC`,
`NoneType`, `ValueError`, `KeyboardInterrupt`, `loss: nan`, `loss: inf`,
`grad_norm: nan`, or `grad_norm: inf`.

## Interpretation

The continuation is clean but confirms that DOTA2 S3 does not improve the paper
route. The primary epoch-12 mean `0.6137` and final epoch-13 mean `0.6128` are
below the useful DOTA2 S1/S2 story. Archive only; do not use this continuation
to reopen DOTA2 S3/S4 claims.
