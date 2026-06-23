# 2026-06-23 DOTA2 S3 Scene-Adapter Long-12 Override Complete

## Scope

Archive the completed DOTA2 S3 long-12 override launched on 2026-06-20. This
record closes the missing completion note for
`20260620_dota2_s3_scene_adapter_long12_override_launch.md`.

This remains archive-only / negative-to-neutral DOTA2 S3 evidence. It does not
reopen S4, pseudo-labeling, FAIR1M, or a DOTA2 scene-adapter claim.

## Workdirs

| Replica | GPU | Workdir | Launch log |
| --- | ---: | --- | --- |
| rep3407 | 1 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep3407_long12_20260620` | `launch_20260620_gpu1.log` |
| rep4407 | 2 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep4407_long12_20260620` | `launch_20260620_gpu2.log` |
| rep5407 | 3 | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s3_scene_adapter_loss0_best_rep5407_long12_20260620` | `launch_20260620_gpu3.log` |

All three workdirs contain checkpoints through `epoch_12.pth`.

## Metrics

Metrics are `dota/mAP/dota/AP50` from the run `vis_data` files.

| Replica | E1 | E2 | E3 | E4 | E5 | E6 | E7 | E8 | E9 | E10 | E11 | E12 final | Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rep3407 | `0.6158/0.6160` | `0.6216/0.6220` | `0.6142/0.6140` | `0.6134/0.6130` | `0.6142/0.6140` | `0.6169/0.6170` | `0.6159/0.6160` | `0.6146/0.6150` | `0.6096/0.6100` | `0.6119/0.6120` | `0.6122/0.6120` | `0.6122/0.6120` | `0.6216` |
| rep4407 | `0.6177/0.6180` | `0.6139/0.6140` | `0.6154/0.6150` | `0.6154/0.6150` | `0.6154/0.6150` | `0.6170/0.6170` | `0.6149/0.6150` | `0.6116/0.6120` | `0.6151/0.6150` | `0.6150/0.6150` | `0.6152/0.6150` | `0.6150/0.6150` | `0.6177` |
| rep5407 | `0.6215/0.6210` | `0.6211/0.6210` | `0.6113/0.6110` | `0.6197/0.6200` | `0.6206/0.6210` | `0.6211/0.6210` | `0.6168/0.6170` | `0.6185/0.6180` | `0.6150/0.6150` | `0.6151/0.6150` | `0.6117/0.6120` | `0.6118/0.6120` | `0.6215` |

Aggregate:

- Best mean mAP: `0.6203`.
- Final epoch-12 mean mAP: `0.6130`.
- Best single checkpoint: rep3407 epoch 2, `0.6216`.

## Failure Scan

Scoped scan across the three long-12 workdirs found no matches for:
`Traceback`, CUDA OOM, `out-of-memory`, `out of memory`, `libpng`, `CRC`,
`NoneType`, `ValueError`, `KeyboardInterrupt`, `loss: nan`, `loss: inf`,
`grad_norm: nan`, or `grad_norm: inf`.

## Interpretation

The long-12 extension did not recover DOTA2 S3 into useful evidence. Best mean
`0.6203` only matches the prior DOTA2 S2 early-checkpoint story, while final
mean `0.6130` is below DOTA2 S1/S2 useful checkpoints. Keep this as clean
negative-to-neutral archive evidence.
