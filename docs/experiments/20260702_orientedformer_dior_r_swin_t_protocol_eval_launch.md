# 2026-07-02 OrientedFormer Swin-T DIOR-R Protocol Eval

## Intent

Comparator/protocol evaluation for DIOR-R using the released OrientedFormer Swin-T checkpoint, not new GeoNexus training. This is protocol-grounding evidence only, because the labels are bridged from sanitized GeoNexus DOTA-style text into OrientedFormer DIOR XML.

## Paths

- Source labels: `/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/{train_val,test}/labelTxt_sanitized_invalidsize_20260612`
- Source images: `/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/{train_val,test}/images`
- Bridge root: `/data5/2025/ldh/orientedformer_protocol_eval_20260702/DIOR_R_geonexus_xml_20260702`
- Config: `/data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_test_20260702.py`
- Checkpoint: `/data5/2025/ldh/orientedformer_protocol_eval_20260702/checkpoints/orientedformer_hf/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dior/epoch_12.pth`
- Eval work dir: `/data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_eval`

Note: `/data5/2025/ldh/OrientedFormer` is read-only to user `zwl` (`nobody:nogroup` ownership), so untracked experiment artifacts were placed in the writable sibling directory above instead of inside the checkout.

## Bridge Validation

- `train_ids=11725`, `train_objects=68070`
- `val_ids=11738`, `test_ids=11738`, `test_objects=124443`
- `unknown_classes=0`
- `invalid_polygons=0`
- Test image symlinks: `11738`
- XML files: `23463` total (`11725` trainval + `11738` test)

## Config Preflight

`tools/misc/print_config.py` passed with:

```bash
PYTHONPATH=/data5/2025/ldh/orientedformer_protocol_eval_20260702/pyshim \
PYTHONNOUSERSITE=1 \
MPLCONFIGDIR=/tmp/matplotlib_orientedformer_swin_t_dior_r_20260702 \
/data1/anaconda3/envs/zwl_mmrotate/bin/python \
tools/misc/print_config.py \
/data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_test_20260702.py
```

The `pyshim` exposes only `projects/` from the OrientedFormer checkout, avoiding local `mmrotate` shadowing of the installed environment package.

## Launch Preflight

- Existing screen before launch: `3470174.s0_result_log_monitor_20260603`
- GPU snapshot before launch:
  - `0, 14 MiB, 0 %`
  - `1, 14 MiB, 0 %`
  - `2, 14 MiB, 0 %`
  - `3, 14 MiB, 0 %`
  - `4, 14 MiB, 0 %`
  - `5, 8435 MiB, 9 %`
  - `6, 22613 MiB, 16 %`
- Launch GPU: `0`

## Launch Command

```bash
screen -dmS orientedformer_swin_t_dior_r_eval_20260702 bash -lc 'cd /data5/2025/ldh/OrientedFormer && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data5/2025/ldh/orientedformer_protocol_eval_20260702/pyshim PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/matplotlib_orientedformer_swin_t_dior_r_20260702 /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/test.py /data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_test_20260702.py /data5/2025/ldh/orientedformer_protocol_eval_20260702/checkpoints/orientedformer_hf/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dior/epoch_12.pth --work-dir /data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_eval --out /data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_eval/preds.pkl > /data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_eval/test_stdout.log 2>&1'
```

## Runtime Adjustment

The first launch failed before inference because the installed `DIORDataset` at `/data3/2021/wxq/mmrotate-1.0.0rc1` does not accept the newer `backend_args` key. The config override was patched to use `file_client_args=dict(backend='disk')`, then `print_config.py` was rerun successfully and the job relaunched with the same model/checkpoint/data.

## Completion

- Relaunch screen: `3941891.orientedformer_swin_t_dior_r_eval_20260702`
- Start: `2026-07-02 09:56:38`
- End: `2026-07-02 10:08:55`
- Screen status after completion: exited; only pre-existing `3470174.s0_result_log_monitor_20260603` remained.
- Metrics JSON: `/data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_eval/20260702_095640/20260702_095640.json`
- Runtime log: `/data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_eval/20260702_095640/20260702_095640.log`
- Prediction dump: `/data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_eval/preds.pkl` (`175M`)

Final metrics:

```json
{"dota/mAP": 0.6882880926132202, "dota/AP50": 0.688, "data_time": 0.0018643497466066055, "time": 0.05492989873536747}
```

Class AP table from stdout:

| class | gts | dets | recall | ap |
| --- | ---: | ---: | ---: | ---: |
| airplane | 8212 | 127499 | 0.795 | 0.675 |
| airport | 666 | 100191 | 0.950 | 0.509 |
| baseballfield | 3434 | 80434 | 0.897 | 0.784 |
| basketballcourt | 2146 | 196571 | 0.952 | 0.862 |
| bridge | 2589 | 413865 | 0.752 | 0.430 |
| chimney | 1031 | 53512 | 0.871 | 0.779 |
| expressway-service-area | 1085 | 43814 | 0.956 | 0.841 |
| expressway-toll-station | 688 | 49394 | 0.884 | 0.755 |
| dam | 538 | 130653 | 0.948 | 0.408 |
| golffield | 575 | 65937 | 0.963 | 0.778 |
| groundtrackfield | 1885 | 52669 | 0.971 | 0.818 |
| harbor | 3105 | 236082 | 0.768 | 0.462 |
| overpass | 1782 | 274530 | 0.802 | 0.566 |
| ship | 35184 | 157739 | 0.933 | 0.886 |
| stadium | 672 | 39915 | 0.963 | 0.753 |
| storagetank | 23361 | 215621 | 0.807 | 0.738 |
| tenniscourt | 7343 | 83503 | 0.923 | 0.867 |
| trainstation | 509 | 58389 | 0.961 | 0.606 |
| vehicle | 26640 | 1012704 | 0.731 | 0.569 |
| windmill | 2998 | 128378 | 0.933 | 0.681 |
| mAP | | | | 0.688 |

Failure scan:

- `Traceback|Error|Exception|CUDA out|out of memory|No such file|AssertionError|RuntimeError|failed|Failed` matched only the config text `allow_failed_imports=False`.
- No actual traceback, exception, OOM, missing file, assertion, or runtime error was present in the completed run logs.
