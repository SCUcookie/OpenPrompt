# 2026-07-03 Strip R-CNN-S DIOR-R Protocol Eval

## Intent

Comparator/protocol evaluation for DIOR-R using the released Strip R-CNN-S
checkpoint path advertised by the official Strip R-CNN README, not new
GeoNexus training. This is protocol-grounding evidence only, because labels are
bridged from sanitized GeoNexus DOTA-style text into Strip R-CNN DIOR XML.

## Paths

- Source labels: `/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/{train_val,test}/labelTxt_sanitized_invalidsize_20260612`
- Source images: `/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/{train_val,test}/images`
- Bridge root: `/data5/2025/ldh/strip_rcnn_protocol_eval_20260703/DIOR_R_geonexus_xml_20260703`
- Config: `/data5/2025/ldh/strip_rcnn_protocol_eval_20260703/strip_rcnn_s_dior_r_geonexus_test_20260703.py`
- Checkpoint directory: `/data5/2025/ldh/strip_rcnn_protocol_eval_20260703/checkpoints/strip_rcnn_s_dior_r/`
- Eval work dir: `/data5/2025/ldh/strip_rcnn_protocol_eval_20260703/strip_rcnn_s_dior_r_geonexus_eval`

## Bridge Validation

Completed with:

- `train_ids=11725`, `train_objects=68070`
- `val_ids=11738`, `test_ids=11738`, `test_objects=124443`
- `unknown_classes=0`
- `invalid_polygons=0`

Builder:
`/data5/2025/ldh/strip_rcnn_protocol_eval_20260703/build_dior_r_geonexus_xml_bridge.py`

Output layout:

- `ImageSets/{train,val,test}.txt`
- `JPEGImages/trainval/*.jpg` symlinked to source PNG images
- `JPEGImages/test/*.jpg` symlinked to source PNG images
- `Annotations/Oriented Bounding Boxes/*.xml`

## Config Preflight

Passed with:

```bash
cd /data5/2025/ldh/Strip-R-CNN
/data1/anaconda3/envs/lcs_mmrotate0.3/bin/python \
  tools/misc/print_config.py \
  /data5/2025/ldh/strip_rcnn_protocol_eval_20260703/strip_rcnn_s_dior_r_geonexus_test_20260703.py
```

Runtime environment note: `/data1/anaconda3/envs/zwl_mmrotate/bin/python` has
`mmcv==2.0.0rc4` and does not expose `mmcv.Config`, so old-style Strip R-CNN
tools fail there before config parsing. `/data1/anaconda3/envs/lcs_mmrotate0.3`
has `mmcv==1.7.2` and passed `print_config.py`.

## Checkpoint Source

- README DIOR-R model URL ID: `1_c2aXANKHl0cIBb370LNIkCyDmQpA3_o`
- Download command:

```bash
/data1/anaconda3/envs/zwl_mmrotate/bin/python -m gdown \
  --id 1_c2aXANKHl0cIBb370LNIkCyDmQpA3_o \
  -O /data5/2025/ldh/strip_rcnn_protocol_eval_20260703/checkpoints/strip_rcnn_s_dior_r/
```

Important risk: the same README ID is also listed as the ImageNet 300-epoch
pre-trained Strip R-CNN-S backbone. Before launching, inspect checkpoint keys
and stop if detector head keys are missing.

Download attempts on `2026-07-03` did not produce a checkpoint file:

1. `gdown --id 1_c2aXANKHl0cIBb370LNIkCyDmQpA3_o -O <checkpoint_dir>/`
   failed with `HTTPSConnectionPool(host='drive.google.com', port=443)` and
   `Failed to establish a new connection: [Errno 110] Connection timed out`.
2. A retry with
   `gdown 'https://drive.google.com/uc?id=1_c2aXANKHl0cIBb370LNIkCyDmQpA3_o'`
   created no partial file after roughly one minute and was interrupted.

Checkpoint directory remained empty:
`/data5/2025/ldh/strip_rcnn_protocol_eval_20260703/checkpoints/strip_rcnn_s_dior_r/`.

Because no checkpoint was available, detector-key inspection was not possible.

## Uploaded Checkpoint Inspection

The user uploaded:
`/data5/2025/ldh/strip_rcnn_protocol_eval_20260703/checkpoints/strip_rcnn_s_dior_r/stripnet_s.pth`
(`166370373` bytes).

Inspection command:

```bash
/data1/anaconda3/envs/lcs_mmrotate0.3/bin/python -c "import torch; ..."
```

Checkpoint structure:

- Top-level keys: `epoch`, `arch`, `state_dict`, `optimizer`, `version`,
  `args`, `amp_scaler`, `metric`
- `state_dict` keys: `338`
- Sample keys begin with raw backbone names such as `patch_embed1.proj.weight`
  and `block1.0.layer_scale_1`
- Detector keys present:
  - `backbone.*`: no
  - `neck.*`: no
  - `rpn_head.*`: no
  - `roi_head.*`: no
  - `bbox_head`: no

Conclusion: this is the StripNet-S ImageNet/backbone pretrain, not a full
Strip R-CNN-S DIOR-R detector checkpoint. Evaluation was not launched.

Two additional uploaded detector checkpoints were found outside the checkpoint
subdirectory:

- `/data5/2025/ldh/strip_rcnn_protocol_eval_20260703/strip_rcnn_s_dota.pth`
- `/data5/2025/ldh/strip_rcnn_protocol_eval_20260703/strip_rcnn_s_fair1m.pth`

Both are full detector checkpoints with `backbone`, `neck`, `rpn_head`,
`roi_head`, and `bbox_head` keys, but neither is a DIOR-R detector checkpoint.
Their metadata and classifier shapes are incompatible with the DIOR-R
20-class config:

- `strip_rcnn_s_dota.pth`: DOTA classes, `roi_head.bbox_head.fc_cls.weight`
  shape `(16, 1024)` for 15 classes plus background.
- `strip_rcnn_s_fair1m.pth`: FAIR1M classes,
  `roi_head.bbox_head.fc_cls.weight` shape `(38, 1024)` for 37 classes plus
  background.
- DIOR-R config expects 20 classes plus background, so the matching classifier
  shape would be `(21, 1024)`.

Conclusion: these two files are useful evidence that the model zoo detector
format is understood, but they are not valid for the DIOR-R protocol eval.

## Archived Blocker Status

Rechecked on `2026-07-06` and archived as blocked by missing released DIOR-R
detector checkpoint. The official/user-provided checkpoint set still contains
no usable full Strip R-CNN-S DIOR-R detector:

- `stripnet_s.pth`: ImageNet/backbone pretrain only, with raw
  `patch_embed*`/`block*` keys and no `backbone.*`, `neck.*`, `rpn_head.*`,
  `roi_head.*`, or `bbox_head` detector keys.
- `strip_rcnn_s_dota.pth`: full detector checkpoint, but DOTA metadata has
  15 classes and `roi_head.bbox_head.fc_cls.weight` shape `(16, 1024)`.
- `strip_rcnn_s_fair1m.pth`: full detector checkpoint, but FAIR1M metadata
  has 37 classes and `roi_head.bbox_head.fc_cls.weight` shape `(38, 1024)`.
- DIOR-R requires 20 classes plus background, so the detector classifier head
  must have shape `(21, 1024)`.

Runtime state on archive: no `strip_rcnn_s_dior_r_eval_20260703` screen was
running; only `s0_result_log_monitor_20260603` was present. GPUs `0`, `2`,
`3`, `4`, and `5` were effectively idle, while unrelated jobs occupied GPUs
`1` and `6`. No new GPU job was launched.

Next action: keep this comparator archived as a released-checkpoint blocker and
move to paper/result analysis or another explicitly approved comparator. Do not
train a replacement Strip R-CNN-S DIOR-R detector and do not evaluate
mismatched DOTA, FAIR1M, or backbone-only weights.

## Launch Status

Blocked before launch and archived. Treat the command below as an inactive
template; launch only after:

- Bridge validation matches expected counts.
- Config preflight succeeds.
- Checkpoint inspection confirms a full detector checkpoint, not only a
  backbone pretrain.
- GPU selection avoids occupied devices, preferring GPU `0` if idle.

Current blocker: the official/user-provided checkpoint set does not include a
usable full DIOR-R detector. `stripnet_s.pth` is only a backbone pretrain, and
the DOTA/FAIR1M detector checkpoints have incompatible classifier heads. Do not
launch fallback training or evaluate mismatched checkpoints; provide a valid
full DIOR-R detector checkpoint with `neck`, `rpn_head`, `roi_head`, and a
DIOR-R `fc_cls` head of shape `(21, 1024)`.

Planned command shape:

```bash
screen -dmS strip_rcnn_s_dior_r_eval_20260703 bash -lc 'cd /data5/2025/ldh/Strip-R-CNN && CUDA_VISIBLE_DEVICES=0 /data1/anaconda3/envs/lcs_mmrotate0.3/bin/python tools/test.py /data5/2025/ldh/strip_rcnn_protocol_eval_20260703/strip_rcnn_s_dior_r_geonexus_test_20260703.py <checkpoint> --work-dir /data5/2025/ldh/strip_rcnn_protocol_eval_20260703/strip_rcnn_s_dior_r_geonexus_eval --out /data5/2025/ldh/strip_rcnn_protocol_eval_20260703/strip_rcnn_s_dior_r_geonexus_eval/preds.pkl --eval mAP --eval-options iou_thr=0.5 > /data5/2025/ldh/strip_rcnn_protocol_eval_20260703/strip_rcnn_s_dior_r_geonexus_eval/test_stdout.log 2>&1'
```
