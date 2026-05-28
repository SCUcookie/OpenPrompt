#!/bin/bash
# GeoNexus-RSD Parallel Launch Script
# Launches all 7 GPU routes for the comprehensive improvement plan.
# Usage: bash scripts/launch_all_routes.sh

set -euo pipefail

MMROTATE_ENV=/data1/anaconda3/envs/zwl_mmrotate/bin/python
OPENRSD_ROOT=/data5/2025/ldh/OpenRSD
OPENPROMPT_ROOT=/data5/2025/ldh/OpenPrompt
MMROTATE_TOOLS=$OPENRSD_ROOT/tools

echo "=== GeoNexus-RSD: Launching all 7 GPU routes ==="
echo "Started at: $(date)"
echo ""

# --- Route A.1: Oriented R-CNN 1024x1024, 36ep, augmentations [GPU 0] ---
echo "[GPU 0] Route A.1: Oriented R-CNN upgraded baseline"
CUDA_VISIBLE_DEVICES=0 nohup $MMROTATE_ENV \
  $OPENRSD_ROOT/tools/bootstrap_run.py $MMROTATE_TOOLS/train.py \
  $OPENRSD_ROOT/mmrotate_configs/strong_baseline_dota15/oriented-rcnn-le90_r50_fpn_amp-1x_dota15.py \
  --work-dir $OPENRSD_ROOT/work_dirs/route_a1_oriented_rcnn \
  > $OPENRSD_ROOT/work_dirs/route_a1_oriented_rcnn/train.log 2>&1 &
echo "  PID: $!"

# --- Route A.2: RoI Transformer [GPU 1] ---
echo "[GPU 1] Route A.2: RoI Transformer stable rerun"
CUDA_VISIBLE_DEVICES=1 nohup $MMROTATE_ENV \
  $OPENRSD_ROOT/tools/bootstrap_run.py $MMROTATE_TOOLS/train.py \
  $OPENRSD_ROOT/mmrotate_configs/strong_baseline_dota15/roi-trans-le90_r50_fpn_amp-1x_dota15.py \
  --work-dir $OPENRSD_ROOT/work_dirs/route_a2_roi_transformer \
  > $OPENRSD_ROOT/work_dirs/route_a2_roi_transformer/train.log 2>&1 &
echo "  PID: $!"

# --- Route B.2: ReDet rerun (with pretrained weights) [GPU 3] ---
echo "[GPU 3] Route B.2: ReDet rerun with pretrained ReResNet-50"
# First, try to use the pretrained ReResNet backbone
RE_RESNET_CKPT=/data5/2025/temp/Supplements/re_resnet50_c8_batch256-25b16846.pth
if [ -f "$RE_RESNET_CKPT" ]; then
    echo "  Using ReResNet-50 pretrained: $RE_RESNET_CKPT"
    # The ReDet config clears init_cfg; override to load pretrained
    CUDA_VISIBLE_DEVICES=3 nohup $MMROTATE_ENV \
      $OPENRSD_ROOT/tools/bootstrap_run.py $MMROTATE_TOOLS/train.py \
      $OPENRSD_ROOT/mmrotate_configs/strong_baseline_dota15/redet-le90_re50_refpn_amp-1x_dota15.py \
      --work-dir $OPENRSD_ROOT/work_dirs/route_b2_redet \
      --cfg-options model.backbone.init_cfg.checkpoint=$RE_RESNET_CKPT \
      > $OPENRSD_ROOT/work_dirs/route_b2_redet/train.log 2>&1 &
else
    CUDA_VISIBLE_DEVICES=3 nohup $MMROTATE_ENV \
      $OPENRSD_ROOT/tools/bootstrap_run.py $MMROTATE_TOOLS/train.py \
      $OPENRSD_ROOT/mmrotate_configs/strong_baseline_dota15/redet-le90_re50_refpn_amp-1x_dota15.py \
      --work-dir $OPENRSD_ROOT/work_dirs/route_b2_redet \
      > $OPENRSD_ROOT/work_dirs/route_b2_redet/train.log 2>&1 &
fi
echo "  PID: $!"

# --- Route C.1: Oriented R-CNN + DFLA + DHEAD [GPU 4] ---
echo "[GPU 4] Route C.1: Oriented R-CNN with DFLA + DHEAD"
CUDA_VISIBLE_DEVICES=4 nohup $MMROTATE_ENV \
  $OPENRSD_ROOT/tools/bootstrap_run.py $MMROTATE_TOOLS/train.py \
  $OPENRSD_ROOT/mmrotate_configs/strong_baseline_dota15/oriented-rcnn-le90_r50_fpn_dfla_dhead-1x_dota15.py \
  --work-dir $OPENRSD_ROOT/work_dirs/route_c1_dflan \
  > $OPENRSD_ROOT/work_dirs/route_c1_dflan/train.log 2>&1 &
echo "  PID: $!"

# --- Route C.2: Oriented R-CNN + full aug (already in config) + tail-class focus [GPU 5] ---
echo "[GPU 5] Route C.2: Full augmentation suite (already in route A.1 config)"
echo "  Note: augmentation pipeline integrated in route A.1 config."
echo "  GPU 5 available for hyperparameter sweep or second seed."

# --- Route D: Pseudo-label generation (run after A.1 finishes) [GPU 6] ---
echo "[GPU 6] Route D: Reserved for pseudo-label generation"
echo "  Will run after Route A.1 produces best checkpoint."
echo "  Command: PYTHONPATH=$OPENPROMPT_ROOT/src python $OPENPROMPT_ROOT/scripts/self_train.py --config ..."

# --- GPU 2: Scaffold training with RemoteCLIP [GPU 2] ---
echo "[GPU 2] Route B.1: Scaffold training with RemoteCLIP prompt bank"
echo "  Command (run manually after confirming prompt bank):"
echo "  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$OPENPROMPT_ROOT/src python $OPENPROMPT_ROOT/scripts/train.py --config $OPENPROMPT_ROOT/configs/experiments/dota_v15_geonexus.yaml"

echo ""
echo "=== All routes launched ==="
echo "Check progress:"
echo "  tail -f $OPENRSD_ROOT/work_dirs/route_a1_oriented_rcnn/train.log"
echo "  tail -f $OPENRSD_ROOT/work_dirs/route_a2_roi_transformer/train.log"
echo "  tail -f $OPENRSD_ROOT/work_dirs/route_b2_redet/train.log"
echo "  tail -f $OPENRSD_ROOT/work_dirs/route_c1_dflan/train.log"
echo ""
echo "Status check: ps aux | grep 'train.py' | grep -v grep"
