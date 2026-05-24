#!/usr/bin/env bash
set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH=src

CONFIG_PATH="configs/experiments/dota_v2_baseline.yaml"
RESUME_ARG="${RESUME_ARG:-latest}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1 3 4}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-10}"

attempt=0
while true; do
  for gpu in $GPU_CANDIDATES; do
    attempt=$((attempt + 1))
    printf '[%s] attempt %d on host GPU %s\n' "$(date '+%F %T %Z')" "$attempt" "$gpu"
    export CUDA_VISIBLE_DEVICES="$gpu"
    python -u scripts/train.py --config "$CONFIG_PATH" --resume "$RESUME_ARG"
    status=$?
    if [ "$status" -eq 0 ]; then
      printf '[%s] training completed successfully on host GPU %s\n' "$(date '+%F %T %Z')" "$gpu"
      exit 0
    fi
    printf '[%s] training exited with status %d on host GPU %s; retrying in %s seconds\n' \
      "$(date '+%F %T %Z')" "$status" "$gpu" "$RETRY_DELAY_SECONDS"
    sleep "$RETRY_DELAY_SECONDS"
  done
done
