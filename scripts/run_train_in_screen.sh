#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${1:-openprompt_dota_v1_baseline}"
CONFIG_PATH="${2:-configs/experiments/dota_v1_baseline_repro.yaml}"
LOG_DIR="${3:-${REPO_ROOT}/outputs/${SESSION_NAME}}"
LOG_FILE="${LOG_DIR}/train.log"

mkdir -p "${LOG_DIR}"

screen -dmS "${SESSION_NAME}" bash -lc "source /data1/anaconda3/etc/profile.d/conda.sh && conda activate zwl_oneformer_ViT_P && cd '${REPO_ROOT}' && PYTHONPATH=src python scripts/train.py --config '${CONFIG_PATH}' |& tee -a '${LOG_FILE}'"

echo "Started screen session: ${SESSION_NAME}"
echo "Log file: ${LOG_FILE}"
echo "Attach with: screen -r ${SESSION_NAME}"
