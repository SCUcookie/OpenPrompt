#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH=src
exec python3 -u scripts/train.py --config configs/experiments/dota_v2_baseline.yaml "$@"
