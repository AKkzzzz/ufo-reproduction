#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/inspire/hdd/global_user/guoluosong-253108120129/yx-ufo"
GSAM_PYTHON="/inspire/hdd3/project/intelligent-driving-agent/public/workspace/yx/.venv-groundedsam2/bin/python"
RUNNER="$REPO_ROOT/tools/r9/preprocess_waymo_sam2_tracks_persistent.py"
LOG="$REPO_ROOT/outputs/r9_sam_preprocess_4090.log"

mkdir -p "$REPO_ROOT/outputs"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

exec "$GSAM_PYTHON" "$RUNNER" \
  --waymo-root /inspire/hdd/global_user/guoluosong-253108120129/aaai/long_dggt/data/waymo/processed \
  --output-root /inspire/hdd/global_user/guoluosong-253108120129/yx-ufo/data/r9_sam_tracks \
  "$@" 2>&1 | tee -a "$LOG"
