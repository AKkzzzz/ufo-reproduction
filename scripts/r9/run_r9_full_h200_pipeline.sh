#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

WAYMO_ROOT="/inspire/hdd/global_user/guoluosong-253108120129/aaai/long_dggt/data/waymo/processed"
SAM_ROOT="${REPO_ROOT}/data/r9_sam_tracks"
GSAM_ROOT="${REPO_ROOT}/third_party/Grounded-SAM-2"
GSAM_PYTHON="${REPO_ROOT}/third_party/groundedsam2_env/bin/python"
HF_CACHE="${REPO_ROOT}/third_party/hf_cache"
CHECKPOINT="${GSAM_ROOT}/checkpoints/sam2.1_hiera_large.pt"
DGGT_PYTHON="/root/miniconda3/envs/dggt_data/bin/python"
DGGT_TORCHRUN="/root/miniconda3/envs/dggt_data/bin/torchrun"
FULL_CONFIG="${REPO_ROOT}/configs/h200/ufo_r9_waymo_full_global64.json"
PROFILE="${REPO_ROOT}/outputs/h200_r9_profile/recommended.env"

echo "STEP 0: preflight"
mkdir -p "${SAM_ROOT}"
for path in \
  "${WAYMO_ROOT}/training" \
  "${GSAM_ROOT}" \
  "${HF_CACHE}" \
  "${SAM_ROOT}" \
  "${REPO_ROOT}/ufo"; do
  test -d "${path}" || { echo "Missing directory: ${path}" >&2; exit 1; }
done
for path in \
  "${GSAM_PYTHON}" \
  "${CHECKPOINT}" \
  "${DGGT_PYTHON}" \
  "${DGGT_TORCHRUN}" \
  "${FULL_CONFIG}" \
  "${REPO_ROOT}/main.py" \
  "${REPO_ROOT}/scripts/r9/run_waymo_sam2_tracks_h200_8gpu.sh" \
  "${REPO_ROOT}/scripts/h200/benchmark_r9_h200.sh" \
  "${REPO_ROOT}/scripts/r9/run_r9_waymo_full_h200.sh"; do
  test -e "${path}" || { echo "Missing file: ${path}" >&2; exit 1; }
done
test -x "${GSAM_PYTHON}" || { echo "GroundedSAM Python is not executable: ${GSAM_PYTHON}" >&2; exit 1; }
test -x "${DGGT_PYTHON}" || { echo "dggt_data Python is not executable: ${DGGT_PYTHON}" >&2; exit 1; }
test -x "${DGGT_TORCHRUN}" || { echo "dggt_data torchrun is not executable: ${DGGT_TORCHRUN}" >&2; exit 1; }
grounding_weights="$(find "${HF_CACHE}/hub/models--IDEA-Research--grounding-dino-tiny" -name model.safetensors -print -quit)"
test -n "${grounding_weights}" -a -s "${grounding_weights}" || {
  echo "GroundingDINO offline weights missing from ${HF_CACHE}" >&2
  exit 1
}
"${GSAM_PYTHON}" tools/r9/validate_r9_full_config.py "${FULL_CONFIG}"

if [[ "${R9_PIPELINE_DRY_RUN:-0}" == "1" ]]; then
  echo "DRY RUN: GPU count check bypassed; no model, preprocessing, benchmark, or training will run."
  echo "STEP 1 command: bash scripts/r9/run_waymo_sam2_tracks_h200_8gpu.sh"
  R9_SAM_DRY_RUN=1 bash scripts/r9/run_waymo_sam2_tracks_h200_8gpu.sh
  echo "STEP 2 command: ${GSAM_PYTHON} tools/r9/audit_waymo_sam_tracks.py --require-full"
  echo "STEP 3 command: ${DGGT_PYTHON} tools/r9/check_sam_track_contract.py --num-scenes 8"
  echo "STEP 4 command: bash scripts/h200/benchmark_r9_h200.sh"
  echo "STEP 5 profile: ${PROFILE}"
  echo "STEP 6: print selected 8xH200 global-batch-64 configuration"
  echo "STEP 7 command: bash scripts/r9/run_r9_waymo_full_h200.sh"
  echo "PIPELINE_DRY_RUN_PASS"
  exit 0
fi

gpu_count="$(nvidia-smi -L | wc -l)"
if (( gpu_count < 8 )); then
  echo "Need at least 8 visible GPUs, found ${gpu_count}" >&2
  exit 1
fi
gpu_models="$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader)"
printf '%s\n' "${gpu_models}"
if ! grep -qi 'H200' <<<"${gpu_models}"; then
  echo "WARNING: GPU names do not contain H200; continuing after printing detected models." >&2
fi

export HF_HOME="${HF_CACHE}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
"${GSAM_PYTHON}" -c 'import torch, transformers, sam2, numpy, PIL, supervision'
export UFO_PYTHON_BIN="${DGGT_PYTHON}"
export UFO_TORCHRUN_BIN="${DGGT_TORCHRUN}"
source scripts/h200/env_h200_offline.sh
"${UFO_PYTHON_BIN}" -c 'import torch, gsplat'

echo "STEP 1: 8xH200 persistent SAM preprocessing"
bash scripts/r9/run_waymo_sam2_tracks_h200_8gpu.sh

echo "STEP 2: direct full SAM audit"
"${GSAM_PYTHON}" tools/r9/audit_waymo_sam_tracks.py \
  --waymo-root "${WAYMO_ROOT}" \
  --output-root "${SAM_ROOT}" \
  --require-full

echo "STEP 3: R9 loader contract"
"${UFO_PYTHON_BIN}" tools/r9/check_sam_track_contract.py \
  --waymo-root "${WAYMO_ROOT}" \
  --sam-root "${SAM_ROOT}" \
  --num-scenes 8

echo "STEP 4: 8xH200 R9 throughput benchmark"
export R9_SAM_TRACK_ROOT="${SAM_ROOT}"
bash scripts/h200/benchmark_r9_h200.sh

echo "STEP 5: benchmark recommendation"
test -f "${PROFILE}" || { echo "Missing recommendation: ${PROFILE}" >&2; exit 1; }
source "${PROFILE}"
: "${H200_BATCH_SIZE:?recommended.env lacks H200_BATCH_SIZE}"
: "${H200_ACCUMULATION_STEPS:?recommended.env lacks H200_ACCUMULATION_STEPS}"
global_batch=$((8 * H200_BATCH_SIZE * H200_ACCUMULATION_STEPS))
test "${global_batch}" -eq 64

echo "STEP 6: final training configuration"
echo "GPU count: 8"
echo "GPU model(s):"
printf '%s\n' "${gpu_models}"
echo "batch/GPU: ${H200_BATCH_SIZE}"
echo "accumulation: ${H200_ACCUMULATION_STEPS}"
echo "global batch: ${global_batch}"
echo "iterations: 100000"
echo "SAM root: ${SAM_ROOT}"
echo "output dir: ${REPO_ROOT}/outputs/r9_waymo_full_h200"

echo "STEP 7: start foreground R9 Full Waymo 100k training"
exec bash scripts/r9/run_r9_waymo_full_h200.sh
