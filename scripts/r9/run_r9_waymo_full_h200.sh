#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DGGT_PYTHON="/root/miniconda3/envs/dggt_data/bin/python"
DGGT_TORCHRUN="/root/miniconda3/envs/dggt_data/bin/torchrun"
test -x "${DGGT_PYTHON}" || { echo "Missing dggt_data Python: ${DGGT_PYTHON}" >&2; exit 1; }
test -x "${DGGT_TORCHRUN}" || { echo "Missing dggt_data torchrun: ${DGGT_TORCHRUN}" >&2; exit 1; }
gpu_count="$(nvidia-smi -L | wc -l)"
if (( gpu_count < 8 )); then
  echo "Need at least 8 visible GPUs, found ${gpu_count}" >&2
  exit 1
fi
export UFO_PYTHON_BIN="${DGGT_PYTHON}"
export UFO_TORCHRUN_BIN="${DGGT_TORCHRUN}"
source "${REPO_ROOT}/scripts/h200/env_h200_offline.sh"
cd "${REPO_ROOT}"

PROFILE="${REPO_ROOT}/outputs/h200_r9_profile/recommended.env"
CONFIG="${REPO_ROOT}/configs/h200/ufo_r9_waymo_full_global64.json"
SAM_ROOT="${REPO_ROOT}/data/r9_sam_tracks"
RUN_ROOT="${REPO_ROOT}/outputs/r9_waymo_full_h200"

test -f "${PROFILE}" || { echo "Missing benchmark recommendation: ${PROFILE}" >&2; exit 1; }
test -f "${CONFIG}" || { echo "Missing full config: ${CONFIG}" >&2; exit 1; }
test -d "${SAM_ROOT}" || { echo "Missing SAM tracks: ${SAM_ROOT}" >&2; exit 1; }
test -f "${UFO_DYNAMIC_POOL}" || { echo "Missing dynamic pool: ${UFO_DYNAMIC_POOL}" >&2; exit 1; }
source "${PROFILE}"
: "${H200_BATCH_SIZE:?recommended.env lacks H200_BATCH_SIZE}"
: "${H200_ACCUMULATION_STEPS:?recommended.env lacks H200_ACCUMULATION_STEPS}"

global_batch=$((8 * H200_BATCH_SIZE * H200_ACCUMULATION_STEPS))
if (( global_batch != 64 )); then
  echo "Invalid effective global batch: ${global_batch}" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "Starting R9 Full Waymo scratch training on 8xH200"
echo "GPUs=8 batch_per_gpu=${H200_BATCH_SIZE} accumulation=${H200_ACCUMULATION_STEPS} global_batch=${global_batch}"
echo "iterations=100000 SAM=${SAM_ROOT} output=${RUN_ROOT}"

"${UFO_TORCHRUN_BIN}" \
  --standalone \
  --nproc_per_node=8 \
  main.py \
  --config "${CONFIG}" \
  --project r9_waymo_full_h200 \
  --exp_name R9_WaymoFull_Scratch100k_Global64 \
  --output_dir "${REPO_ROOT}/outputs" \
  --data_root "${UFO_DATA_ROOT}" \
  --sam_track_root "${SAM_ROOT}" \
  --dynamic_rich_pool "${UFO_DYNAMIC_POOL}" \
  --batch_size "${H200_BATCH_SIZE}" \
  --gradient_accumulation_steps "${H200_ACCUMULATION_STEPS}" \
  --ddp_accumulation_no_sync \
  2>&1 | tee -a "${RUN_ROOT}/train.log"
