#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WAYMO_ROOT="/inspire/hdd/global_user/guoluosong-253108120129/aaai/long_dggt/data/waymo/processed"
OUTPUT_ROOT="${REPO_ROOT}/data/r9_sam_tracks"
GSAM_ROOT="${REPO_ROOT}/third_party/Grounded-SAM-2"
GSAM_PYTHON="${REPO_ROOT}/third_party/groundedsam2_env/bin/python"
HF_CACHE="${REPO_ROOT}/third_party/hf_cache"
CHECKPOINT="${GSAM_ROOT}/checkpoints/sam2.1_hiera_large.pt"
RUNNER="${REPO_ROOT}/tools/r9/preprocess_waymo_sam2_tracks_persistent.py"
LOG_ROOT="${REPO_ROOT}/outputs/r9_sam_h200"

for path in \
  "${WAYMO_ROOT}/training" \
  "${GSAM_ROOT}" \
  "${HF_CACHE}"; do
  test -d "${path}" || { echo "Missing directory: ${path}" >&2; exit 1; }
done
for path in "${GSAM_PYTHON}" "${CHECKPOINT}" "${RUNNER}"; do
  test -e "${path}" || { echo "Missing file: ${path}" >&2; exit 1; }
done
grounding_weights="$(find "${HF_CACHE}/hub/models--IDEA-Research--grounding-dino-tiny" -name model.safetensors -print -quit)"
test -n "${grounding_weights}" -a -s "${grounding_weights}" || {
  echo "GroundingDINO offline weights missing from ${HF_CACHE}" >&2
  exit 1
}

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}"
export HF_HOME="${HF_CACHE}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

if [[ "${R9_SAM_DRY_RUN:-0}" == "1" ]]; then
  for shard in {0..7}; do
    echo "CUDA_VISIBLE_DEVICES=${shard} ${GSAM_PYTHON} ${RUNNER} --waymo-root ${WAYMO_ROOT} --output-root ${OUTPUT_ROOT} --gsam-root ${GSAM_ROOT} --checkpoint ${CHECKPOINT} --num-shards 8 --shard-index ${shard}"
  done
  exit 0
fi

gpu_count="$(nvidia-smi -L | wc -l)"
if (( gpu_count < 8 )); then
  echo "Need at least 8 visible GPUs, found ${gpu_count}" >&2
  exit 1
fi

pids=()
for shard in {0..7}; do
  log="${LOG_ROOT}/shard_${shard}.log"
  pid_file="${LOG_ROOT}/shard_${shard}.pid"
  : > "${log}"
  (
    export CUDA_VISIBLE_DEVICES="${shard}"
    "${GSAM_PYTHON}" "${RUNNER}" \
      --waymo-root "${WAYMO_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --gsam-root "${GSAM_ROOT}" \
      --checkpoint "${CHECKPOINT}" \
      --num-shards 8 \
      --shard-index "${shard}" \
      2>&1 | tee -a "${log}"
  ) &
  pid=$!
  pids+=("${pid}")
  printf '%s\n' "${pid}" > "${pid_file}"
  echo "Started shard=${shard} gpu=${shard} pid=${pid} log=${log}"
done

status=0
for shard in {0..7}; do
  if ! wait "${pids[${shard}]}"; then
    echo "Shard ${shard} failed (pid=${pids[${shard}]})" >&2
    status=1
  fi
done
exit "${status}"
