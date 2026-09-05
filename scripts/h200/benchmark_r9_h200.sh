#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env_h200_offline.sh"
cd "${UFO_ROOT}"

CONFIG="configs/h200/ufo_r9_h200_benchmark.json"
ROOT="${UFO_OUTPUT_ROOT}/h200_r9_profile"
SAM_ROOT="${R9_SAM_TRACK_ROOT:-${UFO_ROOT}/data/r9_sam_tracks}"

mkdir -p "${ROOT}"
rm -f \
    "${ROOT}/results.tsv" \
    "${ROOT}/results.csv" \
    "${ROOT}/summary.json" \
    "${ROOT}/recommended.env"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "=================================================="
echo "R9 / 8xH200 throughput benchmark"
echo "effective global batch always = 64"
echo "SAM=${SAM_ROOT}"
echo "=================================================="

test -d "${SAM_ROOT}" || {
    echo "ERROR: SAM root missing: ${SAM_ROOT}"
    exit 1
}

run_profile () {
    local batch="$1"
    local accum="$2"
    local tag="b${batch}_a${accum}"

    local timing="${ROOT}/${tag}_timing.json"
    local log="${ROOT}/${tag}.log"

    rm -f "${timing}" "${log}"
    rm -rf "${UFO_OUTPUT_ROOT}/h200_r9_benchmark/${tag}"

    echo
    echo "--------------------------------------------------"
    echo "${tag}: 8 x ${batch} x ${accum} = $((8*batch*accum))"
    echo "--------------------------------------------------"

    set +e

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    "${UFO_TORCHRUN_BIN}" \
        --standalone \
        --nproc_per_node=8 \
        main.py \
        --config "${CONFIG}" \
        --project h200_r9_benchmark \
        --exp_name "${tag}" \
        --output_dir "${UFO_OUTPUT_ROOT}" \
        --data_root "${UFO_DATA_ROOT}" \
        --sam_track_root "${SAM_ROOT}" \
        --batch_size "${batch}" \
        --gradient_accumulation_steps "${accum}" \
        --ddp_accumulation_no_sync \
        --num_iterations 20 \
        --benchmark_timing_output "${timing}" \
        --benchmark_warmup_steps 5 \
        2>&1 | tee "${log}"

    status=${PIPESTATUS[0]}

    set -e

    if [[ ${status} -ne 0 ]]; then
        if grep -qiE \
            'out of memory|CUDA.*out of memory|CUDA.*OOM' \
            "${log}"
        then
            printf "%s\tOOM\tOOM\tOOM\tOOM\n" "${tag}" \
                >> "${ROOT}/results.tsv"
            echo "${tag}: OOM"
            return
        fi

        echo "${tag}: NON-OOM FAILURE"
        exit "${status}"
    fi

    test -f "${timing}" || {
        echo "${tag}: timing file missing"
        exit 1
    }

    "${UFO_PYTHON_BIN}" - \
        "${tag}" "${timing}" "${ROOT}/results.tsv" <<'PY'
import json
import math
import sys

tag, timing_file, output = sys.argv[1:]

x = json.load(open(timing_file))

forward = float(x["forward_seconds_mean"])
backward = float(x["backward_seconds_mean"])
optimizer = float(x["optimizer_seconds_mean"])
compute = float(x["compute_step_seconds_mean"])
vram = float(x["peak_vram_mb"])
samples = int(x.get("compute_step_samples", 0))

if samples < 1 or not all(math.isfinite(value) and value > 0 for value in (
    forward, backward, optimizer, compute, vram
)):
    raise RuntimeError(f"unstable/non-finite benchmark timing: {x}")

with open(output, "a") as f:
    f.write(
        f"{tag}\t{compute:.6f}\t"
        f"{forward:.6f}\t{backward:.6f}\t{vram:.1f}\n"
    )

print(
    f"{tag}: compute={compute:.4f}s  "
    f"forward={forward:.4f}s  "
    f"backward={backward:.4f}s  "
    f"peak={vram:.0f}MB"
)
PY
}

# Test fastest-looking configurations first.
run_profile 8 1
run_profile 4 2
run_profile 2 4
run_profile 1 8

echo
echo "================ RESULTS ================="
cat "${ROOT}/results.tsv"

"${UFO_PYTHON_BIN}" - "${ROOT}" <<'PY'
import csv
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])

valid = []
records = []

for line in (root / "results.tsv").read_text().splitlines():
    fields = line.split("\t")
    tag = fields[0]

    if fields[1] == "OOM":
        records.append({"candidate": tag, "status": "OOM"})
        continue

    compute = float(fields[1])
    forward = float(fields[2])
    backward = float(fields[3])
    peak = float(fields[4])
    batch = int(tag.split("_")[0][1:])
    accum = int(tag.split("_")[1][1:])
    optimizer_step = compute * accum

    valid.append((optimizer_step, peak, tag, compute))
    records.append({
        "candidate": tag,
        "status": "PASS",
        "batch_size_per_gpu": batch,
        "accumulation_steps": accum,
        "compute_seconds": compute,
        "global64_optimizer_step_seconds": optimizer_step,
        "forward_seconds": forward,
        "backward_seconds": backward,
        "peak_vram_mb": peak,
    })

if not valid:
    raise RuntimeError("all R9 H200 profiles failed")

valid.sort()
optimizer_step, peak, tag, compute = valid[0]

batch = int(tag.split("_")[0][1:])
accum = int(tag.split("_")[1][1:])

(root / "recommended.env").write_text(
    f"H200_BATCH_SIZE={batch}\n"
    f"H200_ACCUMULATION_STEPS={accum}\n"
    f"R9_H200_BATCH_SIZE={batch}\n"
    f"R9_H200_ACCUMULATION_STEPS={accum}\n"
)
(root / "summary.json").write_text(json.dumps({
    "world_size": 8,
    "effective_global_batch": 64,
    "candidates": records,
    "recommended": {
        "candidate": tag,
        "batch_size_per_gpu": batch,
        "accumulation_steps": accum,
        "micro_step_compute_seconds": compute,
        "global64_optimizer_step_seconds": optimizer_step,
        "peak_vram_mb": peak,
    },
}, indent=2) + "\n")
with (root / "results.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        "candidate", "status", "batch_size_per_gpu", "accumulation_steps",
        "compute_seconds", "global64_optimizer_step_seconds",
        "forward_seconds", "backward_seconds", "peak_vram_mb",
    ])
    writer.writeheader()
    writer.writerows(records)

print()
print("BEST =", tag)
print("micro-step compute seconds =", compute)
print("estimated optimizer-step seconds/global64 =", optimizer_step)
print("peak VRAM MB =", peak)
print()
print((root / "recommended.env").read_text())
PY
