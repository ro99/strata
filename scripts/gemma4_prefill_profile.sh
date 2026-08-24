#!/usr/bin/env bash
# Gemma 4 single-3090 prefill cost-model arm at the production operating point.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
outdir=${1:-"${repo_root}/results/gemma4-prefill/profile-main"}
binary=${BINARY:-"${repo_root}/build-release/strata-gemma4-run"}
model=${MODEL:-"${repo_root}/models/gemma4"}
prompt=${PROMPT:-"Count from one to forty, separated by commas."}
max_new=${MAX_NEW:-32}

if [[ ! -x "${binary}" ]]; then
    echo "error: ${binary} is missing" >&2
    exit 2
fi
build_dir=$(dirname "${binary}")
build_type=$(sed -n 's/^CMAKE_BUILD_TYPE:STRING=//p' \
                 "${build_dir}/CMakeCache.txt")
if [[ "${build_type}" != Release ]]; then
    echo "error: ${build_dir} is CMAKE_BUILD_TYPE=${build_type:-unset}, not Release" >&2
    exit 2
fi

mkdir -p "${outdir}"
nvidia-smi --query-gpu=index,pci.bus_id,name,power.limit,clocks.sm,clocks.max.sm,clocks.mem,memory.total,memory.used \
    --format=csv,noheader >"${outdir}/gpu-operating-point.txt"

CUDA_DEVICE_ORDER=PCI_BUS_ID \
/usr/bin/time -v "${binary}" \
    --model "${model}" --devices 1 --vram-fraction 0.95 \
    --max-context 512 --max-new "${max_new}" \
    --prompt "${prompt}" --temperature 0 --seed 33377335 \
    --phase-profile --quiet --json \
    --route-census "${outdir}/route-census.json" \
    >"${outdir}/generation.json" 2>"${outdir}/run.log"
