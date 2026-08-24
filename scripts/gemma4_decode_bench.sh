#!/usr/bin/env bash
# Gemma 4 batch-one decode repetitions at the locked production point.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
outdir=${1:-"${repo_root}/results/gemma4-decode"}
binary=${BINARY:-"${repo_root}/build-release/strata-gemma4-run"}
model=${MODEL:-"${repo_root}/models/gemma4"}
repetitions=${REPETITIONS:-3}
prompt='The following sequence contains the integers from 1 through 100: 1,'

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
nvidia-smi \
    --query-gpu=index,pci.bus_id,name,power.limit,clocks.sm,clocks.max.sm,clocks.mem,memory.total,memory.used \
    --format=csv,noheader >"${outdir}/gpu-operating-point.txt"

for repetition in $(seq 1 "${repetitions}"); do
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    /usr/bin/time -v "${binary}" \
        --model "${model}" --devices 1 --vram-fraction 0.95 \
        --max-context 512 --max-new 128 --prompt "${prompt}" \
        --temperature 0 --seed 33377335 --phase-profile --quiet --json \
        >"${outdir}/r${repetition}.json" \
        2>"${outdir}/r${repetition}.log"
done

printf 'run\tsteady_tokens\tsteady_seconds\ttok/s\n'
for result in "${outdir}"/r*.json; do
    jq -r --arg run "$(basename "${result}" .json)" '
        [$run, (.metrics.steady_decode_tokens | tostring),
         (.metrics.steady_decode_seconds | tostring),
         (.metrics.steady_decode_tokens /
          .metrics.steady_decode_seconds | tostring)] | @tsv' "${result}"
done
