#!/usr/bin/env bash
# Interleaved Gemma 4 three-page production-prefill A/B.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
outdir=${1:-"${repo_root}/results/gemma4-production-prefill-ab"}
binary=${BINARY:-"${repo_root}/build-release/strata-gemma4-run"}
model=${MODEL:-"${repo_root}/models/gemma4"}
repetitions=${REPETITIONS:-3}
prompt=
for ((index = 0; index < 334; ++index)); do
    prompt+="France "
done

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

run_arm() {
    local repetition=$1
    local arm=$2
    local enabled=1
    if [[ "${arm}" == control ]]; then enabled=0; fi
    CUDA_DEVICE_ORDER=PCI_BUS_ID STRATA_GEMMA4_DEVICE_PAGE=${enabled} \
    /usr/bin/time -v "${binary}" \
        --model "${model}" --devices 1 --vram-fraction 0.95 \
        --max-context 512 --max-new 1 --prompt "${prompt}" \
        --temperature 0 --seed 33377335 --quiet --json \
        >"${outdir}/r${repetition}-${arm}.json" \
        2>"${outdir}/r${repetition}-${arm}.log"
}

for repetition in $(seq 1 "${repetitions}"); do
    if ((repetition % 2 == 1)); then
        run_arm "${repetition}" candidate
        run_arm "${repetition}" control
    else
        run_arm "${repetition}" control
        run_arm "${repetition}" candidate
    fi
done

printf 'run\tarm\ttokens\tseconds\ttok/s\tfirst_token\n'
for result in "${outdir}"/r*-*.json; do
    run=$(basename "${result}" .json)
    jq -r --arg run "${run}" '
        [$run, (.metrics.prefill_tokens | tostring),
         (.metrics.prefill_seconds | tostring),
         (.metrics.prefill_tokens / .metrics.prefill_seconds | tostring),
         (.generated_token_ids[0] | tostring)] | @tsv' "${result}" |
        awk -F '\t' 'BEGIN {OFS="\t"} {
            split($1, parts, "-"); print parts[1], parts[2], $2, $3, $4, $5
        }'
done
