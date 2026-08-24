#!/usr/bin/env bash
# Counterbalanced Gemma 4 text-prefill GeGLU A/B at the production point.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
outdir=${1:-"${repo_root}/results/gemma4-prefill/geglu-ab"}
binary=${BINARY:-"${repo_root}/build-release/strata-gemma4-run"}
model=${MODEL:-"${repo_root}/models/gemma4"}
prompt=${PROMPT:-"Count from one to forty, separated by commas."}
repetitions=${REPETITIONS:-3}

build_dir=$(dirname "${binary}")
build_type=$(sed -n 's/^CMAKE_BUILD_TYPE:STRING=//p' \
                 "${build_dir}/CMakeCache.txt")
if [[ ! -x "${binary}" || "${build_type}" != Release ]]; then
    echo "error: ${binary} must come from a Release build" >&2
    exit 2
fi
mkdir -p "${outdir}"
nvidia-smi --query-gpu=index,pci.bus_id,name,power.limit,clocks.sm,clocks.max.sm,memory.total,memory.used \
    --format=csv,noheader >"${outdir}/gpu-operating-point.txt"

run_arm() {
    local arm=$1
    local enabled=$2
    local repetition=$3
    local stem="${outdir}/${arm}-rep${repetition}"
    echo "${arm} repetition ${repetition} $(date -Is)"
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    STRATA_GEMMA4_PARALLEL_PREFILL="${enabled}" \
    /usr/bin/time -v "${binary}" \
        --model "${model}" --devices 1 --vram-fraction 0.95 \
        --max-context 512 --max-new 32 \
        --prompt "${prompt}" --temperature 0 --seed 33377335 \
        --quiet --json --route-census "${stem}.census.json" \
        >"${stem}.json" 2>"${stem}.log"
}

for repetition in $(seq 1 "${repetitions}"); do
    if ((repetition % 2 == 1)); then
        run_arm parallel 1 "${repetition}"
        run_arm scalar 0 "${repetition}"
    else
        run_arm scalar 0 "${repetition}"
        run_arm parallel 1 "${repetition}"
    fi
done

for arm in parallel scalar; do
    for repetition in $(seq 1 "${repetitions}"); do
        jq -r --arg arm "${arm}" --arg repetition "${repetition}" \
            '[ $arm, $repetition, .metrics.prefill_tokens,
               .metrics.prefill_seconds, .metrics.steady_decode_tokens,
               .metrics.steady_decode_seconds ] | @tsv' \
            "${outdir}/${arm}-rep${repetition}.json"
    done
done >"${outdir}/summary.tsv"
cat "${outdir}/summary.tsv"
