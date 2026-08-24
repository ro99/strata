#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
binary=${BINARY:-"${repo_dir}/build-release/strata-inkling-attention-probe"}
results_dir=${RESULTS_DIR:-"${repo_dir}/results/inkling-attention-microbench"}
mkdir -p "${results_dir}"
log="${results_dir}/target-shape.log"
: >"${log}"

export CUDA_DEVICE_ORDER=FASTEST_FIRST
for repetition in 0 1 2; do
    for rows in 512 4096 16384; do
        for device in 0 2; do
            printf 'repetition=%s device=%s ' "${repetition}" "${device}" | tee -a "${log}"
            "${binary}" --rows "${rows}" --repeats 20 --device "${device}" | tee -a "${log}"
        done
    done
done
