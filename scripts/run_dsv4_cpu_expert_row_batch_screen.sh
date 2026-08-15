#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_dir=${RESULT_DIR:-"${repo_root}/results/dsv4-cpu-expert-row-batch"}
probe=${PROBE:-"${repo_root}/build-nccl/strata-dsv4-expert-row-batch-probe"}

mkdir -p "${result_dir}"
{
    date --iso-8601=seconds
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" status --short --branch
    lscpu
    numactl --hardware
} >"${result_dir}/system.txt"
git -C "${repo_root}" diff --binary >"${result_dir}/candidate.diff"

for rows in 4 8 12 16; do
    "${probe}" --rows "${rows}" --repetitions 5 \
        >"${result_dir}/rows-${rows}.json"
done

jq -s '{arms: ., minimum_speedup: (map(.speedup) | min),
         exact: (map(.exact) | all)}' \
    "${result_dir}"/rows-*.json >"${result_dir}/summary.json"
jq -e '.exact and .minimum_speedup > 1.05' \
    "${result_dir}/summary.json" >/dev/null
