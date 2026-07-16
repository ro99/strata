#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
baseline_runner=${BASELINE_RUNNER:?set BASELINE_RUNNER to the baseline executable}
candidate_runner=${CANDIDATE_RUNNER:-"${repo_root}/build/strata-deepseek-run"}
result_root=${RESULT_ROOT:-"${repo_root}/results/deepseek-v4-ab-matrix"}
repetitions=${REPETITIONS:-3}

mkdir -p "${result_root}"
: >"${result_root}/order.txt"
for repetition in $(seq 1 "${repetitions}"); do
    for variant in baseline candidate; do
        runner=${baseline_runner}
        if [[ "${variant}" == candidate ]]; then runner=${candidate_runner}; fi
        result_dir="${result_root}/${variant}-${repetition}"
        printf '%s %s %s\n' "${repetition}" "${variant}" "${runner}" \
            | tee -a "${result_root}/order.txt"
        CAPTURE_TELEMETRY=0 RUNNER="${runner}" RESULT_DIR="${result_dir}" \
            "${repo_root}/scripts/run_deepseek_v4_observability.sh"
    done
done
