#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_dir=${RESULT_DIR:-"${repo_root}/results/inkling-expert-parallel/ab"}
tokens=${TOKENS:-128}
mkdir -p "${result_dir}"

run_arm() {
    local arm=$1
    local repetition=$2
    local expert_parallel=0
    if [[ ${arm} == expert-parallel ]]; then
        expert_parallel=1
    fi
    RESULT_DIR="${result_dir}/${arm}-${repetition}" \
        EXPERT_PARALLEL=${expert_parallel} WARM_EXPERT_PAGES=0 \
        TOKENS=${tokens} REPEATS=1 \
        "${repo_root}/scripts/run_inkling_decode_profile.sh" \
        >"${result_dir}/${arm}-${repetition}.log" 2>&1
}

run_arm layer-local 1
run_arm expert-parallel 1
run_arm expert-parallel 2
run_arm layer-local 2
run_arm layer-local 3
run_arm expert-parallel 3

rg 'repeat 0:|decode staged|continuation:|route census:' "${result_dir}"/*.log
