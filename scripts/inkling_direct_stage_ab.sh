#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_dir=${RESULT_DIR:-"${repo_root}/results/inkling-direct-stage-ab"}
mkdir -p "${result_dir}"

run_arm() {
    local arm=$1
    local repetition=$2
    local direct=0
    if [[ ${arm} == direct ]]; then
        direct=1
    fi
    RESULT_DIR="${result_dir}/${arm}-${repetition}" \
        PINNED_STAGE=$((1 - direct)) WARM_EXPERT_PAGES=0 TOKENS=16 REPEATS=1 \
        "${repo_root}/scripts/run_inkling_decode_profile.sh" \
        >"${result_dir}/${arm}-${repetition}.log" 2>&1
}

run_arm pinned 1
run_arm direct 1
run_arm direct 2
run_arm pinned 2
run_arm pinned 3
run_arm direct 3

rg 'repeat 0:|staged .*H2D|upload split:' "${result_dir}"/*.log
