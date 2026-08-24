#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_dir=${RESULT_DIR:-"${repo_root}/results/inkling-deferred-upload"}
tokens=${TOKENS:-16}
mkdir -p "${result_dir}"

run_arm() {
    local arm=$1
    local repetition=$2
    local defer=1
    if [[ ${arm} == synchronous ]]; then
        defer=0
    fi
    RESULT_DIR="${result_dir}/${arm}-${repetition}" \
        DEFER_EXPERT_UPLOADS=${defer} WARM_EXPERT_PAGES=0 \
        TOKENS=${tokens} REPEATS=1 \
        "${repo_root}/scripts/run_inkling_decode_profile.sh" \
        >"${result_dir}/${arm}-${repetition}.log" 2>&1
}

run_arm synchronous 1
run_arm deferred 1
run_arm deferred 2
run_arm synchronous 2
run_arm synchronous 3
run_arm deferred 3

rg 'repeat 0:|decode staged|decode upload/kernel|continuation:|route census:' \
    "${result_dir}"/*.log
