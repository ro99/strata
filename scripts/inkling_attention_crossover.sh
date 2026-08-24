#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_dir=${RESULT_DIR:-"${repo_root}/results/inkling-attention-crossover"}
mkdir -p "${result_dir}"

run_arm() {
    local arm=$1
    local tokens=$2
    local host=0
    if [[ ${arm} == host ]]; then
        host=1
    fi
    RESULT_DIR="${result_dir}/${arm}-${tokens}" HOST_ATTENTION=${host} \
        DEVICE_ATTENTION_MIN_ROWS=32 \
        WARM_EXPERT_PAGES=0 TOKENS=${tokens} REPEATS=1 \
        "${repo_root}/scripts/run_inkling_decode_profile.sh" \
        >"${result_dir}/${arm}-${tokens}.log" 2>&1
}

run_arm host 32
run_arm device 32
run_arm device 64
run_arm host 64

rg 'repeat 0:|attention ' "${result_dir}"/*.log
