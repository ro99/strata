#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
baseline_repo=${BASELINE_REPO:?set BASELINE_REPO to a fac2f20 worktree}
model_dir=${MODEL_DIR:-"${repo_root}/models/glm52"}
result_root=${RESULT_ROOT:-"${repo_root}/results/glm-w4a16-absorbed-attention-ab"}

run_arm() {
    local source_root=$1
    local result_dir=$2
    env MODEL_DIR="${model_dir}" RESULT_DIR="${result_dir}" REPETITIONS=1 \
        MAX_NEW_TOKENS=2 MAX_CONTEXT_TOKENS=256 CUDA_DEVICES=0,1,2 \
        VRAM_FRACTION=.85 TRACE_ROUTES=1 DETAILED_TIMING=1 FLASH_ATTENTION=1 \
        "${source_root}/scripts/run_glm52_baseline.sh"
}

for repetition in 1 2 3; do
    suffix=$(printf '%02d' "${repetition}")
    run_arm "${baseline_repo}" "${result_root}/baseline-${suffix}"
    run_arm "${repo_root}" "${result_root}/absorbed-${suffix}"
done
