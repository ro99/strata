#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_dir=${RESULT_DIR:-"${repo_root}/results/gemma4-smoke"}
max_new=${MAX_NEW:-1}
sampling=()
if [[ ${FUTURE_ENTROPY_SMOKE:-0} == 1 ]]; then
    sampling=(--future-entropy 2 --future-entropy-top-n 2 --alpha 0)
fi
mkdir -p "${result_dir}"

{
    date --iso-8601=seconds
    git -C "${repo_root}" rev-parse HEAD
    free -b
    nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv
} >"${result_dir}/system-before.txt"
git -C "${repo_root}" diff --binary >"${result_dir}/candidate.diff"

/usr/bin/time -v "${repo_root}/build/strata-chat" \
    --model "${repo_root}/models/gemma4" \
    --model-type gemma4 \
    --devices 0,1,2 \
    --context-size 128 \
    --max-new "${max_new}" \
    --prompt Hello \
    "${sampling[@]}" \
    >"${result_dir}/generation.txt" \
    2>"${result_dir}/generation.log"

{
    date --iso-8601=seconds
    free -b
    nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv
} >"${result_dir}/system-after.txt"
