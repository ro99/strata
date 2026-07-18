#!/usr/bin/env bash
# Portuguese greedy generation check for the mHC combination-orientation fix.
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/DeepSeek-V4-Flash-DSpark"}
result_dir="${repo_root}/results/deepseek-v4-pt-br-mhc-fix"
devices=${CUDA_DEVICES:-0,1,2}
runs=${RUNS:-1}
max_new=${MAX_NEW_TOKENS:-64}
mkdir -p "${result_dir}"

for run in $(seq 1 "${runs}"); do
    tag=$(printf 'pt-br-mhc-fix-%02d' "${run}")
    "${repo_root}/build/strata-deepseek-run" \
        --model "${model_dir}" \
        --devices "${devices}" \
        --max-context 2048 \
        --max-new "${max_new}" \
        --logit-trace --logit-trace-top-k 20 \
        --layer-hash-trace \
        --json \
        --prompt 'voce fala portugues? o que e caipirinha?' \
        2>"${result_dir}/${tag}.stderr" \
        >"${result_dir}/${tag}.json"
done
