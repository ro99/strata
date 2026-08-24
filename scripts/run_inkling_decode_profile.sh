#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_dir=${RESULT_DIR:-"${repo_root}/results/inkling-decode-profile"}
model_dir=${MODEL_DIR:-"${repo_root}/models/inkling"}
runner=${RUNNER:-"${repo_root}/build-release/strata-inkling-probe"}
devices=${DEVICES:-"0,1,2"}
tokens=${TOKENS:-16}
repeats=${REPEATS:-3}
stage_args=()
if [[ ${PINNED_STAGE:-0} == 1 ]]; then
    stage_args+=(--pinned-stage)
fi
if [[ ${WARM_EXPERT_PAGES:-1} == 0 ]]; then
    stage_args+=(--no-warm)
fi
if [[ ${HOST_ATTENTION:-0} == 1 ]]; then
    stage_args+=(--host-attention)
fi
if [[ -n ${DEVICE_ATTENTION_MIN_ROWS:-} ]]; then
    stage_args+=(--device-attention-min-rows "${DEVICE_ATTENTION_MIN_ROWS}")
fi
if [[ ${WEIGHT_ARENA:-1} == 0 ]]; then
    stage_args+=(--no-weight-arena)
fi
if [[ ${DEFER_EXPERT_UPLOADS:-1} == 0 ]]; then
    stage_args+=(--sync-expert-uploads)
fi
if [[ ${EXPERT_PARALLEL:-0} == 1 ]]; then
    stage_args+=(--expert-parallel)
fi

mkdir -p "${result_dir}"
export CUDA_DEVICE_ORDER=FASTEST_FIRST

{
    git -C "${repo_root}" rev-parse HEAD
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.used \
        --format=csv,noheader
    printf 'model=%s devices=%s tokens=%s repeats=%s\n' \
        "${model_dir}" "${devices}" "${tokens}" "${repeats}"
} >"${result_dir}/environment.log"

exec "${runner}" --model "${model_dir}" --devices "${devices}" \
    --prompt 'The capital of France is' --tokens "${tokens}" \
    --repeat "${repeats}" "${stage_args[@]}"
