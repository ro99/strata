#!/usr/bin/env bash
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/DeepSeek-V4-Flash-DSpark"}
result_dir=${RESULT_DIR:-"${repo_root}/results/strata-chat-smoke"}
devices=${CUDA_DEVICES:-0,1,2}
maximum_context=${MAX_CONTEXT_TOKENS:-8192}
maximum_new=${MAX_NEW_TOKENS:-16}
prompt=${PROMPT:-hello}
temperature=${TEMPERATURE:-0}
vram_fraction=${VRAM_FRACTION:-}
seed=${SAMPLING_SEED:-33377335}

mkdir -p "${result_dir}"

command_line=$(printf '%q ' \
    "${repo_root}/build/strata-chat" \
    --model "${model_dir}" \
    --model-type deepseek \
    --devices "${devices}" \
    --context-size "${maximum_context}" \
    --max-new "${maximum_new}" \
    --temperature "${temperature}" \
    --seed "${seed}" \
    ${vram_fraction:+--vram-fraction "${vram_fraction}"} \
    --prompt "${prompt}")

script --quiet --return --command "${command_line}" \
    "${result_dir}/terminal.typescript"
