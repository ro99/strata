#!/usr/bin/env bash
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/DeepSeek-V4-Flash-DSpark"}
result_dir=${RESULT_DIR:-"${repo_root}/results/deepseek-v4-smoke"}
devices=${CUDA_DEVICES:-0,1,2}
host_memory=${HOST_MEMORY:-216G}
vram_fraction=${VRAM_FRACTION:-0.85}
maximum_context=${MAX_CONTEXT_TOKENS:-2048}
maximum_new=${MAX_NEW_TOKENS:-2}
prompt=${PROMPT:-Hi}
expected_index_sha256=98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b

mkdir -p "${result_dir}"
cmake -S "${repo_root}" -B "${repo_root}/build" \
    -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON
cmake --build "${repo_root}/build" --parallel

actual_index_sha256=$(sha256sum "${model_dir}/model.safetensors.index.json" | awk '{print $1}')
if [[ "${actual_index_sha256}" != "${expected_index_sha256}" ]]; then
    echo "error: DeepSeek checkpoint index SHA-256 does not match the pinned revision" >&2
    exit 1
fi

"${repo_root}/build/strata-inspect" --model "${model_dir}" --headers --json \
    >"${result_dir}/checkpoint.json"
"${repo_root}/build/strata-deepseek-run" \
    --model "${model_dir}" --devices "${devices}" \
    --host-memory "${host_memory}" --vram-fraction "${vram_fraction}" \
    --max-context "${maximum_context}" --admission-only --json --quiet \
    >"${result_dir}/admission.json"

{
    date --iso-8601=seconds
    free -h
    nvidia-smi --query-gpu=index,name,memory.free,memory.total \
        --format=csv
} >"${result_dir}/system-before.txt"

/usr/bin/time -v "${repo_root}/build/strata-deepseek-run" \
    --model "${model_dir}" --devices "${devices}" \
    --host-memory "${host_memory}" --vram-fraction "${vram_fraction}" \
    --max-context "${maximum_context}" --max-new "${maximum_new}" \
    --prompt "${prompt}" --route-trace "${result_dir}/routes.jsonl" \
    --json \
    >"${result_dir}/generation.json" \
    2>"${result_dir}/generation.log"

{
    date --iso-8601=seconds
    free -h
    nvidia-smi --query-gpu=index,name,memory.free,memory.total \
        --format=csv
} >"${result_dir}/system-after.txt"

cat "${result_dir}/generation.json"
