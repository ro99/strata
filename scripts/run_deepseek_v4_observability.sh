#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/DeepSeek-V4-Flash-DSpark"}
result_dir=${RESULT_DIR:-"${repo_root}/results/deepseek-v4-observability"}
capture_telemetry=${CAPTURE_TELEMETRY:-1}
runner=${RUNNER:-"${repo_root}/build/strata-deepseek-run"}
mkdir -p "${result_dir}"

telemetry_pid=
cleanup() {
    if [[ -n "${telemetry_pid}" ]]; then
        kill "${telemetry_pid}" 2>/dev/null || true
        wait "${telemetry_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ "${capture_telemetry}" == 1 ]]; then
    nvidia-smi dmon -s pucvmet -d 1 -o DT >"${result_dir}/nvidia-dmon.txt" &
    telemetry_pid=$!
fi

{
    date --iso-8601=seconds
    git -C "${repo_root}" rev-parse HEAD
    printf '%s\n' "${runner}"
    if [[ "${capture_telemetry}" == 1 ]]; then
        free -b
        nvidia-smi --query-gpu=index,name,pci.bus_id,memory.free,memory.total \
            --format=csv
    fi
} >"${result_dir}/system-before.txt"
git -C "${repo_root}" diff --binary >"${result_dir}/candidate.diff"

/usr/bin/time -v "${runner}" \
    --model "${model_dir}" \
    --devices 0,1,2 \
    --host-memory 216G \
    --vram-fraction 0.85 \
    --max-context 2048 \
    --max-new 128 \
    --prompt Hello \
    --detailed-timing \
    --json \
    >"${result_dir}/generation.json" \
    2>"${result_dir}/generation.log"

cleanup
telemetry_pid=

{
    date --iso-8601=seconds
    if [[ "${capture_telemetry}" == 1 ]]; then
        free -b
        nvidia-smi --query-gpu=index,name,pci.bus_id,memory.free,memory.total \
            --format=csv
    fi
} >"${result_dir}/system-after.txt"

jq '{
    prompt_tokens,
    decode_steps,
    prefill_seconds,
    decode_seconds,
    decode_tok_s: (.decode_steps / .decode_seconds),
    generated_token_ids,
    phases
}' "${result_dir}/generation.json" >"${result_dir}/summary.json"
