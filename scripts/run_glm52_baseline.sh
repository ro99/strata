#!/usr/bin/env bash
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID

if [[ "${1:-}" == "--help" ]]; then
    cat <<'EOF'
usage: scripts/run_glm52_baseline.sh

Environment overrides:
  MODEL_DIR, RESULT_DIR, REPETITIONS, MAX_NEW_TOKENS, MAX_CONTEXT_TOKENS,
  CUDA_DEVICES, VRAM_FRACTION, TRACE_ROUTES (0|1), DETAILED_TIMING (0|1),
  FLASH_ATTENTION (0|1)
EOF
    exit 0
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/glm52"}
result_dir=${RESULT_DIR:-"${repo_root}/results/glm52-baseline"}
repetitions=${REPETITIONS:-3}
maximum_new_tokens=${MAX_NEW_TOKENS:-128}
maximum_context_tokens=${MAX_CONTEXT_TOKENS:-256}
devices=${CUDA_DEVICES:-0,1,2}
vram_fraction=${VRAM_FRACTION:-0.85}
trace_routes=${TRACE_ROUTES:-1}
detailed_timing=${DETAILED_TIMING:-0}
flash_attention=${FLASH_ATTENTION:-0}
prompt='What is the closer start to sun, and how distant it is from it?'
expected_index_sha256=43298345833417b1ad2a8b76d012a83d4f2275d532e5ab38e118566f1ac7b12b
model_source=$(findmnt -no SOURCE --target "${model_dir}")
block_name=$(lsblk -no PKNAME "${model_source}" | head -n 1)
if [[ -z "${block_name}" ]]; then
    block_name=$(basename "${model_source}")
fi
block_stat="/sys/class/block/${block_name}/stat"

case "${flash_attention}" in
    0|1) ;;
    *) echo "error: FLASH_ATTENTION must be 0 or 1" >&2; exit 2 ;;
esac

mkdir -p "${result_dir}"
if ! command -v jq >/dev/null 2>&1; then
    echo "error: jq is required to aggregate the benchmark repetitions" >&2
    exit 1
fi

cmake -S "${repo_root}" -B "${repo_root}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_SANITIZERS=OFF \
    -DSTRATA_ENABLE_CUDA=ON
cmake --build "${repo_root}/build" --parallel

actual_index_sha256=$(sha256sum "${model_dir}/model.safetensors.index.json" | awk '{print $1}')
if [[ "${actual_index_sha256}" != "${expected_index_sha256}" ]]; then
    echo "error: checkpoint index SHA-256 does not match the pinned revision" >&2
    exit 1
fi

"${repo_root}/build/strata-inspect" \
    --model "${model_dir}" --headers --json \
    >"${result_dir}/checkpoint.json"
"${repo_root}/build/strata-tokenize" \
    --tokenizer "${model_dir}/tokenizer.json" --model-type glm \
    --prompt "${prompt}" \
    >"${result_dir}/prompt-tokens.txt"

{
    date --iso-8601=seconds
    uname -a
    lscpu
    free -h
    df -h "${model_dir}"
    nvidia-smi -q
    nvidia-smi topo -m
} >"${result_dir}/system.txt"

monitor_pid=
vmstat_pid=
cleanup() {
    if [[ -n "${monitor_pid}" ]]; then
        kill "${monitor_pid}" 2>/dev/null || true
        wait "${monitor_pid}" 2>/dev/null || true
    fi
    if [[ -n "${vmstat_pid}" ]]; then
        kill "${vmstat_pid}" 2>/dev/null || true
        wait "${vmstat_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

nvidia-smi dmon -s pucvmet -d 1 >"${result_dir}/gpu-dmon.log" 2>&1 &
monitor_pid=$!
vmstat 1 >"${result_dir}/vmstat.log" 2>&1 &
vmstat_pid=$!

for ((run = 1; run <= repetitions; ++run)); do
    run_name=$(printf 'run-%02d' "${run}")
    run_args=(
        --model "${model_dir}"
        --prompt "${prompt}"
        --devices "${devices}"
        --vram-fraction "${vram_fraction}"
        --max-context "${maximum_context_tokens}"
        --max-new "${maximum_new_tokens}"
        --json
    )
    if [[ "${trace_routes}" == 1 ]]; then
        run_args+=(--route-trace "${result_dir}/${run_name}.routes.jsonl")
    fi
    if [[ "${detailed_timing}" == 1 ]]; then
        run_args+=(--detailed-timing)
    fi
    if [[ "${flash_attention}" == 1 ]]; then
        run_args+=(--flash-attention)
    else
        run_args+=(--scalar-attention)
    fi
    read -r _ _ read_sectors _ _ _ write_sectors _ <"${block_stat}"
    /usr/bin/time -v \
        "${repo_root}/build/strata-run" "${run_args[@]}" \
        >"${result_dir}/${run_name}.json" \
        2>"${result_dir}/${run_name}.log"
    read -r _ _ final_read_sectors _ _ _ final_write_sectors _ <"${block_stat}"
    physical_read_bytes=$(((final_read_sectors - read_sectors) * 512))
    physical_write_bytes=$(((final_write_sectors - write_sectors) * 512))
    maximum_resident_kib=$(awk -F: '/Maximum resident set size/ {gsub(/^[[:space:]]+/, "", $2); print $2}' \
        "${result_dir}/${run_name}.log")
    printf '{"block_device":"%s","physical_read_bytes":%d,"physical_write_bytes":%d,"maximum_resident_kib":%d}\n' \
        "${block_name}" "${physical_read_bytes}" "${physical_write_bytes}" \
        "${maximum_resident_kib}" \
        >"${result_dir}/${run_name}.io.json"
    if [[ "${trace_routes}" == 1 ]]; then
        jq -e . "${result_dir}/${run_name}.routes.jsonl" >/dev/null
    fi
done

run_files=()
io_files=()
for ((run = 1; run <= repetitions; ++run)); do
    run_name=$(printf 'run-%02d' "${run}")
    run_files+=("${result_dir}/${run_name}.json")
    io_files+=("${result_dir}/${run_name}.io.json")
done
jq -s '
    def median:
        sort as $s | length as $n |
        if ($n % 2) == 1 then $s[$n / 2 | floor]
        else (($s[$n / 2 - 1] + $s[$n / 2]) / 2) end;
    def phase_reconciles($run; $phase):
        (($phase.cuda.devices | map(.weight_h2d_bytes) | add) == $phase.cuda.weight_h2d_bytes) and
        (($phase.cuda.devices | map(.activation_h2d_bytes) | add) == $phase.cuda.activation_h2d_bytes) and
        (($phase.cuda.devices | map(.activation_d2h_bytes) | add) == $phase.cuda.activation_d2h_bytes) and
        (($phase.cache.device_hits | add) == $phase.cache.hits) and
        (($phase.cache.device_misses | add) == $phase.cache.misses) and
        (($phase.cache.device_evictions | add) == $phase.cache.evictions);
    def run_reconciles:
        (.checkpoint_read_bytes == (.phases.prefill.checkpoint_read_bytes + .phases.decode.checkpoint_read_bytes)) and
        (.weight_h2d_bytes == (.phases.prefill.cuda.weight_h2d_bytes + .phases.decode.cuda.weight_h2d_bytes)) and
        (.activation_h2d_bytes == (.phases.prefill.cuda.activation_h2d_bytes + .phases.decode.cuda.activation_h2d_bytes)) and
        (.activation_d2h_bytes == (.phases.prefill.cuda.activation_d2h_bytes + .phases.decode.cuda.activation_d2h_bytes)) and
        (.vram_cache_hits == (.phases.prefill.cache.hits + .phases.decode.cache.hits)) and
        (.vram_cache_misses == (.phases.prefill.cache.misses + .phases.decode.cache.misses)) and
        (.vram_cache_evictions == (.phases.prefill.cache.evictions + .phases.decode.cache.evictions)) and
        phase_reconciles(.; .phases.prefill) and phase_reconciles(.; .phases.decode);
    {
        repetitions: length,
        detailed_timing: map(.detailed_timing),
        accounting_reconciles: map(run_reconciles),
        initialization_seconds: map(.initialization_seconds),
        initialization_seconds_median: (map(.initialization_seconds) | median),
        prompt_processing_tok_s: map(.prompt_processing_tok_s),
        prompt_processing_tok_s_median: (map(.prompt_processing_tok_s) | median),
        prompt_processing_seconds: map(.prompt_processing_seconds),
        prompt_processing_seconds_median: (map(.prompt_processing_seconds) | median),
        generation_tok_s: map(.generation_tok_s),
        generation_tok_s_median: (map(.generation_tok_s) | median),
        generation_seconds: map(.generation_seconds),
        generation_seconds_median: (map(.generation_seconds) | median),
        checkpoint_read_bytes: map(.checkpoint_read_bytes),
        checkpoint_read_bytes_median: (map(.checkpoint_read_bytes) | median),
        prefill_checkpoint_read_bytes: map(.phases.prefill.checkpoint_read_bytes),
        decode_checkpoint_read_bytes: map(.phases.decode.checkpoint_read_bytes),
        weight_h2d_bytes: map(.weight_h2d_bytes),
        weight_h2d_bytes_median: (map(.weight_h2d_bytes) | median),
        prefill_weight_h2d_bytes: map(.phases.prefill.cuda.weight_h2d_bytes),
        decode_weight_h2d_bytes: map(.phases.decode.cuda.weight_h2d_bytes),
        vram_cache_hits: map(.vram_cache_hits),
        vram_cache_hits_median: (map(.vram_cache_hits) | median),
        vram_cache_misses: map(.vram_cache_misses),
        vram_cache_misses_median: (map(.vram_cache_misses) | median),
        vram_cache_evictions: map(.vram_cache_evictions),
        vram_cache_evictions_median: (map(.vram_cache_evictions) | median),
        pinned_resident_spine_bytes: map(.pinned_resident_spine_bytes),
        final_evictable_expert_bytes: map(.evictable_expert_bytes),
        per_device_prefill: map(.phases.prefill.cuda.devices),
        per_device_decode: map(.phases.decode.cuda.devices)
    }
' "${run_files[@]}" >"${result_dir}/summary-runtime.json"
jq -s '
    def median:
        sort as $s | length as $n |
        if ($n % 2) == 1 then $s[$n / 2 | floor]
        else (($s[$n / 2 - 1] + $s[$n / 2]) / 2) end;
    {
        physical_read_bytes: map(.physical_read_bytes),
        physical_read_bytes_median: (map(.physical_read_bytes) | median),
        physical_write_bytes: map(.physical_write_bytes),
        physical_write_bytes_median: (map(.physical_write_bytes) | median),
        maximum_resident_kib: map(.maximum_resident_kib),
        maximum_resident_kib_median: (map(.maximum_resident_kib) | median)
    }
' "${io_files[@]}" >"${result_dir}/summary-io.json"
jq -s '.[0] + .[1]' \
    "${result_dir}/summary-runtime.json" "${result_dir}/summary-io.json" \
    >"${result_dir}/summary.json"

cleanup
monitor_pid=
vmstat_pid=

{
    date --iso-8601=seconds
    free -h
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
        --format=csv
} >"${result_dir}/final-system.txt"

cat "${result_dir}/summary.json"
