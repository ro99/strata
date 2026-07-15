#!/usr/bin/env bash
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/glm52"}
result_dir=${RESULT_DIR:-"${repo_root}/results/glm52-host-expert-matrix"}
repetitions=${REPETITIONS:-3}
maximum_new_tokens=${MAX_NEW_TOKENS:-128}
maximum_context_tokens=${MAX_CONTEXT_TOKENS:-256}
devices=${CUDA_DEVICES:-0,1,2}
vram_fraction=${VRAM_FRACTION:-0.95}
host_workers=${HOST_WORKERS:-36}
prompt='What is the closer start to sun, and how distant it is from it?'
expected_index_sha256=43298345833417b1ad2a8b76d012a83d4f2275d532e5ab38e118566f1ac7b12b

mkdir -p "${result_dir}"
command -v jq >/dev/null 2>&1 || {
    echo "error: jq is required" >&2
    exit 1
}

cmake -S "${repo_root}" -B "${repo_root}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_SANITIZERS=OFF \
    -DSTRATA_ENABLE_CUDA=ON
cmake --build "${repo_root}/build" --parallel
ctest --test-dir "${repo_root}/build" --output-on-failure

actual_index_sha256=$(sha256sum \
    "${model_dir}/model.safetensors.index.json" | awk '{print $1}')
if [[ "${actual_index_sha256}" != "${expected_index_sha256}" ]]; then
    echo "error: checkpoint index SHA-256 does not match the pinned revision" >&2
    exit 1
fi

model_source=$(findmnt -no SOURCE --target "${model_dir}")
block_name=$(lsblk -no PKNAME "${model_source}" | head -n 1)
if [[ -z "${block_name}" ]]; then
    block_name=$(basename "${model_source}")
fi
block_stat="/sys/class/block/${block_name}/stat"

{
    date --iso-8601=seconds
    git -C "${repo_root}" status --short --branch
    git -C "${repo_root}" rev-parse HEAD
    uname -a
    lscpu
    free -h
    nvidia-smi -q
    nvidia-smi topo -m
} >"${result_dir}/system-before.txt"

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

run_one() {
    local variant=$1
    local repetition=$2
    local name
    name=$(printf '%s-%02d' "${variant}" "${repetition}")
    local args=(
        --model "${model_dir}"
        --prompt "${prompt}"
        --devices "${devices}"
        --vram-fraction "${vram_fraction}"
        --max-context "${maximum_context_tokens}"
        --max-new "${maximum_new_tokens}"
        --route-trace "${result_dir}/${name}.routes.jsonl"
        --json
    )
    if [[ "${variant}" == host ]]; then
        args+=(--host-cold-experts --host-workers "${host_workers}")
    fi
    read -r _ _ read_sectors _ _ _ write_sectors _ <"${block_stat}"
    /usr/bin/time -v "${repo_root}/build/strata-run" "${args[@]}" \
        >"${result_dir}/${name}.json" \
        2>"${result_dir}/${name}.log"
    read -r _ _ final_read_sectors _ _ _ final_write_sectors _ <"${block_stat}"
    local physical_read_bytes=$(((final_read_sectors - read_sectors) * 512))
    local physical_write_bytes=$(((final_write_sectors - write_sectors) * 512))
    local maximum_resident_kib
    maximum_resident_kib=$(awk -F: \
        '/Maximum resident set size/ {gsub(/^[[:space:]]+/, "", $2); print $2}' \
        "${result_dir}/${name}.log")
    printf '{"block_device":"%s","physical_read_bytes":%d,"physical_write_bytes":%d,"maximum_resident_kib":%d}\n' \
        "${block_name}" "${physical_read_bytes}" "${physical_write_bytes}" \
        "${maximum_resident_kib}" >"${result_dir}/${name}.io.json"
    jq -e . "${result_dir}/${name}.json" >/dev/null
    jq -e . "${result_dir}/${name}.routes.jsonl" >/dev/null
}

for ((repetition = 1; repetition <= repetitions; ++repetition)); do
    if ((repetition % 2 == 1)); then
        run_one current "${repetition}"
        run_one host "${repetition}"
    else
        run_one host "${repetition}"
        run_one current "${repetition}"
    fi
    suffix=$(printf '%02d' "${repetition}")
    cmp "${result_dir}/current-${suffix}.routes.jsonl" \
        "${result_dir}/host-${suffix}.routes.jsonl"
    jq -e -n \
        --slurpfile current "${result_dir}/current-${suffix}.json" \
        --slurpfile host "${result_dir}/host-${suffix}.json" \
        '$current[0].generated_token_ids == $host[0].generated_token_ids' >/dev/null
done

run_files=()
io_files=()
for variant in current host; do
    for ((repetition = 1; repetition <= repetitions; ++repetition)); do
        suffix=$(printf '%02d' "${repetition}")
        run_files+=("${result_dir}/${variant}-${suffix}.json")
        io_files+=("${result_dir}/${variant}-${suffix}.io.json")
    done
done

jq -s --argjson repetitions "${repetitions}" '
    def median:
        sort as $s | length as $n |
        if ($n % 2) == 1 then $s[$n / 2 | floor]
        else (($s[$n / 2 - 1] + $s[$n / 2]) / 2) end;
    . as $runs |
    {
        repetitions: $repetitions,
        current_generation_tok_s:
            ($runs[0:$repetitions] | map(.generation_tok_s)),
        host_generation_tok_s:
            ($runs[$repetitions:] | map(.generation_tok_s)),
        current_generation_tok_s_median:
            ($runs[0:$repetitions] | map(.generation_tok_s) | median),
        host_generation_tok_s_median:
            ($runs[$repetitions:] | map(.generation_tok_s) | median),
        speedup:
            (($runs[$repetitions:] | map(.generation_tok_s) | median) /
             ($runs[0:$repetitions] | map(.generation_tok_s) | median)),
        generated_tokens_equal:
            ([range(0; $repetitions) as $i |
              $runs[$i].generated_token_ids ==
              $runs[$repetitions + $i].generated_token_ids]),
        host_decode_weight_h2d_bytes:
            ($runs[$repetitions:] | map(.phases.decode.cuda.weight_h2d_bytes)),
        host_experts:
            ($runs[$repetitions:] | map(.phases.decode.host_experts)),
        host_expert_service_seconds:
            ($runs[$repetitions:] | map(.phases.decode.host_expert_service_seconds)),
        host_mapping_sweeps:
            ($runs[$repetitions:] | map(.phases.decode.host_mapping_sweeps))
    }
' "${run_files[@]}" >"${result_dir}/summary-runtime.json"

jq -s --argjson repetitions "${repetitions}" '
    def median:
        sort as $s | length as $n |
        if ($n % 2) == 1 then $s[$n / 2 | floor]
        else (($s[$n / 2 - 1] + $s[$n / 2]) / 2) end;
    . as $runs |
    {
        current_maximum_resident_kib:
            ($runs[0:$repetitions] | map(.maximum_resident_kib)),
        host_maximum_resident_kib:
            ($runs[$repetitions:] | map(.maximum_resident_kib)),
        host_maximum_resident_kib_median:
            ($runs[$repetitions:] | map(.maximum_resident_kib) | median),
        current_physical_read_bytes:
            ($runs[0:$repetitions] | map(.physical_read_bytes)),
        host_physical_read_bytes:
            ($runs[$repetitions:] | map(.physical_read_bytes))
    }
' "${io_files[@]}" >"${result_dir}/summary-io.json"

jq -s '.[0] + .[1]' \
    "${result_dir}/summary-runtime.json" \
    "${result_dir}/summary-io.json" \
    >"${result_dir}/summary.json"

cleanup
monitor_pid=
vmstat_pid=
{
    date --iso-8601=seconds
    free -h
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
        --format=csv
} >"${result_dir}/system-after.txt"
cat "${result_dir}/summary.json"
