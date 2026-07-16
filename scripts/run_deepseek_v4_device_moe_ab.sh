#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Reproducible DeepSeek-V4 reference/device-MoE comparison.
#
# Default performance matrix: three repetitions per variant in ABBAAB order,
# where A is the reference path and B adds --device-moe. Performance runs leave
# diagnostic tracing disabled. Use CORRECTNESS_RUN=1 for a separate one-pair
# trace gate with --logit-trace and --layer-hash-trace.
#
# Common overrides:
#   RESULT_DIR=/ignored/deterministic/path
#   REFERENCE_BINARY=/path/to/reference/strata-deepseek-run
#   CANDIDATE_BINARY=/path/to/candidate/strata-deepseek-run
#   REPETITIONS=3
#   SMOKE=1 MAX_NEW_TOKENS=2
#   CORRECTNESS_RUN=1 LOGIT_TRACE_TOP_K=20
#   MODEL_DIR=... CUDA_DEVICES=0,1,2 HOST_MEMORY=216G
#   VRAM_FRACTION=0.85 MAX_CONTEXT_TOKENS=2048 PROMPT=Hello
#   REFERENCE_ARGS='...'
#   CANDIDATE_ARGS='...'
#
# Suggested launch:
#   tmux new-session -d -s dsv4-device-moe-ab \
#     -c /home/rodrigo/Developer/strata \
#     './scripts/run_deepseek_v4_device_moe_ab.sh'

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/DeepSeek-V4-Flash-DSpark"}
reference_binary=${REFERENCE_BINARY:-"${repo_root}/build/strata-deepseek-run"}
candidate_binary=${CANDIDATE_BINARY:-"${repo_root}/build/strata-deepseek-run"}
smoke=${SMOKE:-0}
correctness_run=${CORRECTNESS_RUN:-0}
logit_trace_top_k=${LOGIT_TRACE_TOP_K:-20}
devices=${CUDA_DEVICES:-0,1,2}
host_memory=${HOST_MEMORY:-216G}
host_memory_limit_kib=${HOST_MEMORY_LIMIT_KIB:-226492416}
vram_fraction=${VRAM_FRACTION:-0.85}
maximum_context_tokens=${MAX_CONTEXT_TOKENS:-2048}
prompt=${PROMPT:-Hello}
minimum_decode_steps_per_second=${MIN_DECODE_STEPS_PER_SECOND:-5.0}
enforce_acceptance=${ENFORCE_ACCEPTANCE:-0}
expected_index_sha256=98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b

case "${smoke}" in
    0|1) ;;
    *) echo "error: SMOKE must be 0 or 1" >&2; exit 2 ;;
esac
case "${correctness_run}" in
    0|1) ;;
    *) echo "error: CORRECTNESS_RUN must be 0 or 1" >&2; exit 2 ;;
esac
case "${enforce_acceptance}" in
    0|1) ;;
    *) echo "error: ENFORCE_ACCEPTANCE must be 0 or 1" >&2; exit 2 ;;
esac

if [[ -n ${REPETITIONS+x} ]]; then
    repetitions=${REPETITIONS}
elif [[ "${smoke}" == 1 || "${correctness_run}" == 1 ]]; then
    repetitions=1
else
    repetitions=3
fi
if [[ -n ${MAX_NEW_TOKENS+x} ]]; then
    maximum_new_tokens=${MAX_NEW_TOKENS}
elif [[ "${smoke}" == 1 ]]; then
    maximum_new_tokens=2
else
    maximum_new_tokens=128
fi

if [[ ! "${repetitions}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: REPETITIONS must be a positive integer" >&2
    exit 2
fi
if [[ ! "${maximum_new_tokens}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: MAX_NEW_TOKENS must be a positive integer" >&2
    exit 2
fi
if [[ ! "${maximum_context_tokens}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: MAX_CONTEXT_TOKENS must be a positive integer" >&2
    exit 2
fi
if [[ ! "${logit_trace_top_k}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: LOGIT_TRACE_TOP_K must be a positive integer" >&2
    exit 2
fi
if [[ ! "${host_memory_limit_kib}" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: HOST_MEMORY_LIMIT_KIB must be a positive integer" >&2
    exit 2
fi

if [[ "${correctness_run}" == 1 ]]; then
    default_result_dir="${repo_root}/results/deepseek-v4-device-moe-correctness"
    run_mode=correctness
else
    default_result_dir="${repo_root}/results/deepseek-v4-device-moe-ab"
    run_mode=performance
fi
result_dir=${RESULT_DIR:-"${default_result_dir}"}

for command in awk find findmnt free git jq lscpu lsblk nvidia-smi \
               readlink sha256sum sort uname vmstat; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "error: required command is unavailable: ${command}" >&2
        exit 1
    fi
done
if [[ ! -x /usr/bin/time ]]; then
    echo "error: /usr/bin/time is required" >&2
    exit 1
fi
if [[ ! -x "${reference_binary}" ]]; then
    echo "error: reference binary is not executable: ${reference_binary}" >&2
    exit 1
fi
if [[ ! -x "${candidate_binary}" ]]; then
    echo "error: candidate binary is not executable: ${candidate_binary}" >&2
    exit 1
fi
index_path="${model_dir}/model.safetensors.index.json"
if [[ ! -f "${index_path}" ]]; then
    echo "error: DeepSeek model index is missing: ${index_path}" >&2
    exit 1
fi

actual_index_sha256=$(sha256sum "${index_path}" | awk '{print $1}')
if [[ "${actual_index_sha256}" != "${expected_index_sha256}" ]]; then
    echo "error: DeepSeek checkpoint index SHA-256 does not match the pinned revision" >&2
    exit 1
fi

if [[ -d "${result_dir}" ]] &&
   [[ -n $(find "${result_dir}" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    echo "error: RESULT_DIR is not empty; choose a fresh deterministic path: ${result_dir}" >&2
    exit 1
fi
mkdir -p "${result_dir}/runs/reference" \
             "${result_dir}/runs/candidate" \
             "${result_dir}/comparisons"

reference_extra_args=()
candidate_extra_args=()
common_extra_args=()
if [[ -n ${REFERENCE_ARGS:-} ]]; then
    read -r -a reference_extra_args <<<"${REFERENCE_ARGS}"
fi
if [[ -n ${CANDIDATE_ARGS:-} ]]; then
    read -r -a candidate_extra_args <<<"${CANDIDATE_ARGS}"
fi
if [[ -n ${COMMON_ARGS:-} ]]; then
    read -r -a common_extra_args <<<"${COMMON_ARGS}"
fi

ensure_no_deepseek_runner() {
    local proc_link target pid found=0
    for proc_link in /proc/[0-9]*/exe; do
        target=$(readlink "${proc_link}" 2>/dev/null || true)
        case "${target}" in
            */strata-deepseek-run|*/strata-deepseek-run\ \(deleted\))
                pid=${proc_link#/proc/}
                pid=${pid%/exe}
                echo "error: concurrent strata-deepseek-run process found: PID ${pid}" >&2
                ps -o pid,ppid,stat,etime,rss,cmd -p "${pid}" >&2 || true
                found=1
                ;;
        esac
    done
    if ((found != 0)); then
        return 1
    fi
}

capture_system_state() {
    local output=$1
    {
        date --iso-8601=seconds
        uname -a
        free -b
        lscpu
        nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,pstate,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used,memory.free,memory.total \
            --format=csv
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
            --format=csv,noheader || true
    } >"${output}"
}

monitor_pids=()
cleanup_monitors() {
    local pid
    for pid in "${monitor_pids[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    for pid in "${monitor_pids[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done
    monitor_pids=()
}
trap cleanup_monitors EXIT
trap 'exit 130' INT TERM

model_source=$(findmnt -no SOURCE --target "${model_dir}" | head -n 1)
block_name=$(lsblk -no PKNAME "${model_source}" 2>/dev/null | head -n 1 || true)
if [[ -z "${block_name}" ]]; then
    block_name=$(basename "${model_source}")
fi
block_stat="/sys/class/block/${block_name}/stat"
if [[ ! -r "${block_stat}" ]]; then
    block_stat=
fi

read_block_sectors() {
    if [[ -z "${block_stat}" ]]; then
        printf '0 0\n'
        return
    fi
    local read_sectors write_sectors
    read -r _ _ read_sectors _ _ _ write_sectors _ <"${block_stat}"
    printf '%s %s\n' "${read_sectors}" "${write_sectors}"
}

file_sha256() {
    local path=$1
    if [[ -f "${path}" ]]; then
        sha256sum "${path}" | awk '{print $1}'
    else
        printf ''
    fi
}

summarize_gpu_peaks() {
    local dmon_path=$1
    local output_path=$2
    local temporary_path="${output_path}.tsv"
    awk '
        $1 !~ /^#/ && NF >= 17 && $3 ~ /^[0-9]+$/ && $17 ~ /^[0-9]+$/ {
            device = $3
            if (!(device in peak) || $17 > peak[device]) peak[device] = $17
        }
        END {
            for (device in peak) print device "\t" peak[device]
        }
    ' "${dmon_path}" | sort -n -k1,1 >"${temporary_path}"
    jq -Rn '[inputs | split("\t") |
        {device:(.[0] | tonumber), peak_framebuffer_mib:(.[1] | tonumber)}
    ]' <"${temporary_path}" >"${output_path}"
    rm -f "${temporary_path}"
}

write_run_summary() {
    local variant=$1
    local repetition=$2
    local order=$3
    local run_dir=$4
    local runner_exit_code=$5
    local valid=$6
    local generation_path="${run_dir}/generation.json"
    local route_path="${run_dir}/routes.jsonl"
    local time_path="${run_dir}/time.txt"
    local maximum_resident_kib
    local route_sha256 diagnostics_sha256

    maximum_resident_kib=$(awk -F: '
        /Maximum resident set size/ {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
            print $2
        }
    ' "${time_path}" 2>/dev/null | tail -n 1)
    if [[ ! "${maximum_resident_kib}" =~ ^[0-9]+$ ]]; then
        maximum_resident_kib=null
    fi
    route_sha256=$(file_sha256 "${route_path}")

    if [[ "${valid}" == 1 ]]; then
        diagnostics_sha256=$(jq -S -c '.diagnostics // null' \
            "${generation_path}" | sha256sum | awk '{print $1}')
        jq \
            --arg variant "${variant}" \
            --argjson repetition "${repetition}" \
            --argjson order "${order}" \
            --arg run_dir "${run_dir}" \
            --arg route_sha256 "${route_sha256}" \
            --arg diagnostics_sha256 "${diagnostics_sha256}" \
            --argjson runner_exit_code "${runner_exit_code}" \
            --argjson maximum_resident_kib "${maximum_resident_kib}" \
            --slurpfile gpu_peaks "${run_dir}/gpu-vram-peaks.json" \
            --slurpfile physical_io "${run_dir}/physical-io.json" '
            {
                variant:$variant,
                repetition:$repetition,
                order:$order,
                run_dir:$run_dir,
                status:"ok",
                runner_exit_code:$runner_exit_code,
                execution:(.execution // null),
                dspark:(.dspark // null),
                device_moe:(.device_moe // null),
                detailed_timing:(.detailed_timing // null),
                initialization_seconds:(.initialization_seconds // null),
                resident_staging_seconds:(.resident_staging_seconds // null),
                prompt_tokens:(.prompt_tokens // null),
                generated_tokens:(.generated_tokens // null),
                decode_steps:(.decode_steps // null),
                prefill_seconds:(.prefill_seconds // null),
                decode_seconds:(.decode_seconds // null),
                decode_steps_per_second:
                    (if ((.decode_seconds // 0) > 0) then
                         (.decode_steps / .decode_seconds)
                     else null end),
                prompt_token_ids:(.prompt_token_ids // []),
                generated_token_ids:(.generated_token_ids // []),
                route_sha256:$route_sha256,
                diagnostics_sha256:$diagnostics_sha256,
                diagnostic_trace:{
                    logit_enabled:(.diagnostics.logits.enabled // false),
                    logit_top_k:(.diagnostics.logits.top_k // null),
                    logit_forward_count:
                        (.diagnostics.logits.aggregate.forward_count // null),
                    logit_non_finite_count:
                        (.diagnostics.logits.aggregate.non_finite_count // null),
                    logit_trace_hash:
                        (.diagnostics.logits.aggregate.trace_hash // null),
                    layer_hash_enabled:
                        (.diagnostics.layer_hidden_hashes.enabled // false),
                    layer_hash_trace_hash:
                        (.diagnostics.layer_hidden_hashes.aggregate.trace_hash // null)
                },
                checkpoint_bytes:{
                    generation:(.generation_checkpoint_read_bytes // null),
                    prefill:(.phases.prefill.checkpoint_read_bytes // null),
                    decode:(.phases.decode.checkpoint_read_bytes //
                            .decode_checkpoint_read_bytes // null)
                },
                decode_cuda:{
                    weight_h2d_bytes:
                        (.phases.decode.cuda.weight_h2d_bytes // null),
                    activation_h2d_bytes:
                        (.phases.decode.cuda.activation_h2d_bytes // null),
                    activation_d2h_bytes:
                        (.phases.decode.cuda.activation_d2h_bytes // null),
                    matmul_calls:(.phases.decode.cuda.matmul_calls // null),
                    synchronization_calls:
                        (.phases.decode.cuda.synchronization_calls // null),
                    synchronization_seconds:
                        (.phases.decode.cuda.critical_path_synchronization_seconds // null),
                    upload_wait_seconds:
                        (.phases.decode.cuda.critical_path_upload_wait_seconds // null),
                    kernel_seconds:
                        (.phases.decode.cuda.critical_path_kernel_seconds // null),
                    deepseek_moe_calls:
                        (.phases.decode.cuda.deepseek_moe_calls // null),
                    deepseek_moe_kernel_launches:
                        (.phases.decode.cuda.deepseek_moe_kernel_launches // null),
                    deepseek_moe_h2d_transfers:
                        (.phases.decode.cuda.deepseek_moe_h2d_transfers // null),
                    deepseek_moe_d2h_transfers:
                        (.phases.decode.cuda.deepseek_moe_d2h_transfers // null),
                    deepseek_moe_h2d_bytes:
                        (.phases.decode.cuda.deepseek_moe_h2d_bytes // null),
                    deepseek_moe_d2h_bytes:
                        (.phases.decode.cuda.deepseek_moe_d2h_bytes // null),
                    deepseek_moe_h2d_seconds:
                        (.phases.decode.cuda.maximum_device_deepseek_moe_h2d_seconds // null),
                    deepseek_moe_kernel_seconds:
                        (.phases.decode.cuda.maximum_device_deepseek_moe_kernel_seconds // null),
                    deepseek_moe_d2h_seconds:
                        (.phases.decode.cuda.maximum_device_deepseek_moe_d2h_seconds // null),
                    deepseek_moe_seconds:
                        (.phases.decode.cuda.maximum_device_deepseek_moe_seconds // null),
                    devices:(.phases.decode.cuda.devices // [])
                },
                decode_cache:(.phases.decode.cache // {}),
                prefill_cache:(.phases.prefill.cache // {}),
                decode_device_moe_runtime:
                    (.phases.decode.device_moe_runtime // {}),
                memory_plan:(.memory_plan // {}),
                maximum_resident_kib:$maximum_resident_kib,
                per_gpu_vram_peaks:$gpu_peaks[0],
                physical_io:$physical_io[0]
            }
        ' "${generation_path}" >"${run_dir}/run-summary.json"
    else
        jq -n \
            --arg variant "${variant}" \
            --argjson repetition "${repetition}" \
            --argjson order "${order}" \
            --arg run_dir "${run_dir}" \
            --arg route_sha256 "${route_sha256}" \
            --argjson runner_exit_code "${runner_exit_code}" \
            --argjson maximum_resident_kib "${maximum_resident_kib}" \
            --slurpfile gpu_peaks "${run_dir}/gpu-vram-peaks.json" \
            --slurpfile physical_io "${run_dir}/physical-io.json" '
            {
                variant:$variant,
                repetition:$repetition,
                order:$order,
                run_dir:$run_dir,
                status:"failed",
                runner_exit_code:$runner_exit_code,
                decode_steps:null,
                decode_seconds:null,
                decode_steps_per_second:null,
                route_sha256:$route_sha256,
                maximum_resident_kib:$maximum_resident_kib,
                per_gpu_vram_peaks:$gpu_peaks[0],
                physical_io:$physical_io[0]
            }
        ' >"${run_dir}/run-summary.json"
    fi
}

run_one() {
    local variant=$1
    local repetition=$2
    local order=$3
    local suffix run_dir binary runner_exit_code valid=1
    local initial_read_sectors initial_write_sectors
    local final_read_sectors final_write_sectors
    local physical_read_bytes physical_write_bytes
    local -a args

    suffix=$(printf '%02d' "${repetition}")
    run_dir="${result_dir}/runs/${variant}/run-${suffix}"
    mkdir -p "${run_dir}"
    if ! ensure_no_deepseek_runner; then
        return 1
    fi

    if [[ "${variant}" == reference ]]; then
        binary=${reference_binary}
    else
        binary=${candidate_binary}
    fi
    args=(
        --model "${model_dir}"
        --devices "${devices}"
        --host-memory "${host_memory}"
        --vram-fraction "${vram_fraction}"
        --max-context "${maximum_context_tokens}"
        --max-new "${maximum_new_tokens}"
        --prompt "${prompt}"
        --route-trace "${run_dir}/routes.jsonl"
        --detailed-timing
        --json
    )
    args+=("${common_extra_args[@]}")
    if [[ "${variant}" == candidate ]]; then
        args+=(--device-moe)
        args+=("${candidate_extra_args[@]}")
    else
        args+=("${reference_extra_args[@]}")
    fi
    if [[ "${correctness_run}" == 1 ]]; then
        args+=(--logit-trace --logit-trace-top-k "${logit_trace_top_k}" \
               --layer-hash-trace)
    fi

    {
        printf '%q ' "${binary}" "${args[@]}"
        printf '\n'
    } >"${run_dir}/command.txt"
    capture_system_state "${run_dir}/system-before.txt"
    read -r initial_read_sectors initial_write_sectors < <(read_block_sectors)

    nvidia-smi dmon -s pucvmet -d 1 -o DT \
        >"${run_dir}/nvidia-dmon.txt" 2>&1 &
    monitor_pids+=("$!")
    vmstat 1 >"${run_dir}/vmstat.txt" 2>&1 &
    monitor_pids+=("$!")

    set +e
    /usr/bin/time -v -o "${run_dir}/time.txt" \
        "${binary}" "${args[@]}" \
        >"${run_dir}/generation.json" \
        2>"${run_dir}/generation.log"
    runner_exit_code=$?
    set -e

    cleanup_monitors
    read -r final_read_sectors final_write_sectors < <(read_block_sectors)
    physical_read_bytes=$(((final_read_sectors - initial_read_sectors) * 512))
    physical_write_bytes=$(((final_write_sectors - initial_write_sectors) * 512))
    jq -n \
        --arg block_device "${block_name}" \
        --argjson physical_read_bytes "${physical_read_bytes}" \
        --argjson physical_write_bytes "${physical_write_bytes}" \
        '{block_device:$block_device,
          physical_read_bytes:$physical_read_bytes,
          physical_write_bytes:$physical_write_bytes}' \
        >"${run_dir}/physical-io.json"
    summarize_gpu_peaks "${run_dir}/nvidia-dmon.txt" \
        "${run_dir}/gpu-vram-peaks.json"
    capture_system_state "${run_dir}/system-after.txt"
    printf '%s\n' "${runner_exit_code}" >"${run_dir}/exit-status.txt"

    if ((runner_exit_code != 0)); then
        valid=0
    elif ! jq -e --arg variant "${variant}" '
        type == "object" and
        .execution == "exact_base_autoregressive" and
        .dspark == "disabled" and
        (.device_moe == ($variant == "candidate")) and
        .detailed_timing == true and
        ((.decode_seconds // 0) > 0) and
        ((.decode_steps // 0) > 0) and
        (.phases.prefill.cache.lease_acquires ==
         .phases.prefill.cache.lease_releases) and
        (.phases.decode.cache.lease_acquires ==
         .phases.decode.cache.lease_releases) and
        ([.phases.prefill.cache.active_leases[],
          .phases.decode.cache.active_leases[]] | all(. == 0)) and
        ([.phases.prefill.cache.leased_bytes[],
          .phases.decode.cache.leased_bytes[]] | all(. == 0)) and
        (if $variant == "candidate" then
             .phases.decode.cache.lease_acquires > 0 and
             .phases.decode.device_moe_runtime.batches ==
                 (.decode_steps * 43) and
             .phases.decode.device_moe_runtime.routed_experts ==
                 (.decode_steps * 43 * 6) and
             .phases.decode.device_moe_runtime.shared_experts ==
                 (.decode_steps * 43) and
             .phases.decode.device_moe_runtime.device_commands >=
                 .phases.decode.device_moe_runtime.batches and
             .phases.decode.device_moe_runtime.device_commands <=
                 (.phases.decode.device_moe_runtime.batches * 3) and
             .phases.decode.device_moe_runtime.execution_seconds > 0
         else
             .phases.prefill.cache.lease_acquires == 0 and
             .phases.decode.cache.lease_acquires == 0 and
             .phases.decode.device_moe_runtime.batches == 0
         end)
    ' "${run_dir}/generation.json" >/dev/null; then
        echo "error: invalid or non-exact generation JSON: ${run_dir}" >&2
        valid=0
    elif [[ ! -s "${run_dir}/routes.jsonl" ]] ||
         ! jq -e . "${run_dir}/routes.jsonl" >/dev/null; then
        echo "error: missing or invalid route trace: ${run_dir}" >&2
        valid=0
    elif [[ "${correctness_run}" == 1 ]] &&
         ! jq -e --argjson top_k "${logit_trace_top_k}" '
             .diagnostics.logits.enabled == true and
             .diagnostics.layer_hidden_hashes.enabled == true and
             .diagnostics.logits.top_k == $top_k and
             .diagnostics.logits.aggregate.non_finite_count == 0
         ' "${run_dir}/generation.json" >/dev/null; then
        echo "error: correctness diagnostics are missing or invalid: ${run_dir}" >&2
        valid=0
    fi

    write_run_summary "${variant}" "${repetition}" "${order}" \
        "${run_dir}" "${runner_exit_code}" "${valid}"
    if [[ "${valid}" != 1 ]]; then
        return 1
    fi
}

compare_pair() {
    local repetition=$1
    local suffix reference_dir candidate_dir
    local tokens_equal=false routes_equal=false diagnostics_equal=null
    suffix=$(printf '%02d' "${repetition}")
    reference_dir="${result_dir}/runs/reference/run-${suffix}"
    candidate_dir="${result_dir}/runs/candidate/run-${suffix}"

    if jq -e -n \
        --slurpfile reference "${reference_dir}/generation.json" \
        --slurpfile candidate "${candidate_dir}/generation.json" '
        $reference[0].prompt_token_ids == $candidate[0].prompt_token_ids and
        $reference[0].generated_token_ids == $candidate[0].generated_token_ids
    ' >/dev/null; then
        tokens_equal=true
    fi
    if cmp -s "${reference_dir}/routes.jsonl" \
              "${candidate_dir}/routes.jsonl"; then
        routes_equal=true
    fi
    if [[ "${correctness_run}" == 1 ]]; then
        diagnostics_equal=false
        if jq -e -n \
            --slurpfile reference "${reference_dir}/generation.json" \
            --slurpfile candidate "${candidate_dir}/generation.json" '
            $reference[0].diagnostics == $candidate[0].diagnostics
        ' >/dev/null; then
            diagnostics_equal=true
        fi
    fi

    jq -n \
        --argjson repetition "${repetition}" \
        --argjson tokens_equal "${tokens_equal}" \
        --argjson routes_equal "${routes_equal}" \
        --argjson diagnostics_equal "${diagnostics_equal}" '
        {
            repetition:$repetition,
            tokens_equal:$tokens_equal,
            routes_equal:$routes_equal,
            diagnostics_equal:$diagnostics_equal
        }
    ' >"${result_dir}/comparisons/repetition-${suffix}.json"

    if [[ "${tokens_equal}" != true || "${routes_equal}" != true ]]; then
        echo "error: token or route mismatch in repetition ${repetition}" >&2
        return 1
    fi
    if [[ "${correctness_run}" == 1 &&
          "${diagnostics_equal}" != true ]]; then
        echo "error: diagnostic trace mismatch in repetition ${repetition}" >&2
        return 1
    fi
}

write_final_summary() {
    local -a run_summaries comparison_summaries
    mapfile -t run_summaries < <(
        find "${result_dir}/runs" -name run-summary.json -type f | sort
    )
    mapfile -t comparison_summaries < <(
        find "${result_dir}/comparisons" -name 'repetition-*.json' -type f | sort
    )
    if ((${#run_summaries[@]} == 0)); then
        printf '[]\n' >"${result_dir}/summary-runs.json"
    else
        jq -s 'sort_by(.order)' "${run_summaries[@]}" \
            >"${result_dir}/summary-runs.json"
    fi
    if ((${#comparison_summaries[@]} == 0)); then
        printf '[]\n' >"${result_dir}/summary-comparisons.json"
    else
        jq -s 'sort_by(.repetition)' "${comparison_summaries[@]}" \
            >"${result_dir}/summary-comparisons.json"
    fi

    jq -n \
        --arg schema strata.deepseek_v4.device_moe_ab.v1 \
        --arg mode "${run_mode}" \
        --arg result_dir "${result_dir}" \
        --arg model_index_sha256 "${actual_index_sha256}" \
        --arg git_revision "$(git -C "${repo_root}" rev-parse HEAD)" \
        --argjson repetitions "${repetitions}" \
        --argjson expected_run_count "$((repetitions * 2))" \
        --argjson minimum_decode_steps_per_second \
            "${minimum_decode_steps_per_second}" \
        --argjson host_memory_limit_kib "${host_memory_limit_kib}" \
        --slurpfile runs "${result_dir}/summary-runs.json" \
        --slurpfile comparisons "${result_dir}/summary-comparisons.json" '
        def median:
            sort as $sorted | ($sorted | length) as $count |
            if $count == 0 then null
            elif ($count % 2) == 1 then $sorted[($count / 2) | floor]
            else (($sorted[($count / 2) - 1] + $sorted[$count / 2]) / 2)
            end;
        def variant_summary($name):
            [$runs[0][] | select(.variant == $name)] as $selected |
            {
                run_count:($selected | length),
                decode_steps_per_second:
                    ($selected | map(.decode_steps_per_second)),
                median_decode_steps_per_second:
                    ($selected | map(.decode_steps_per_second) | median),
                decode_checkpoint_bytes:
                    ($selected | map(.checkpoint_bytes.decode)),
                decode_weight_h2d_bytes:
                    ($selected | map(.decode_cuda.weight_h2d_bytes)),
                decode_activation_h2d_bytes:
                    ($selected | map(.decode_cuda.activation_h2d_bytes)),
                decode_activation_d2h_bytes:
                    ($selected | map(.decode_cuda.activation_d2h_bytes)),
                decode_synchronization_calls:
                    ($selected | map(.decode_cuda.synchronization_calls)),
                decode_deepseek_moe_calls:
                    ($selected | map(.decode_cuda.deepseek_moe_calls)),
                decode_deepseek_moe_kernel_launches:
                    ($selected | map(.decode_cuda.deepseek_moe_kernel_launches)),
                decode_deepseek_moe_h2d_bytes:
                    ($selected | map(.decode_cuda.deepseek_moe_h2d_bytes)),
                decode_deepseek_moe_d2h_bytes:
                    ($selected | map(.decode_cuda.deepseek_moe_d2h_bytes)),
                decode_deepseek_moe_seconds:
                    ($selected | map(.decode_cuda.deepseek_moe_seconds)),
                decode_cache:($selected | map(.decode_cache)),
                prefill_cache:($selected | map(.prefill_cache)),
                decode_device_moe_runtime:
                    ($selected | map(.decode_device_moe_runtime)),
                maximum_resident_kib:
                    ($selected | map(.maximum_resident_kib)),
                per_gpu_vram_peaks:
                    ($selected | map(.per_gpu_vram_peaks)),
                physical_read_bytes:
                    ($selected | map(.physical_io.physical_read_bytes)),
                physical_write_bytes:
                    ($selected | map(.physical_io.physical_write_bytes))
            };
        (variant_summary("reference")) as $reference |
        (variant_summary("candidate")) as $candidate |
        {
            schema:$schema,
            mode:$mode,
            result_dir:$result_dir,
            git_revision:$git_revision,
            model_index_sha256:$model_index_sha256,
            repetitions:$repetitions,
            expected_run_count:$expected_run_count,
            completed_run_count:($runs[0] | length),
            execution_order:
                ($runs[0] | map({order,variant,repetition,status})),
            runs:$runs[0],
            comparisons:$comparisons[0],
            reference:$reference,
            candidate:$candidate,
            median_speedup:
                (if (($reference.median_decode_steps_per_second // 0) > 0) then
                     ($candidate.median_decode_steps_per_second /
                      $reference.median_decode_steps_per_second)
                 else null end),
            minimum_decode_steps_per_second:
                $minimum_decode_steps_per_second,
            gates:{
                all_runs_completed:
                    (($runs[0] | length) == $expected_run_count),
                all_runs_succeeded:
                    ([$runs[0][] | .status == "ok"] | all),
                tokens_equal:
                    ([$comparisons[0][] | .tokens_equal] |
                     length == $repetitions and all),
                routes_equal:
                    ([$comparisons[0][] | .routes_equal] |
                     length == $repetitions and all),
                diagnostics_equal:
                    (if $mode == "correctness" then
                         ([$comparisons[0][] | .diagnostics_equal] |
                          length == $repetitions and all)
                     else null end),
                decode_checkpoint_bytes_zero:
                    ([$runs[0][] | .checkpoint_bytes.decode == 0] | all),
                host_memory_within_ceiling:
                    ([$runs[0][] |
                      (.maximum_resident_kib != null and
                       .maximum_resident_kib <= $host_memory_limit_kib)] | all),
                cache_leases_balanced:
                    ([$runs[0][] |
                      (.prefill_cache.lease_acquires ==
                       .prefill_cache.lease_releases) and
                      (.decode_cache.lease_acquires ==
                       .decode_cache.lease_releases) and
                      ([((.prefill_cache.active_leases // []) +
                         (.decode_cache.active_leases // []))[]] |
                       all(. == 0)) and
                      ([((.prefill_cache.leased_bytes // []) +
                         (.decode_cache.leased_bytes // []))[]] |
                       all(. == 0))] | all),
                candidate_median_meets_target:
                    (($candidate.median_decode_steps_per_second // 0) >=
                     $minimum_decode_steps_per_second)
            }
        } |
        .acceptance_pass =
            (.gates.all_runs_completed and
             .gates.all_runs_succeeded and
             .gates.tokens_equal and
             .gates.routes_equal and
             ((.gates.diagnostics_equal == null) or
              .gates.diagnostics_equal) and
             .gates.decode_checkpoint_bytes_zero and
             .gates.host_memory_within_ceiling and
             .gates.cache_leases_balanced and
             .gates.candidate_median_meets_target)
    ' >"${result_dir}/summary.json"
}

ensure_no_deepseek_runner
capture_system_state "${result_dir}/system-before.txt"
{
    date --iso-8601=seconds
    git -C "${repo_root}" status --short --branch
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" log -1 --oneline --decorate
} >"${result_dir}/revision.txt"
sha256sum "${index_path}" >"${result_dir}/model-index.sha256"
sha256sum "${reference_binary}" >"${result_dir}/reference-binary.sha256"
sha256sum "${candidate_binary}" >"${result_dir}/candidate-binary.sha256"
{
    printf 'mode=%q\n' "${run_mode}"
    printf 'result_dir=%q\n' "${result_dir}"
    printf 'model_dir=%q\n' "${model_dir}"
    printf 'reference_binary=%q\n' "${reference_binary}"
    printf 'candidate_binary=%q\n' "${candidate_binary}"
    printf 'repetitions=%q\n' "${repetitions}"
    printf 'maximum_new_tokens=%q\n' "${maximum_new_tokens}"
    printf 'maximum_context_tokens=%q\n' "${maximum_context_tokens}"
    printf 'prompt=%q\n' "${prompt}"
    printf 'devices=%q\n' "${devices}"
    printf 'host_memory=%q\n' "${host_memory}"
    printf 'vram_fraction=%q\n' "${vram_fraction}"
    printf 'correctness_run=%q\n' "${correctness_run}"
    printf 'logit_trace_top_k=%q\n' "${logit_trace_top_k}"
    printf 'minimum_decode_steps_per_second=%q\n' \
        "${minimum_decode_steps_per_second}"
    printf 'tmux_session=%q\n' "${TMUX:-not-running-under-tmux}"
} >"${result_dir}/config.txt"
nvidia-smi -q >"${result_dir}/nvidia-smi-q-before.txt"
nvidia-smi topo -m >"${result_dir}/nvidia-topology.txt"

order=0
matrix_failed=0
for ((repetition = 1; repetition <= repetitions; ++repetition)); do
    if ((repetition % 2 == 1)); then
        variants=(reference candidate)
    else
        variants=(candidate reference)
    fi
    for variant in "${variants[@]}"; do
        ((order += 1))
        if ! run_one "${variant}" "${repetition}" "${order}"; then
            matrix_failed=1
            break 2
        fi
    done
    if ! compare_pair "${repetition}"; then
        matrix_failed=1
        break
    fi
done

cleanup_monitors
capture_system_state "${result_dir}/system-after.txt"
nvidia-smi -q >"${result_dir}/nvidia-smi-q-after.txt"
write_final_summary
cat "${result_dir}/summary.json"

if ((matrix_failed != 0)); then
    exit 1
fi
if [[ "${enforce_acceptance}" == 1 ]] &&
   ! jq -e '.acceptance_pass == true' "${result_dir}/summary.json" >/dev/null; then
    echo "error: device-MoE acceptance gates did not pass" >&2
    exit 1
fi
