#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/DeepSeek-V4-Flash-DSpark"}
runner=${RUNNER:-"${repo_root}/build/strata-deepseek-run"}
reference_runner=${REFERENCE_RUNNER:-"${runner}"}
candidate_runner=${CANDIDATE_RUNNER:-"${runner}"}
result_dir=${RESULT_DIR:-"${repo_root}/results/deepseek-v4-bounded-prefill-matrix"}
repetitions=${REPETITIONS:-3}
prompt_tokens=${PROMPT_TOKENS:-2050}
reference_page_tokens=${REFERENCE_PAGE_TOKENS:-1}
candidate_page_tokens=${CANDIDATE_PAGE_TOKENS:-64}
require_projection_reduction=${REQUIRE_PROJECTION_REDUCTION:-true}
mkdir -p "${result_dir}/runs"

prompt=""
for ((index=0; index<prompt_tokens; ++index)); do prompt+=' x'; done

run_variant() {
    local variant=$1
    local page_tokens=$2
    local repetition=$3
    local executable=$4
    local run_dir="${result_dir}/runs/${variant}/run-${repetition}"
    mkdir -p "${run_dir}"
    /usr/bin/time -v "${executable}" \
        --model "${model_dir}" \
        --devices 0,1,2 \
        --host-memory 216G \
        --vram-fraction 0.85 \
        --max-context 32768 \
        --prefill-page-tokens "${page_tokens}" \
        --max-new 2 \
        --prompt "${prompt}" \
        --detailed-timing \
        --quiet \
        --json \
        >"${run_dir}/generation.json" \
        2>"${run_dir}/generation.log"
    local maximum_resident_kib
    maximum_resident_kib=$(awk -F ': ' \
        '/Maximum resident set size \(kbytes\)/ {print $2}' \
        "${run_dir}/generation.log")
    if [[ -z "${maximum_resident_kib}" ]]; then maximum_resident_kib=0; fi
    jq --arg variant "${variant}" \
       --argjson repetition "${repetition}" \
       --argjson maximum_resident_kib "${maximum_resident_kib}" '
        {variant: $variant, repetition: $repetition,
          prompt_tokens, prefill_seconds, decode_seconds,
          maximum_resident_kib: $maximum_resident_kib,
          memory_plan,
          prefill: .phases.prefill.graph,
          prefill_cuda: .phases.prefill.cuda,
          prefill_cache: .phases.prefill.cache,
          prefill_device_moe: (.phases.prefill.device_moe_runtime |
            .unique_routed_experts =
              (.unique_routed_experts // .routed_experts)),
          decode: .phases.decode.graph,
          decode_cuda: .phases.decode.cuda,
          decode_cache: .phases.decode.cache,
          decode_device_moe: (.phases.decode.device_moe_runtime |
            .unique_routed_experts =
              (.unique_routed_experts // .routed_experts)),
          generated_token_ids,
          prefill_checkpoint_read_bytes: .phases.prefill.checkpoint_read_bytes,
          decode_checkpoint_read_bytes,
          generation_checkpoint_read_bytes}' \
        "${run_dir}/generation.json" >"${run_dir}/summary.json"
}

{
    date --iso-8601=seconds
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" status --short --branch
    printf 'reference_runner=%s\n' "${reference_runner}"
    printf 'candidate_runner=%s\n' "${candidate_runner}"
    printf 'prompt_tokens=%s\n' "${prompt_tokens}"
    printf 'reference_page_tokens=%s\n' "${reference_page_tokens}"
    printf 'candidate_page_tokens=%s\n' "${candidate_page_tokens}"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.free,memory.total --format=csv
} >"${result_dir}/system.txt"
git -C "${repo_root}" diff --binary >"${result_dir}/candidate.diff"

for repetition in $(seq 1 "${repetitions}"); do
    run_variant reference "${reference_page_tokens}" "${repetition}" \
        "${reference_runner}"
    run_variant candidate "${candidate_page_tokens}" "${repetition}" \
        "${candidate_runner}"
done

jq -s --argjson require_projection_reduction "${require_projection_reduction}" '
    def median: sort | .[(length / 2) | floor];
    map(select(.variant == "reference")) as $reference |
    map(select(.variant == "candidate")) as $candidate |
    ($reference | sort_by(.prefill_seconds) | .[(length / 2) | floor]) as $reference_median |
    ($candidate | sort_by(.prefill_seconds) | .[(length / 2) | floor]) as $candidate_median |
    {
      repetitions: ($reference | length),
      reference_runs: $reference,
      candidate_runs: $candidate,
      median: {
        reference_prefill_seconds: $reference_median.prefill_seconds,
        candidate_prefill_seconds: $candidate_median.prefill_seconds,
        reference_prefill_tokens_per_second:
          ($reference_median.prompt_tokens / $reference_median.prefill_seconds),
        candidate_prefill_tokens_per_second:
          ($candidate_median.prompt_tokens / $candidate_median.prefill_seconds),
        reference_maximum_resident_kib:
          ([$reference[].maximum_resident_kib] | median),
        candidate_maximum_resident_kib:
          ([$candidate[].maximum_resident_kib] | median),
        prefill_seconds_change:
          (($candidate_median.prefill_seconds /
            $reference_median.prefill_seconds) - 1),
        prefill_tokens_per_second_change:
          (($candidate_median.prompt_tokens / $candidate_median.prefill_seconds) /
           ($reference_median.prompt_tokens / $reference_median.prefill_seconds) - 1),
        attention: {
          reference: {
            query_seconds: ([$reference[].prefill.attention_query_seconds] | median),
            kv_seconds: ([$reference[].prefill.attention_kv_seconds] | median),
            index_seconds: ([$reference[].prefill.attention_index_seconds] | median),
            score_seconds: ([$reference[].prefill.attention_score_seconds] | median),
            output_seconds: ([$reference[].prefill.attention_output_seconds] | median),
            projection_matmul_calls:
              ([$reference[].prefill.attention_projection_matmul_calls] | median),
            projection_matmul_rows:
              ([$reference[].prefill.attention_projection_matmul_rows] | median)
          },
          candidate: {
            query_seconds: ([$candidate[].prefill.attention_query_seconds] | median),
            kv_seconds: ([$candidate[].prefill.attention_kv_seconds] | median),
            index_seconds: ([$candidate[].prefill.attention_index_seconds] | median),
            score_seconds: ([$candidate[].prefill.attention_score_seconds] | median),
            output_seconds: ([$candidate[].prefill.attention_output_seconds] | median),
            projection_matmul_calls:
              ([$candidate[].prefill.attention_projection_matmul_calls] | median),
            projection_matmul_rows:
              ([$candidate[].prefill.attention_projection_matmul_rows] | median)
          }
        },
        moe: {
          reference: {
            expert_rows: ([$reference[].prefill_device_moe.routed_experts] | median),
            unique_experts:
              ([$reference[].prefill_device_moe |
                (.unique_routed_experts // .routed_experts)] | median),
            batches: ([$reference[].prefill_device_moe.batches] | median),
            execution_seconds:
              ([$reference[].prefill_device_moe.execution_seconds] | median),
            preparation_seconds: ([$reference[].prefill.moe_prepare_seconds] | median),
            kernel_launches:
              ([$reference[].prefill_cuda.deepseek_moe_kernel_launches] | median),
            h2d_bytes: ([$reference[].prefill_cuda.deepseek_moe_h2d_bytes] | median),
            d2h_bytes: ([$reference[].prefill_cuda.deepseek_moe_d2h_bytes] | median),
            cache_hits: ([$reference[].prefill_cache.hits] | median),
            cache_misses: ([$reference[].prefill_cache.misses] | median)
          },
          candidate: {
            expert_rows: ([$candidate[].prefill_device_moe.routed_experts] | median),
            unique_experts:
              ([$candidate[].prefill_device_moe |
                (.unique_routed_experts // .routed_experts)] | median),
            batches: ([$candidate[].prefill_device_moe.batches] | median),
            execution_seconds:
              ([$candidate[].prefill_device_moe.execution_seconds] | median),
            preparation_seconds: ([$candidate[].prefill.moe_prepare_seconds] | median),
            kernel_launches:
              ([$candidate[].prefill_cuda.deepseek_moe_kernel_launches] | median),
            h2d_bytes: ([$candidate[].prefill_cuda.deepseek_moe_h2d_bytes] | median),
            d2h_bytes: ([$candidate[].prefill_cuda.deepseek_moe_d2h_bytes] | median),
            cache_hits: ([$candidate[].prefill_cache.hits] | median),
            cache_misses: ([$candidate[].prefill_cache.misses] | median)
          }
        },
        decode: {
          reference: {
            seconds: ([$reference[].decode_seconds] | median),
            expert_rows: ([$reference[].decode_device_moe.routed_experts] | median),
            kernel_launches:
              ([$reference[].decode_cuda.deepseek_moe_kernel_launches] | median),
            synchronization_calls:
              ([$reference[].decode_cuda.synchronization_calls] | median)
          },
          candidate: {
            seconds: ([$candidate[].decode_seconds] | median),
            expert_rows: ([$candidate[].decode_device_moe.routed_experts] | median),
            kernel_launches:
              ([$candidate[].decode_cuda.deepseek_moe_kernel_launches] | median),
            synchronization_calls:
              ([$candidate[].decode_cuda.synchronization_calls] | median)
          }
        },
        cuda: {
          reference: {
            matmul_calls: ([$reference[].prefill_cuda.matmul_calls] | median),
            activation_h2d_bytes:
              ([$reference[].prefill_cuda.activation_h2d_bytes] | median),
            activation_d2h_bytes:
              ([$reference[].prefill_cuda.activation_d2h_bytes] | median),
            synchronization_calls:
              ([$reference[].prefill_cuda.synchronization_calls] | median),
            synchronization_seconds:
              ([$reference[].prefill_cuda.critical_path_synchronization_seconds] | median),
            activation_h2d_seconds:
              ([$reference[].prefill_cuda.critical_path_activation_h2d_seconds] | median),
            activation_d2h_seconds:
              ([$reference[].prefill_cuda.critical_path_activation_d2h_seconds] | median),
            kernel_seconds:
              ([$reference[].prefill_cuda.critical_path_kernel_seconds] | median)
          },
          candidate: {
            matmul_calls: ([$candidate[].prefill_cuda.matmul_calls] | median),
            activation_h2d_bytes:
              ([$candidate[].prefill_cuda.activation_h2d_bytes] | median),
            activation_d2h_bytes:
              ([$candidate[].prefill_cuda.activation_d2h_bytes] | median),
            synchronization_calls:
              ([$candidate[].prefill_cuda.synchronization_calls] | median),
            synchronization_seconds:
              ([$candidate[].prefill_cuda.critical_path_synchronization_seconds] | median),
            activation_h2d_seconds:
              ([$candidate[].prefill_cuda.critical_path_activation_h2d_seconds] | median),
            activation_d2h_seconds:
              ([$candidate[].prefill_cuda.critical_path_activation_d2h_seconds] | median),
            kernel_seconds:
              ([$candidate[].prefill_cuda.critical_path_kernel_seconds] | median)
          }
        }
      },
      gates: {
        prompt_tokens_equal:
          (([$reference[].prompt_tokens] | unique | length) == 1 and
           ([$candidate[].prompt_tokens] | unique | length) == 1 and
           $reference[0].prompt_tokens == $candidate[0].prompt_tokens),
        generated_tokens_equal:
          ([$reference[].generated_token_ids] == [$candidate[].generated_token_ids]),
        expert_rows_equal:
          (([$reference[].prefill_device_moe.routed_experts] | unique | length) == 1 and
           ([$candidate[].prefill_device_moe.routed_experts] | unique | length) == 1 and
           $reference[0].prefill_device_moe.routed_experts ==
             $candidate[0].prefill_device_moe.routed_experts and
           $reference[0].prefill_device_moe.shared_experts ==
             $candidate[0].prefill_device_moe.shared_experts),
        expert_kernel_launches_reduced:
          (all($candidate[];
             .prefill_cuda.deepseek_moe_kernel_launches <
               $reference[0].prefill_cuda.deepseek_moe_kernel_launches)),
        leases_balanced:
          (all($reference[];
             .prefill_cache.lease_acquires == .prefill_cache.lease_releases) and
           all($candidate[];
             .prefill_cache.lease_acquires == .prefill_cache.lease_releases)),
        projection_rows_equal:
          (([$reference[].prefill.attention_projection_matmul_rows] | unique | length) == 1 and
           ([$candidate[].prefill.attention_projection_matmul_rows] | unique | length) == 1 and
           $reference[0].prefill.attention_projection_matmul_rows ==
             $candidate[0].prefill.attention_projection_matmul_rows),
        projection_calls_reduced:
          (($require_projection_reduction | not) or
           all($candidate[];
              .prefill.attention_projection_matmul_calls <
                $reference[0].prefill.attention_projection_matmul_calls)),
        zero_prefill_checkpoint_reads:
          (all($reference[]; .prefill_checkpoint_read_bytes == 0) and
           all($candidate[]; .prefill_checkpoint_read_bytes == 0)),
        zero_decode_checkpoint_reads:
          (all($reference[]; .decode_checkpoint_read_bytes == 0) and
           all($candidate[]; .decode_checkpoint_read_bytes == 0))
      }
    }
    | .acceptance_pass = ([.gates[]] | all)
' "${result_dir}"/runs/*/*/summary.json >"${result_dir}/summary.json"

jq -e '.acceptance_pass == true and .repetitions == 3' \
    "${result_dir}/summary.json" >/dev/null
