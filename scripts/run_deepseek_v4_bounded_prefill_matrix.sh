#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/DeepSeek-V4-Flash-DSpark"}
runner=${RUNNER:-"${repo_root}/build/strata-deepseek-run"}
result_dir=${RESULT_DIR:-"${repo_root}/results/deepseek-v4-bounded-prefill-matrix"}
repetitions=${REPETITIONS:-3}
mkdir -p "${result_dir}/runs"

prompt=""
for ((index=0; index<2050; ++index)); do prompt+=' x'; done

run_variant() {
    local variant=$1
    local page_tokens=$2
    local repetition=$3
    local run_dir="${result_dir}/runs/${variant}/run-${repetition}"
    mkdir -p "${run_dir}"
    /usr/bin/time -v "${runner}" \
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
          decode: .phases.decode.graph,
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
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.free,memory.total --format=csv
} >"${result_dir}/system.txt"
git -C "${repo_root}" diff --binary >"${result_dir}/candidate.diff"

for repetition in $(seq 1 "${repetitions}"); do
    run_variant reference 1 "${repetition}"
    run_variant candidate 64 "${repetition}"
done

jq -s '
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
        projection_rows_equal:
          (([$reference[].prefill.attention_projection_matmul_rows] | unique | length) == 1 and
           ([$candidate[].prefill.attention_projection_matmul_rows] | unique | length) == 1 and
           $reference[0].prefill.attention_projection_matmul_rows ==
             $candidate[0].prefill.attention_projection_matmul_rows),
        projection_calls_reduced:
          (all($candidate[];
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
