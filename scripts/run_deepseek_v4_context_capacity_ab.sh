#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runner=${RUNNER:-"${repo_root}/build/strata-deepseek-run"}
result_root=${RESULT_ROOT:-"${repo_root}/results/deepseek-v4-context-capacity-ab"}
large_context=${LARGE_CONTEXT_TOKENS:-200000}
repetitions=${REPETITIONS:-3}

if [[ -n "$(git -C "${repo_root}" status --porcelain)" ]]; then
    echo "error: context-capacity evidence requires a clean frozen revision" >&2
    exit 1
fi

mkdir -p "${result_root}"
git -C "${repo_root}" rev-parse HEAD >"${result_root}/revision.txt"
: >"${result_root}/order.txt"
for repetition in $(seq 1 "${repetitions}"); do
    for variant in context-2048 context-large; do
        maximum_context=2048
        if [[ "${variant}" == context-large ]]; then
            maximum_context=${large_context}
        fi
        result_dir="${result_root}/${variant}-${repetition}"
        printf '%s %s %s\n' "${repetition}" "${variant}" "${maximum_context}" \
            | tee -a "${result_root}/order.txt"
        CAPTURE_TELEMETRY=0 RUNNER="${runner}" \
            MAX_CONTEXT_TOKENS="${maximum_context}" RESULT_DIR="${result_dir}" \
            "${repo_root}/scripts/run_deepseek_v4_observability.sh"
    done
done

jq -s '
    def median:
        sort as $values |
        ($values | length) as $count |
        if $count % 2 == 1 then $values[$count / 2 | floor]
        else (($values[$count / 2 - 1] + $values[$count / 2]) / 2) end;
    . as $runs |
    {
        revision: $revision,
        repetitions: $repetitions,
        contexts: {
            context_2048: {
                median_initialization_seconds:
                    ([$runs[] | select(.memory_plan.maximum_context_tokens == 2048) |
                      .initialization_seconds] | median),
                median_prefill_seconds:
                    ([$runs[] | select(.memory_plan.maximum_context_tokens == 2048) |
                      .prefill_seconds] | median),
                median_decode_seconds:
                    ([$runs[] | select(.memory_plan.maximum_context_tokens == 2048) |
                      .decode_seconds] | median),
                median_decode_tokens_per_second:
                    ([$runs[] | select(.memory_plan.maximum_context_tokens == 2048) |
                      (.decode_steps / .decode_seconds)] | median),
                required_host_bytes:
                    ([$runs[] | select(.memory_plan.maximum_context_tokens == 2048) |
                      .memory_plan.required_host_bytes][0])
            },
            context_large: {
                tokens: $large_context,
                median_initialization_seconds:
                    ([$runs[] | select(.memory_plan.maximum_context_tokens == $large_context) |
                      .initialization_seconds] | median),
                median_prefill_seconds:
                    ([$runs[] | select(.memory_plan.maximum_context_tokens == $large_context) |
                      .prefill_seconds] | median),
                median_decode_seconds:
                    ([$runs[] | select(.memory_plan.maximum_context_tokens == $large_context) |
                      .decode_seconds] | median),
                median_decode_tokens_per_second:
                    ([$runs[] | select(.memory_plan.maximum_context_tokens == $large_context) |
                      (.decode_steps / .decode_seconds)] | median),
                required_host_bytes:
                    ([$runs[] | select(.memory_plan.maximum_context_tokens == $large_context) |
                      .memory_plan.required_host_bytes][0])
            }
        },
        gates: {
            run_count: ($runs | length),
            generated_tokens_equal:
                ([$runs[].generated_token_ids] | unique | length == 1),
            zero_generation_checkpoint_reads:
                ([$runs[] | .generation_checkpoint_read_bytes == 0] | all),
            zero_decode_checkpoint_reads:
                ([$runs[] | .decode_checkpoint_read_bytes == 0] | all),
            short_request_skips_index:
                ([$runs[] | .phases.decode.graph.attention_index_queries == 0] | all)
        }
    }
' --arg revision "$(git -C "${repo_root}" rev-parse HEAD)" \
  --argjson repetitions "${repetitions}" \
  --argjson large_context "${large_context}" \
  "${result_root}"/*/generation.json >"${result_root}/summary.json"

jq -e --argjson expected_runs "$((repetitions * 2))" '
    .gates.run_count == $expected_runs and
    .gates.generated_tokens_equal and
    .gates.zero_generation_checkpoint_reads and
    .gates.zero_decode_checkpoint_reads and
    .gates.short_request_skips_index
' "${result_root}/summary.json" >/dev/null
cat "${result_root}/summary.json"
