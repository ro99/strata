#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/DeepSeek-V4-Flash-DSpark"}
result_dir=${RESULT_DIR:-"${repo_root}/results/deepseek-v4-bounded-prefill-correctness"}
runner=${RUNNER:-"${repo_root}/build/strata-deepseek-run"}
prompt_repetitions=${PROMPT_REPETITIONS:-0}
maximum_context_tokens=${MAX_CONTEXT_TOKENS:-2048}
mkdir -p "${result_dir}/oracle" "${result_dir}/paged"

prompt=""
if [[ "${prompt_repetitions}" -gt 0 ]]; then
    for ((repetition=0; repetition<prompt_repetitions; ++repetition)); do
        prompt+=" x"
    done
else
    for repetition in $(seq 1 16); do
        prompt+="Bounded prefill must preserve every exact causal transition ${repetition}. "
    done
fi

run_variant() {
    local name=$1
    local page_tokens=$2
    /usr/bin/time -v "${runner}" \
        --model "${model_dir}" \
        --devices 0,1,2 \
        --host-memory 216G \
        --vram-fraction 0.85 \
        --max-context "${maximum_context_tokens}" \
        --prefill-page-tokens "${page_tokens}" \
        --max-new 2 \
        --prompt "${prompt}" \
        --route-trace "${result_dir}/${name}/routes.jsonl" \
        --logit-trace \
        --layer-hash-trace \
        --detailed-timing \
        --quiet \
        --json \
        >"${result_dir}/${name}/generation.json" \
        2>"${result_dir}/${name}/generation.log"
}

{
    date --iso-8601=seconds
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" status --short --branch
    printf 'prompt_repetitions=%s\n' "${prompt_repetitions}"
    printf 'maximum_context_tokens=%s\n' "${maximum_context_tokens}"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.free,memory.total \
        --format=csv
} >"${result_dir}/system.txt"
git -C "${repo_root}" diff --binary >"${result_dir}/candidate.diff"

run_variant oracle 1
run_variant paged 64

jq -n \
    --argjson require_index "$([[ "${prompt_repetitions}" -gt 0 ]] && echo true || echo false)" \
    --slurpfile oracle "${result_dir}/oracle/generation.json" \
    --slurpfile paged "${result_dir}/paged/generation.json" '
    {
        oracle: {
            prompt_tokens: $oracle[0].prompt_tokens,
            prefill_seconds: $oracle[0].prefill_seconds,
            graph: $oracle[0].phases.prefill.graph
        },
        paged: {
            prompt_tokens: $paged[0].prompt_tokens,
            prefill_seconds: $paged[0].prefill_seconds,
            graph: $paged[0].phases.prefill.graph
        },
        gates: {
            generated_tokens_equal:
                ($oracle[0].generated_token_ids == $paged[0].generated_token_ids),
            logits_equal:
                ($oracle[0].diagnostics.logits == $paged[0].diagnostics.logits),
            layer_hashes_equal:
                ($oracle[0].diagnostics.layer_hidden_hashes ==
                 $paged[0].diagnostics.layer_hidden_hashes),
            operation_hashes_equal:
                ($oracle[0].diagnostics.operation_hashes ==
                 $paged[0].diagnostics.operation_hashes),
            finite_values:
                ($oracle[0].diagnostics.logits.aggregate.non_finite_count == 0 and
                 $paged[0].diagnostics.logits.aggregate.non_finite_count == 0 and
                 $oracle[0].diagnostics.logits.aggregate.finite_count ==
                    $oracle[0].diagnostics.logits.aggregate.value_count and
                 $paged[0].diagnostics.logits.aggregate.finite_count ==
                    $paged[0].diagnostics.logits.aggregate.value_count),
            zero_prefill_checkpoint_reads:
                ($oracle[0].phases.prefill.checkpoint_read_bytes == 0 and
                 $paged[0].phases.prefill.checkpoint_read_bytes == 0),
            zero_decode_checkpoint_reads:
                ($oracle[0].decode_checkpoint_read_bytes == 0 and
                 $paged[0].decode_checkpoint_read_bytes == 0),
            index_trace_equal:
                (([$oracle[0].phases.prefill.graph.attention_index_queries,
                   $oracle[0].phases.prefill.graph.attention_index_candidates,
                   $oracle[0].phases.prefill.graph.attention_index_selected]) ==
                 ([$paged[0].phases.prefill.graph.attention_index_queries,
                   $paged[0].phases.prefill.graph.attention_index_candidates,
                   $paged[0].phases.prefill.graph.attention_index_selected])),
            required_index_exercised:
                (($require_index | not) or
                 ($oracle[0].phases.prefill.graph.attention_index_queries > 0 and
                  $paged[0].phases.prefill.graph.attention_index_queries > 0)),
            attention_projection_rows_equal:
                ($oracle[0].phases.prefill.graph.attention_projection_matmul_rows ==
                 $paged[0].phases.prefill.graph.attention_projection_matmul_rows),
            attention_projection_calls_reduced:
                ($paged[0].phases.prefill.graph.attention_projection_matmul_calls <
                 $oracle[0].phases.prefill.graph.attention_projection_matmul_calls)
        }
    }
    | .acceptance_pass = ([.gates[]] | all)
    ' >"${result_dir}/summary.json"

cmp "${result_dir}/oracle/routes.jsonl" "${result_dir}/paged/routes.jsonl"
jq -e '.acceptance_pass == true' "${result_dir}/summary.json" >/dev/null
