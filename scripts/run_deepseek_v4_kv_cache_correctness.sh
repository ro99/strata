#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/DeepSeek-V4-Flash-DSpark"}
runner=${RUNNER:-"${repo_root}/build/strata-deepseek-run"}
result_dir=${RESULT_DIR:-"${repo_root}/results/deepseek-v4-kv-cache-correctness"}
prompt_repetitions=${PROMPT_REPETITIONS:-16}
compact_prompt=${COMPACT_PROMPT:-0}
maximum_context_tokens=${MAX_CONTEXT_TOKENS:-4096}
require_index=${REQUIRE_INDEX:-0}
mkdir -p "${result_dir}/scalar" "${result_dir}/block"
cp --reflink=auto "${runner}" "${result_dir}/strata-deepseek-run"
runner="${result_dir}/strata-deepseek-run"

prompt=
for ((index=0; index<prompt_repetitions; ++index)); do
    if [[ "${compact_prompt}" == 1 ]]; then
        prompt+=' x'
    else
        prompt+="Exact block cache boundaries must preserve every attention row ${index}. "
    fi
done

run_variant() {
    local name=$1
    local cache_flag=$2
    /usr/bin/time -v "${runner}" \
        --model "${model_dir}" \
        --devices 0,1,2 \
        --host-memory 216G \
        --vram-fraction 0.85 \
        --max-context "${maximum_context_tokens}" \
        --max-new 2 \
        --prompt "${prompt}" \
        --route-trace "${result_dir}/${name}/routes.jsonl" \
        --logit-trace \
        --layer-hash-trace \
        --detailed-timing \
        --quiet \
        --json \
        "${cache_flag}" \
        >"${result_dir}/${name}/generation.json" \
        2>"${result_dir}/${name}/generation.log"
}

{
    date --iso-8601=seconds
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" status --short --branch
    printf 'prompt_repetitions=%s\n' "${prompt_repetitions}"
    printf 'compact_prompt=%s\n' "${compact_prompt}"
    printf 'maximum_context_tokens=%s\n' "${maximum_context_tokens}"
    printf 'require_index=%s\n' "${require_index}"
    sha256sum "${runner}"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.free,memory.total \
        --format=csv
} >"${result_dir}/system.txt"
git -C "${repo_root}" diff --binary >"${result_dir}/candidate.diff"

run_variant scalar --scalar-kv-cache
run_variant block --block-kv-cache

jq -n \
    --argjson require_index "$([[ "${require_index}" == 1 ]] && echo true || echo false)" \
    --slurpfile scalar "${result_dir}/scalar/generation.json" \
    --slurpfile block "${result_dir}/block/generation.json" '
    {
        scalar: {
            prompt_tokens: $scalar[0].prompt_tokens,
            prefill_seconds: $scalar[0].prefill_seconds,
            decode_seconds: $scalar[0].decode_seconds,
            kv_cache: $scalar[0].kv_cache
        },
        block: {
            prompt_tokens: $block[0].prompt_tokens,
            prefill_seconds: $block[0].prefill_seconds,
            decode_seconds: $block[0].decode_seconds,
            kv_cache: $block[0].kv_cache
        },
        gates: {
            generated_tokens_equal:
                ($scalar[0].generated_token_ids == $block[0].generated_token_ids),
            logits_equal:
                ($scalar[0].diagnostics.logits == $block[0].diagnostics.logits),
            layer_hashes_equal:
                ($scalar[0].diagnostics.layer_hidden_hashes ==
                 $block[0].diagnostics.layer_hidden_hashes),
            operation_hashes_equal:
                ($scalar[0].diagnostics.operation_hashes ==
                 $block[0].diagnostics.operation_hashes),
            routes_equal:
                ($scalar[0].phases.prefill.graph.attention_index_queries ==
                 $block[0].phases.prefill.graph.attention_index_queries),
            required_index_exercised:
                (($require_index | not) or
                 ($scalar[0].phases.prefill.graph.attention_index_queries > 0 and
                  $block[0].phases.prefill.graph.attention_index_queries > 0)),
            zero_checkpoint_reads:
                ($scalar[0].decode_checkpoint_read_bytes == 0 and
                 $block[0].decode_checkpoint_read_bytes == 0),
            bounded_host_cache:
                ($block[0].kv_cache.host_peak_bytes <=
                    $block[0].kv_cache.host_capacity_bytes and
                 $block[0].kv_cache.host_used_bytes <=
                    $block[0].kv_cache.host_capacity_bytes),
            block_cache_exercised:
                ($scalar[0].kv_cache.allocated_blocks == 0 and
                 $block[0].kv_cache.allocated_blocks > 0 and
                 $block[0].kv_cache.gather_bytes > 0 and
                 $block[0].kv_cache.misses == 0)
        }
    }
    | .acceptance_pass = ([.gates[]] | all)
    ' >"${result_dir}/summary.json"

cmp "${result_dir}/scalar/routes.jsonl" "${result_dir}/block/routes.jsonl"
jq -e '.acceptance_pass == true' "${result_dir}/summary.json" >/dev/null
