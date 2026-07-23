#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/DeepSeek-V4-Flash-DSpark"}
result_dir=${RESULT_DIR:-"${repo_root}/results/deepseek-v4-lightning-indexer-correctness-v2"}
runner=${RUNNER:-"${repo_root}/build/strata-deepseek-run"}
prompt_repetitions=${PROMPT_REPETITIONS:-2050}
maximum_context_tokens=${MAX_CONTEXT_TOKENS:-4096}
kv_device_cache=${KV_DEVICE_CACHE:-256M,256M,256M}

if [[ -n "$(git -C "${repo_root}" status --porcelain)" ]]; then
    echo "error: Lightning Indexer evidence requires a clean frozen revision" >&2
    exit 1
fi
mkdir -p "${result_dir}/reference" "${result_dir}/candidate"
cp --reflink=auto "${runner}" "${result_dir}/strata-deepseek-run"
runner="${result_dir}/strata-deepseek-run"

prompt=
for ((index=0; index<prompt_repetitions; ++index)); do prompt+=' x'; done

run_variant() {
    local name=$1
    local index_flag=$2
    /usr/bin/time -v "${runner}" \
        --model "${model_dir}" --devices 0,1,2 \
        --host-memory 216G --vram-fraction 0.85 \
        --max-context "${maximum_context_tokens}" \
        --kv-device-cache "${kv_device_cache}" \
        --prefill-page-tokens 64 --max-new 2 --prompt "${prompt}" \
        --route-trace "${result_dir}/${name}/routes.jsonl" \
        --logit-trace --layer-hash-trace --detailed-timing --quiet --json \
        "${index_flag}" \
        >"${result_dir}/${name}/generation.json" \
        2>"${result_dir}/${name}/generation.log"
}

{
    date --iso-8601=seconds
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" status --short --branch
    printf 'prompt_repetitions=%s\n' "${prompt_repetitions}"
    printf 'maximum_context_tokens=%s\n' "${maximum_context_tokens}"
    printf 'kv_device_cache=%s\n' "${kv_device_cache}"
    sha256sum "${runner}"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.free,memory.total \
        --format=csv
} >"${result_dir}/system.txt"

run_variant reference --block-kv-cache
run_variant candidate --gpu-lightning-indexer

jq -n \
    --slurpfile reference "${result_dir}/reference/generation.json" \
    --slurpfile candidate "${result_dir}/candidate/generation.json" '
    {
        reference: {
            prompt_tokens: $reference[0].prompt_tokens,
            prefill_seconds: $reference[0].prefill_seconds,
            decode_seconds: $reference[0].decode_seconds,
            graph: $reference[0].phases.prefill.graph
        },
        candidate: {
            prompt_tokens: $candidate[0].prompt_tokens,
            prefill_seconds: $candidate[0].prefill_seconds,
            decode_seconds: $candidate[0].decode_seconds,
            graph: $candidate[0].phases.prefill.graph,
            cuda: $candidate[0].cuda
        },
        gates: {
            generated_tokens_equal:
                ($reference[0].generated_token_ids ==
                 $candidate[0].generated_token_ids),
            index_selections_equal:
                ($reference[0].diagnostics.index_selections ==
                 $candidate[0].diagnostics.index_selections),
            logits_equal:
                ($reference[0].diagnostics.logits ==
                 $candidate[0].diagnostics.logits),
            layer_hashes_equal:
                ($reference[0].diagnostics.layer_hidden_hashes ==
                 $candidate[0].diagnostics.layer_hidden_hashes),
            operation_hashes_equal:
                ($reference[0].diagnostics.operation_hashes ==
                 $candidate[0].diagnostics.operation_hashes),
            index_work_equal:
                (([$reference[0].phases.prefill.graph.attention_index_queries,
                   $reference[0].phases.prefill.graph.attention_index_candidates,
                   $reference[0].phases.prefill.graph.attention_index_selected]) ==
                 ([$candidate[0].phases.prefill.graph.attention_index_queries,
                   $candidate[0].phases.prefill.graph.attention_index_candidates,
                   $candidate[0].phases.prefill.graph.attention_index_selected])),
            required_paths_exercised:
                ($reference[0].phases.prefill.graph.attention_index_scalar_dispatches > 0 and
                 $candidate[0].phases.prefill.graph.attention_index_cuda_dispatches > 0 and
                 $candidate[0].cuda.lightning_index_calls > 0),
            finite_logits:
                ($reference[0].diagnostics.logits.aggregate.non_finite_count == 0 and
                 $candidate[0].diagnostics.logits.aggregate.non_finite_count == 0),
            zero_decode_checkpoint_reads:
                ($reference[0].decode_checkpoint_read_bytes == 0 and
                 $candidate[0].decode_checkpoint_read_bytes == 0)
        }
    }
    | .acceptance_pass = ([.gates[]] | all)
    ' >"${result_dir}/summary.json"

cmp "${result_dir}/reference/routes.jsonl" \
    "${result_dir}/candidate/routes.jsonl"
jq -e '.acceptance_pass == true' "${result_dir}/summary.json" >/dev/null
