#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/DeepSeek-V4-Flash-DSpark"}
runner=${RUNNER:-"${repo_root}/build/strata-deepseek-run"}
result_dir=${RESULT_DIR:-"${repo_root}/results/deepseek-v4-long-context-smoke"}
prompt_repetitions=${PROMPT_REPETITIONS:-2050}
maximum_context_tokens=${MAX_CONTEXT_TOKENS:-32768}
maximum_new_tokens=${MAX_NEW_TOKENS:-2}

if [[ -n "$(git -C "${repo_root}" status --porcelain)" ]]; then
    echo "error: long-context evidence requires a clean frozen revision" >&2
    exit 1
fi

mkdir -p "${result_dir}"
prompt=
for ((index=0; index<prompt_repetitions; ++index)); do
    prompt+=' x'
done

{
    date --iso-8601=seconds
    git -C "${repo_root}" rev-parse HEAD
    printf 'runner=%s\n' "${runner}"
    printf 'prompt_repetitions=%s\n' "${prompt_repetitions}"
    printf 'maximum_context_tokens=%s\n' "${maximum_context_tokens}"
    printf 'maximum_new_tokens=%s\n' "${maximum_new_tokens}"
} >"${result_dir}/run.txt"
git -C "${repo_root}" status --short --branch >"${result_dir}/revision.txt"

/usr/bin/time -v "${runner}" \
    --model "${model_dir}" \
    --devices 0,1,2 \
    --host-memory 216G \
    --vram-fraction 0.85 \
    --max-context "${maximum_context_tokens}" \
    --max-new "${maximum_new_tokens}" \
    --prompt "${prompt}" \
    --logit-trace \
    --layer-hash-trace \
    --detailed-timing \
    --json \
    >"${result_dir}/generation.json" \
    2>"${result_dir}/generation.log"

jq '{
    prompt_tokens,
    decode_steps,
    prefill_seconds,
    generated_token_ids,
    decode_checkpoint_read_bytes,
    generation_checkpoint_read_bytes,
    memory_plan,
    prefill: .phases.prefill,
    decode: .phases.decode,
    diagnostics
}' "${result_dir}/generation.json" >"${result_dir}/summary.json"

jq -e '
    .prompt_tokens > 2048 and
    .decode_steps >= 1 and
    .prefill.graph.attention_index_queries > 0 and
    .decode.graph.attention_index_queries > 0 and
    .decode_checkpoint_read_bytes == 0 and
    .generation_checkpoint_read_bytes == 0 and
    .diagnostics.logits.enabled == true and
    .diagnostics.logits.aggregate.non_finite_count == 0 and
    .diagnostics.layer_hidden_hashes.enabled == true
' "${result_dir}/summary.json" >/dev/null
