#!/usr/bin/env bash
set -euo pipefail

# Sweeps the per-device routed-expert VRAM ceiling to manufacture a genuine
# RAM-to-PCIe offload regime on a machine whose weight arena already holds the
# whole DeepSeek V4 working set, and captures the decode-phase route trace the
# shadow-speculative window oracle needs.
#
# Caching is advisory, so every arm must produce byte-identical output. The
# script gates on that before reporting any regime metric.

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/DeepSeek-V4-Flash-DSpark"}
result_dir=${RESULT_DIR:-"${repo_root}/results/deepseek-v4-offload-regime-profile"}
runner=${RUNNER:-"${repo_root}/build/strata-deepseek-run"}
maximum_new_tokens=${MAX_NEW_TOKENS:-128}
maximum_context_tokens=${MAX_CONTEXT_TOKENS:-4096}
kv_device_cache=${KV_DEVICE_CACHE:-256M,256M,256M}
prompt_sentences=${PROMPT_SENTENCES:-240}
# unbounded is the current behaviour; the rest bound the routed-expert set.
arms=${ARMS:-"unbounded 8G 4G 1500M"}

if [[ -n "$(git -C "${repo_root}" status --porcelain)" ]]; then
    echo "error: regime evidence requires a clean frozen revision" >&2
    exit 1
fi

mkdir -p "${result_dir}"
cp --reflink=auto "${runner}" "${result_dir}/strata-deepseek-run"
runner="${result_dir}/strata-deepseek-run"

# A natural-language prompt. The archived ' x' repetition prompt is
# pathologically low-entropy and routes almost every token to the same experts,
# which flatters any locality measurement.
build_prompt() {
    local -a sentences=(
        "The harbour master kept a ledger of every vessel that crossed the bar at dawn."
        "Salt crusted the iron railings and the gulls argued over a torn net."
        "In the archive room a clerk copied tide tables onto onionskin paper."
        "A surveyor measured the drift of the sandbank against last winter's chart."
        "The lighthouse keeper replaced the mantle and trimmed the wick before dusk."
        "Fishermen mended trawl lines while the tide ran out across the flats."
        "An engineer argued that the breakwater needed a deeper foundation."
        "The customs officer weighed a crate of tin and signed the manifest twice."
        "Rain moved in from the west and the barometer fell through the evening."
        "A cartographer redrew the channel where the current had cut a new mouth."
        "The pilot boat waited beyond the buoys for the grain ship to answer."
        "Children counted the bells that marked each watch from the seawall."
    )
    local index=0
    local text=""
    while ((index < prompt_sentences)); do
        text+="${sentences[index % ${#sentences[@]}]} "
        ((++index))
    done
    printf '%s' "${text}"
}

prompt=$(build_prompt)

run_arm() {
    local name=$1
    local ceiling=$2
    mkdir -p "${result_dir}/${name}"
    local -a ceiling_flag=()
    if [[ "${ceiling}" != "unbounded" ]]; then
        ceiling_flag=(--routed-expert-cache "${ceiling}")
    fi
    /usr/bin/time -v "${runner}" \
        --model "${model_dir}" --devices 0,1,2 \
        --host-memory 216G --vram-fraction 0.85 \
        --max-context "${maximum_context_tokens}" \
        --kv-device-cache "${kv_device_cache}" \
        --prefill-page-tokens 64 \
        --max-new "${maximum_new_tokens}" --prompt "${prompt}" \
        --route-trace "${result_dir}/${name}/routes.jsonl" \
        --logit-trace --layer-hash-trace --detailed-timing --quiet --json \
        "${ceiling_flag[@]}" \
        >"${result_dir}/${name}/generation.json" \
        2>"${result_dir}/${name}/generation.log"
}

{
    date --iso-8601=seconds
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" status --short --branch
    printf 'arms=%s\n' "${arms}"
    printf 'maximum_new_tokens=%s\n' "${maximum_new_tokens}"
    printf 'maximum_context_tokens=%s\n' "${maximum_context_tokens}"
    printf 'kv_device_cache=%s\n' "${kv_device_cache}"
    printf 'prompt_sentences=%s\n' "${prompt_sentences}"
    printf 'prompt_bytes=%s\n' "${#prompt}"
    sha256sum "${runner}"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.free,memory.total \
        --format=csv
} >"${result_dir}/system.txt"

names=()
for ceiling in ${arms}; do
    name="cache-${ceiling}"
    run_arm "${name}" "${ceiling}"
    names+=("${name}")
done

# Caching cannot change output: every arm must match the unbounded reference.
reference="${result_dir}/${names[0]}"
for name in "${names[@]:1}"; do
    cmp "${reference}/routes.jsonl" "${result_dir}/${name}/routes.jsonl"
done

jq -s '
    {
        arms: [ .[] | {
            routed_expert_cache_bytes: .routed_expert_cache_bytes,
            prompt_tokens: .prompt_tokens,
            decode_steps: .decode_steps,
            decode_seconds: .decode_seconds,
            decode_steps_per_second:
                (if .decode_seconds > 0 then .decode_steps / .decode_seconds
                 else 0 end),
            prefill_seconds: .prefill_seconds,
            decode_evictions: .phases.decode.cache.evictions,
            decode_hits: .phases.decode.cache.hits,
            decode_misses: .phases.decode.cache.misses,
            decode_demand_h2d_bytes: .phases.decode.cache.demand_h2d_bytes,
            decode_demand_wait_seconds:
                (.phases.decode.cache.demand_wait_nanoseconds / 1000000000),
            cold_acquisition_fraction:
                (if (.phases.decode.cache.hits + .phases.decode.cache.misses) > 0
                 then .phases.decode.cache.misses /
                      (.phases.decode.cache.hits + .phases.decode.cache.misses)
                 else 0 end),
            unpinned_used_bytes: .weight_cache.unpinned_used_bytes,
            unpinned_capacity_bytes: .weight_cache.unpinned_capacity_bytes,
            decode_checkpoint_read_bytes: .decode_checkpoint_read_bytes,
            generated_token_ids: .generated_token_ids
        } ],
        gates: {
            generated_tokens_equal:
                ([ .[].generated_token_ids ] | unique | length) == 1,
            logits_equal:
                ([ .[].diagnostics.logits ] | unique | length) == 1,
            layer_hashes_equal:
                ([ .[].diagnostics.layer_hidden_hashes ] | unique | length) == 1,
            operation_hashes_equal:
                ([ .[].diagnostics.operation_hashes ] | unique | length) == 1,
            zero_decode_checkpoint_reads:
                ([ .[].decode_checkpoint_read_bytes ] | all(. == 0)),
            finite_logits:
                ([ .[].diagnostics.logits.aggregate.non_finite_count ]
                 | all(. == 0)),
            regime_reached:
                ([ .[].phases.decode.cache.evictions ] | any(. > 0))
        }
    }
    | .acceptance_pass = ([.gates[]] | all)
' "${result_dir}"/cache-*/generation.json >"${result_dir}/summary.json"

jq -e '.gates.generated_tokens_equal == true' "${result_dir}/summary.json" >/dev/null
jq -e '.acceptance_pass == true' "${result_dir}/summary.json" >/dev/null
