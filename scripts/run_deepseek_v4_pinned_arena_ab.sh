#!/usr/bin/env bash
set -euo pipefail

# A/B for page-locking the resident weight arena, interleaved, median of three.
#
# Contract (see docs/experiments for the recorded result):
#   Bottleneck measured first: a 576 ms decode step at 3,565-token context is
#   255 ms cold expert acquisition, 198 ms attention, 43 ms mHC, 66 ms MoE
#   compute. Only 10.5 ms/step of the acquisition is CUDA; the rest is the
#   driver's pageable staging copy out of a cold 147 GB mapping.
#   Target term: that staging copy, which is the largest single term.
#   Sign on other resources: removes host memcpy work, adds no compute, no
#   VRAM, and a one-time registration cost at load. Nothing is inflated.
#   Correctness: pinning moves no byte, so output must be bit-identical.
#   Rollback: median decode not above baseline beyond run variance, or any
#   output byte changing.

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/dsv4f"}
result_dir=${RESULT_DIR:-"${repo_root}/results/deepseek-v4-pinned-arena-ab"}
runner=${RUNNER:-"${repo_root}/build/strata-deepseek-run"}
maximum_new_tokens=${MAX_NEW_TOKENS:-128}
# Prefill, not model load, is what makes a DeepSeek arm expensive: a 3,565-token
# prompt prefills for 38.5 minutes against 51 seconds of initialization and 73
# seconds of decode. Decode throughput is what is under test, so the prompt is
# sized to the shortest that still exercises the sliding window and leaves the
# expert cache cold enough for demand loads to dominate.
maximum_context_tokens=${MAX_CONTEXT_TOKENS:-4096}
kv_device_cache=${KV_DEVICE_CACHE:-256M,256M,256M}
repetitions=${REPETITIONS:-3}
prompt_sentences=${PROMPT_SENTENCES:-34}

if [[ -n "$(git -C "${repo_root}" status --porcelain)" ]]; then
    echo "error: A/B evidence requires a clean frozen revision" >&2
    exit 1
fi

mkdir -p "${result_dir}"
cp --reflink=auto "${runner}" "${result_dir}/strata-deepseek-run"
runner="${result_dir}/strata-deepseek-run"

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
    local index=0 text=""
    while ((index < prompt_sentences)); do
        text+="${sentences[index % ${#sentences[@]}]} "
        ((++index))
    done
    printf '%s' "${text}"
}
prompt=$(build_prompt)

{
    date --iso-8601=seconds
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" status --short --branch
    printf 'repetitions=%s\nprompt_sentences=%s\nprompt_bytes=%s\n' \
        "${repetitions}" "${prompt_sentences}" "${#prompt}"
    sha256sum "${runner}"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.free,memory.total --format=csv
} >"${result_dir}/system.txt"

run_case() {
    local arm=$1 repetition=$2
    local name="${arm}/run-$(printf '%02d' "${repetition}")"
    mkdir -p "${result_dir}/${name}"
    local -a pin=()
    [[ "${arm}" == "pinned" ]] && pin=(--pin-resident-arena)
    /usr/bin/time -v "${runner}" \
        --model "${model_dir}" --devices 0,1,2 \
        --host-memory 216G --vram-fraction 0.85 \
        --max-context "${maximum_context_tokens}" \
        --kv-device-cache "${kv_device_cache}" \
        --prefill-page-tokens 64 \
        --max-new "${maximum_new_tokens}" --prompt "${prompt}" \
        --route-trace "${result_dir}/${name}/routes.jsonl" \
        --logit-trace --layer-hash-trace --detailed-timing --quiet --json \
        "${pin[@]}" \
        >"${result_dir}/${name}/generation.json" \
        2>"${result_dir}/${name}/generation.log"
}

# Interleaved, so thermal and page-cache drift hit both arms equally.
for ((repetition = 1; repetition <= repetitions; ++repetition)); do
    run_case reference "${repetition}"
    run_case pinned "${repetition}"
done

median() { jq -s 'sort | .[(length - 1) / 2 | floor]'; }
metric() {  # arm, jq expression
    for ((repetition = 1; repetition <= repetitions; ++repetition)); do
        jq "$2" "${result_dir}/$1/run-$(printf '%02d' "${repetition}")/generation.json"
    done | median
}

jq -n \
  --argjson ref_rate "$(metric reference '.decode_steps / .decode_seconds')" \
  --argjson pin_rate "$(metric pinned '.decode_steps / .decode_seconds')" \
  --argjson ref_wait "$(metric reference '.phases.decode.cache.demand_wait_seconds')" \
  --argjson pin_wait "$(metric pinned '.phases.decode.cache.demand_wait_seconds')" \
  --argjson ref_dec  "$(metric reference '.decode_seconds')" \
  --argjson pin_dec  "$(metric pinned '.decode_seconds')" \
  --slurpfile r "${result_dir}/reference/run-01/generation.json" \
  --slurpfile p "${result_dir}/pinned/run-01/generation.json" '
  {
    median_decode_steps_per_second: {reference: $ref_rate, pinned: $pin_rate},
    speedup: ($pin_rate / $ref_rate),
    median_demand_wait_seconds: {reference: $ref_wait, pinned: $pin_wait},
    median_decode_seconds: {reference: $ref_dec, pinned: $pin_dec},
    pin_seconds: $p[0].resident_pin_seconds,
    gates: {
      pinning_actually_happened: ($p[0].resident_arena_pinned == true and
                                 $r[0].resident_arena_pinned == false),
      generated_tokens_equal: ($r[0].generated_token_ids == $p[0].generated_token_ids),
      logits_equal: ($r[0].diagnostics.logits == $p[0].diagnostics.logits),
      layer_hashes_equal: ($r[0].diagnostics.layer_hidden_hashes ==
                           $p[0].diagnostics.layer_hidden_hashes),
      operation_hashes_equal: ($r[0].diagnostics.operation_hashes ==
                               $p[0].diagnostics.operation_hashes),
      zero_decode_checkpoint_reads: ($r[0].decode_checkpoint_read_bytes == 0 and
                                     $p[0].decode_checkpoint_read_bytes == 0),
      faster: ($pin_rate > $ref_rate)
    }
  } | .acceptance_pass = ([.gates[]] | all)' >"${result_dir}/summary.json"

cmp "${result_dir}/reference/run-01/routes.jsonl" \
    "${result_dir}/pinned/run-01/routes.jsonl"
cat "${result_dir}/summary.json"
jq -e '.acceptance_pass == true' "${result_dir}/summary.json" >/dev/null
