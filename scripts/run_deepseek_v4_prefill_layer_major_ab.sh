#!/usr/bin/env bash
set -euo pipefail

# A/B for layer-major prefill tiling, two arms from one binary.
#
# Contract:
#   Bottleneck measured first: a 3,565-token prefill moved 3,366,668,206,080
#   bytes of demand H2D. The routed-expert set is 43 layers x 256 experts x
#   3 x 4.46 MB = 147 GB, so the phase streamed it 22.9x over, with 745,172
#   evictions and 1,356 s of its 2,308 s spent in demand wait. 22.9 ~ kLayers
#   is the signature of a page-major/layer-inner loop nest: each page sweeps a
#   147 GB working set through a ~38 GB cache.
#   Target term: prefill demand H2D bytes, which is argmax_r for the phase.
#   Sign on other resources: compute, precision, routing and top-k are
#   untouched; the only cost added is one resident activation tile of
#   tile_tokens x 4 x 4096 x 4 B (about 134 MB at 512 tokens).
#   Correctness: reordering independent layer/page work changes no arithmetic,
#   so output must be bit-identical.
#   Rollback: prefill not faster beyond run variance, or any output byte moves.
#
# Arm budget: prompt ~512 tokens, --max-new 8 (decode is not under test).
#   reference ~5 min, candidate ~3 min, about 10 min total including the
#   ~100 s of fixed initialization and staging each arm pays.

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/dsv4f"}
result_dir=${RESULT_DIR:-"${repo_root}/results/deepseek-v4-prefill-layer-major-ab"}
runner=${RUNNER:-"${repo_root}/build/strata-deepseek-run"}
maximum_new_tokens=${MAX_NEW_TOKENS:-8}
maximum_context_tokens=${MAX_CONTEXT_TOKENS:-4096}
kv_device_cache=${KV_DEVICE_CACHE:-256M,256M,256M}
prompt_sentences=${PROMPT_SENTENCES:-34}
page_tokens=${PAGE_TOKENS:-64}

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
    printf 'prompt_sentences=%s\nprompt_bytes=%s\npage_tokens=%s\n' \
        "${prompt_sentences}" "${#prompt}" "${page_tokens}"
    sha256sum "${runner}"
    nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv
} >"${result_dir}/system.txt"

# reference reproduces the page-major nest exactly by tiling at the page width.
run_case() {
    local arm=$1 tile=$2
    mkdir -p "${result_dir}/${arm}"
    /usr/bin/time -v "${runner}" \
        --model "${model_dir}" --devices 0,1,2 \
        --host-memory 216G --vram-fraction 0.85 \
        --max-context "${maximum_context_tokens}" \
        --kv-device-cache "${kv_device_cache}" \
        --prefill-page-tokens "${page_tokens}" \
        --prefill-layer-tile "${tile}" \
        --max-new "${maximum_new_tokens}" --prompt "${prompt}" \
        --route-trace "${result_dir}/${arm}/routes.jsonl" \
        --logit-trace --layer-hash-trace --detailed-timing --quiet --json \
        >"${result_dir}/${arm}/generation.json" \
        2>"${result_dir}/${arm}/generation.log"
}

run_case reference "${page_tokens}"
run_case candidate 0

jq -n \
  --slurpfile r "${result_dir}/reference/generation.json" \
  --slurpfile c "${result_dir}/candidate/generation.json" '
  ($r[0]) as $r | ($c[0]) as $c |
  {
    prefill_seconds: {reference: $r.prefill_seconds, candidate: $c.prefill_seconds},
    prefill_speedup: ($r.prefill_seconds / $c.prefill_seconds),
    prefill_demand_h2d_bytes: {
      reference: $r.phases.prefill.cache.demand_h2d_bytes,
      candidate: $c.phases.prefill.cache.demand_h2d_bytes},
    h2d_reduction: ($r.phases.prefill.cache.demand_h2d_bytes /
                    $c.phases.prefill.cache.demand_h2d_bytes),
    prefill_evictions: {
      reference: $r.phases.prefill.cache.evictions,
      candidate: $c.phases.prefill.cache.evictions},
    prefill_demand_wait_seconds: {
      reference: $r.phases.prefill.cache.demand_wait_seconds,
      candidate: $c.phases.prefill.cache.demand_wait_seconds},
    prefill_ms_per_token: {
      reference: ($r.prefill_seconds * 1000 / $r.prefill_tokens),
      candidate: ($c.prefill_seconds * 1000 / $c.prefill_tokens)},
    rss_bytes: {reference: $r.rss_bytes, candidate: $c.rss_bytes},
    gates: {
      tiling_actually_differed: ($r.prefill_layer_tile_tokens == '"${page_tokens}"' and
                                 $c.prefill_layer_tile_tokens == 0),
      generated_tokens_equal: ($r.generated_token_ids == $c.generated_token_ids),
      logits_equal: ($r.diagnostics.logits == $c.diagnostics.logits),
      layer_hashes_equal: ($r.diagnostics.layer_hidden_hashes ==
                           $c.diagnostics.layer_hidden_hashes),
      operation_hashes_equal: ($r.diagnostics.operation_hashes ==
                               $c.diagnostics.operation_hashes),
      same_prefill_tokens: ($r.prefill_tokens == $c.prefill_tokens),
      prefill_faster: ($c.prefill_seconds < $r.prefill_seconds),
      moved_less: ($c.phases.prefill.cache.demand_h2d_bytes <
                   $r.phases.prefill.cache.demand_h2d_bytes)
    }
  } | .acceptance_pass = ([.gates[]] | all)' >"${result_dir}/summary.json"

cmp "${result_dir}/reference/routes.jsonl" "${result_dir}/candidate/routes.jsonl"
cat "${result_dir}/summary.json"
jq -e '.acceptance_pass == true' "${result_dir}/summary.json" >/dev/null
