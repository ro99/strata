#!/usr/bin/env bash
set -euo pipefail

# A/B/C for the decode attention dispatch policy, interleaved, median of three.
#
# Contract (see docs/experiments for the recorded result):
#   Bottleneck measured first, at a ~512-token operating point after the
#   layer-major prefill fix and the pinned resident arena (experiments 0023 and
#   0024): a 317.7 ms decode step is 86.3 ms MoE demand wait, 52.0 ms attention
#   score, 47.0 ms MoE compute, 37.4 ms mHC pre, 27.0 ms attention query
#   projection, 24.6 ms attention output projection, and 43.3 ms of remainder.
#   Attention is 112.9 ms/step, and every one of its 5,461 decode dispatches
#   takes the host scalar path.
#
#   Target term: attention_score_seconds. The dispatch crossover
#   (should_dispatch_flash_attention_cuda) is keyed on score_stride --
#   window_count + attended_compressed_count, i.e. KEY rows, not query rows.
#   At this operating point that is 128 + (position+1)/ratio, so the 21
#   ratio-4 layers present 256..288 rows and the 20 ratio-128 layers plus the
#   2 uncompressed layers present 128..133. The 21 ratio-4 layers carry ~67%
#   of the scalar score work and sit exactly at the 256-row threshold.
#
#   Sign on other resources: the CUDA path adds per-call H2D of the packed
#   segments and one D2H of the output, and adds device kernel time. It does
#   not change routing, precision, top-k, expert count, or the numerical
#   contract (f64_dot_f32_score_f32_accum is preserved on both paths).
#   Correctness: output must be bit-identical on every arm.
#   Rollback: median decode not above the scalar reference beyond run variance,
#   or any output byte changing.
#
# Arms (one build, only runtime flags differ):
#   scalar  - current default: every layer on the host scalar path.
#   hybrid  - --flash-attention: CUDA at >= 256 rows, so 21 of 43 layers.
#   forced  - --flash-attention --flash-attention-minimum-rows 0: all layers.
# The third arm is what measures the crossover rather than assuming it.
#
# Budget: ~50 s initialization + ~110 s prefill + ~40 s decode = ~3.5 min per
# arm; 3 arms x 3 repetitions = ~32 minutes total. The prompt is the shortest
# that still exercises the 128-row sliding window and leaves the expert cache
# cold; decode is what is under test, so nothing longer is justified.

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/DeepSeek-V4-Flash-DSpark"}
result_dir=${RESULT_DIR:-"${repo_root}/results/deepseek-v4-attention-dispatch-ab"}
runner=${RUNNER:-"${repo_root}/build/strata-deepseek-run"}
maximum_new_tokens=${MAX_NEW_TOKENS:-128}
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
    local -a dispatch=()
    case "${arm}" in
        scalar) dispatch=() ;;
        hybrid) dispatch=(--flash-attention) ;;
        forced) dispatch=(--flash-attention --flash-attention-minimum-rows 0) ;;
        *) echo "unknown arm ${arm}" >&2; return 1 ;;
    esac
    /usr/bin/time -v "${runner}" \
        --model "${model_dir}" --devices 0,1,2 \
        --host-memory 216G --vram-fraction 0.85 \
        --max-context "${maximum_context_tokens}" \
        --kv-device-cache "${kv_device_cache}" \
        --prefill-page-tokens 64 \
        --pin-resident-arena \
        --max-new "${maximum_new_tokens}" --prompt "${prompt}" \
        --route-trace "${result_dir}/${name}/routes.jsonl" \
        --logit-trace --layer-hash-trace --detailed-timing --quiet --json \
        "${dispatch[@]}" \
        >"${result_dir}/${name}/generation.json" \
        2>"${result_dir}/${name}/generation.log"
}

# Interleaved, so thermal and page-cache drift hit every arm equally.
for ((repetition = 1; repetition <= repetitions; ++repetition)); do
    for arm in scalar hybrid forced; do
        run_case "${arm}" "${repetition}"
    done
done

median() { jq -s 'sort | .[(length - 1) / 2 | floor]'; }
metric() {  # arm, jq expression
    for ((repetition = 1; repetition <= repetitions; ++repetition)); do
        jq "$2" "${result_dir}/$1/run-$(printf '%02d' "${repetition}")/generation.json"
    done | median
}

arm_summary() {  # arm
    jq -n \
      --arg arm "$1" \
      --argjson rate "$(metric "$1" '.decode_steps / .decode_seconds')" \
      --argjson decode "$(metric "$1" '.decode_seconds')" \
      --argjson score "$(metric "$1" '.phases.decode.graph.attention_score_seconds')" \
      --argjson attention "$(metric "$1" '.phases.decode.graph.attention_seconds')" \
      --argjson wait "$(metric "$1" '.phases.decode.cache.demand_wait_seconds')" \
      --argjson cuda "$(metric "$1" '.phases.decode.graph.attention_cuda_dispatches')" \
      --argjson scalar "$(metric "$1" '.phases.decode.graph.attention_scalar_dispatches')" \
      --argjson h2d "$(metric "$1" '.phases.decode.cuda.flash_attention_h2d_bytes')" \
      --argjson kernel "$(metric "$1" '.phases.decode.cuda.maximum_device_flash_attention_seconds')" \
      '{($arm): {decode_steps_per_second: $rate, decode_seconds: $decode,
                 attention_seconds: $attention, attention_score_seconds: $score,
                 demand_wait_seconds: $wait, cuda_dispatches: $cuda,
                 scalar_dispatches: $scalar, flash_h2d_bytes: $h2d,
                 maximum_device_flash_seconds: $kernel}}'
}

jq -s 'add' \
  <(arm_summary scalar) <(arm_summary hybrid) <(arm_summary forced) \
  >"${result_dir}/arms.json"

jq -n \
  --slurpfile arms "${result_dir}/arms.json" \
  --slurpfile s "${result_dir}/scalar/run-01/generation.json" \
  --slurpfile h "${result_dir}/hybrid/run-01/generation.json" \
  --slurpfile f "${result_dir}/forced/run-01/generation.json" '
  ($arms[0]) as $a |
  {
    arms: $a,
    hybrid_speedup: ($a.hybrid.decode_steps_per_second /
                     $a.scalar.decode_steps_per_second),
    forced_speedup: ($a.forced.decode_steps_per_second /
                     $a.scalar.decode_steps_per_second),
    hybrid_score_ratio: ($a.scalar.attention_score_seconds /
                         $a.hybrid.attention_score_seconds),
    forced_score_ratio: ($a.scalar.attention_score_seconds /
                         $a.forced.attention_score_seconds),
    gates: {
      dispatch_actually_differed:
        ($a.scalar.cuda_dispatches == 0 and $a.hybrid.cuda_dispatches > 0 and
         $a.forced.cuda_dispatches > $a.hybrid.cuda_dispatches),
      generated_tokens_equal:
        ($s[0].generated_token_ids == $h[0].generated_token_ids and
         $s[0].generated_token_ids == $f[0].generated_token_ids),
      logits_equal: ($s[0].diagnostics.logits == $h[0].diagnostics.logits and
                     $s[0].diagnostics.logits == $f[0].diagnostics.logits),
      layer_hashes_equal:
        ($s[0].diagnostics.layer_hidden_hashes == $h[0].diagnostics.layer_hidden_hashes and
         $s[0].diagnostics.layer_hidden_hashes == $f[0].diagnostics.layer_hidden_hashes),
      operation_hashes_equal:
        ($s[0].diagnostics.operation_hashes == $h[0].diagnostics.operation_hashes and
         $s[0].diagnostics.operation_hashes == $f[0].diagnostics.operation_hashes),
      zero_decode_checkpoint_reads:
        ($s[0].decode_checkpoint_read_bytes == 0 and
         $h[0].decode_checkpoint_read_bytes == 0 and
         $f[0].decode_checkpoint_read_bytes == 0)
    }
  } | .correctness_pass = ([.gates[]] | all)
    | .best_arm = (["scalar","hybrid","forced"]
        | max_by($arms[0][.].decode_steps_per_second))' \
  >"${result_dir}/summary.json"

cmp "${result_dir}/scalar/run-01/routes.jsonl" \
    "${result_dir}/hybrid/run-01/routes.jsonl"
cmp "${result_dir}/scalar/run-01/routes.jsonl" \
    "${result_dir}/forced/run-01/routes.jsonl"
cat "${result_dir}/summary.json"
jq -e '.correctness_pass == true' "${result_dir}/summary.json" >/dev/null
