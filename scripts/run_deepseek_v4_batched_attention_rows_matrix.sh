#!/usr/bin/env bash
set -euo pipefail

# Promotion matrix for exact batched attention rows at the 511-token operating
# point. Three interleaved repetitions compare main/candidate under forced CUDA
# attention and re-measure the candidate's 256-row hybrid crossover.

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/DeepSeek-V4-Flash-DSpark"}
baseline=${BASELINE_RUNNER:?set BASELINE_RUNNER to the main executable}
candidate=${CANDIDATE_RUNNER:-"${repo_root}/build/strata-deepseek-run"}
result_dir=${RESULT_DIR:-"${repo_root}/results/deepseek-v4-batched-attention-rows-matrix"}
repetitions=${REPETITIONS:-3}
maximum_new_tokens=${MAX_NEW_TOKENS:-128}
prompt_sentences=${PROMPT_SENTENCES:-34}

if [[ -n $(git -C "${repo_root}" status --porcelain) ]]; then
    echo "error: promotion evidence requires a clean frozen revision" >&2
    exit 1
fi
if ((repetitions != 3)); then
    echo "error: promotion matrix requires exactly three repetitions" >&2
    exit 1
fi

mkdir -p "${result_dir}"
cp --reflink=auto "${baseline}" "${result_dir}/baseline-runner"
cp --reflink=auto "${candidate}" "${result_dir}/candidate-runner"
baseline="${result_dir}/baseline-runner"
candidate="${result_dir}/candidate-runner"

sentences=(
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
prompt=
for ((index=0; index<prompt_sentences; ++index)); do
    prompt+="${sentences[index % ${#sentences[@]}]} "
done

{
    date --iso-8601=seconds
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" status --short --branch
    printf 'repetitions=%s\nprompt_sentences=%s\nprompt_bytes=%s\n' \
        "${repetitions}" "${prompt_sentences}" "${#prompt}"
    sha256sum "${baseline}" "${candidate}"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.free,memory.total \
        --format=csv
} >"${result_dir}/system.txt"
: >"${result_dir}/order.txt"

run_arm() {
    local arm=$1 repetition=$2 runner minimum_rows
    case "${arm}" in
        baseline-forced) runner=${baseline}; minimum_rows=0 ;;
        candidate-forced) runner=${candidate}; minimum_rows=0 ;;
        candidate-hybrid) runner=${candidate}; minimum_rows=256 ;;
        *) echo "error: unknown arm ${arm}" >&2; return 1 ;;
    esac
    local stem="${result_dir}/${arm}/run-$(printf '%02d' "${repetition}")"
    mkdir -p "$(dirname "${stem}")"
    printf '%s %s\n' "${repetition}" "${arm}" | tee -a "${result_dir}/order.txt"
    /usr/bin/time -v "${runner}" \
        --model "${model_dir}" --devices 0,1,2 \
        --host-memory 216G --vram-fraction 0.85 --max-context 4096 \
        --kv-device-cache 256M,256M,256M --prefill-page-tokens 64 \
        --max-new "${maximum_new_tokens}" --prompt "${prompt}" \
        --pin-resident-arena --block-kv-cache --detailed-timing --quiet --json \
        --flash-attention --flash-attention-minimum-rows "${minimum_rows}" \
        >"${stem}.json" 2>"${stem}.log"
}

arms=(baseline-forced candidate-hybrid candidate-forced)
for ((repetition=1; repetition<=repetitions; ++repetition)); do
    offset=$((repetition - 1))
    for ((item=0; item<${#arms[@]}; ++item)); do
        run_arm "${arms[(item + offset) % ${#arms[@]}]}" "${repetition}"
    done
done

for arm in "${arms[@]}"; do
    jq -s '.' "${result_dir}/${arm}"/run-*.json >"${result_dir}/${arm}.json"
done

jq -n \
    --slurpfile baseline "${result_dir}/baseline-forced.json" \
    --slurpfile forced "${result_dir}/candidate-forced.json" \
    --slurpfile hybrid "${result_dir}/candidate-hybrid.json" '
    def med: sort | .[length / 2 | floor];
    def arm($runs): {
        initialization_seconds: ([$runs[].initialization_seconds] | med),
        prefill_seconds: ([$runs[].prefill_seconds] | med),
        prefill_attention_seconds:
            ([$runs[].phases.prefill.graph.attention_seconds] | med),
        prefill_attention_score_seconds:
            ([$runs[].phases.prefill.graph.attention_score_seconds] | med),
        prefill_synchronization_seconds:
            ([$runs[].phases.prefill.cuda.critical_path_synchronization_seconds] | med),
        prefill_kernel_seconds:
            ([$runs[].phases.prefill.cuda.critical_path_kernel_seconds] | med),
        prefill_weight_h2d_bytes:
            ([$runs[].phases.prefill.cuda.weight_h2d_bytes] | med),
        prefill_activation_h2d_bytes:
            ([$runs[].phases.prefill.cuda.activation_h2d_bytes] | med),
        prefill_activation_d2h_bytes:
            ([$runs[].phases.prefill.cuda.activation_d2h_bytes] | med),
        prefill_flash_attention_calls:
            ([$runs[].phases.prefill.cuda.flash_attention_calls] | med),
        decode_tokens_per_second: [$runs[] | .decode_steps / .decode_seconds],
        median_decode_tokens_per_second:
            ([$runs[] | .decode_steps / .decode_seconds] | med),
        decode_seconds: [$runs[].decode_seconds],
        median_decode_seconds: ([$runs[].decode_seconds] | med),
        attention_seconds: ([$runs[].phases.decode.graph.attention_seconds] | med),
        attention_score_seconds:
            ([$runs[].phases.decode.graph.attention_score_seconds] | med),
        attention_kv_seconds:
            ([$runs[].phases.decode.graph.attention_kv_seconds] | med),
        attention_query_seconds:
            ([$runs[].phases.decode.graph.attention_query_seconds] | med),
        attention_output_seconds:
            ([$runs[].phases.decode.graph.attention_output_seconds] | med),
        moe_seconds: ([$runs[].phases.decode.graph.moe_seconds] | med),
        mhc_pre_seconds: ([$runs[].phases.decode.graph.mhc_pre_seconds] | med),
        upload_wait_seconds:
            ([$runs[].phases.decode.cuda.critical_path_upload_wait_seconds] | med),
        synchronization_seconds:
            ([$runs[].phases.decode.cuda.critical_path_synchronization_seconds] | med),
        kernel_seconds:
            ([$runs[].phases.decode.cuda.critical_path_kernel_seconds] | med),
        weight_h2d_bytes: ([$runs[].phases.decode.cuda.weight_h2d_bytes] | med),
        activation_h2d_bytes:
            ([$runs[].phases.decode.cuda.activation_h2d_bytes] | med),
        activation_d2h_bytes:
            ([$runs[].phases.decode.cuda.activation_d2h_bytes] | med),
        cache_misses: ([$runs[].phases.decode.cache.misses] | med),
        flash_attention_calls:
            ([$runs[].phases.decode.cuda.flash_attention_calls] | med),
        attention_cuda_dispatches:
            ([$runs[].phases.decode.graph.attention_cuda_dispatches] | med),
        attention_scalar_dispatches:
            ([$runs[].phases.decode.graph.attention_scalar_dispatches] | med),
        rss_bytes: ([$runs[].rss_bytes] | med),
        required_host_bytes: ([$runs[].memory_plan.required_host_bytes] | med),
        kv_host_peak_bytes: ([$runs[].kv_cache.host_peak_bytes] | med),
        kv_device_peak_bytes: [range(0; 3) as $device |
            ([$runs[].kv_cache.device_peak_bytes[$device]] | med)]
    };
    ($baseline[0]) as $b | ($forced[0]) as $f | ($hybrid[0]) as $h |
    (arm($b)) as $bm | (arm($f)) as $fm | (arm($h)) as $hm |
    {
      arms: {baseline_forced: $bm, candidate_forced: $fm,
             candidate_hybrid: $hm},
      candidate_forced_speedup:
          ($fm.median_decode_tokens_per_second /
           $bm.median_decode_tokens_per_second),
      candidate_hybrid_speedup:
          ($hm.median_decode_tokens_per_second /
           $bm.median_decode_tokens_per_second),
      forced_over_hybrid:
          ($fm.median_decode_tokens_per_second /
           $hm.median_decode_tokens_per_second),
      synchronization_change:
          ($fm.synchronization_seconds / $bm.synchronization_seconds - 1),
      gates: {
        prompt_is_511_tokens: ([($b + $f + $h)[].prompt_tokens == 511] | all),
        generated_tokens_equal:
            ([($b + $f + $h)[].generated_token_ids == $b[0].generated_token_ids] | all),
        zero_decode_checkpoint_reads:
            ([($b + $f + $h)[].decode_checkpoint_read_bytes == 0] | all),
        bounded_memory: ([($b + $f + $h)[] |
            .rss_bytes <= .memory_plan.required_host_bytes and
            .kv_cache.host_peak_bytes <= .kv_cache.host_capacity_bytes and
            ([range(0; .kv_cache.device_peak_bytes | length) as $device |
                .kv_cache.device_peak_bytes[$device] <=
                .kv_cache.device_capacity_bytes[$device]] | all)] | all),
        forced_transfer_bytes_unchanged:
            ($fm.weight_h2d_bytes == $bm.weight_h2d_bytes and
             $fm.activation_h2d_bytes == $bm.activation_h2d_bytes and
             $fm.activation_d2h_bytes == $bm.activation_d2h_bytes),
        forced_win_outside_range:
            (($f | map(.decode_steps / .decode_seconds) | min) >
             ($b | map(.decode_steps / .decode_seconds) | max))
      }
    }
    | .correctness_pass = ([.gates | to_entries[] |
        select(.key != "forced_win_outside_range") | .value] | all)
    | .promotion_pass = ([.gates[]] | all)
    | .best_candidate =
        (if $fm.median_decode_tokens_per_second > $hm.median_decode_tokens_per_second
         then "forced" else "hybrid" end)
    ' >"${result_dir}/summary.json"

cat "${result_dir}/summary.json"
