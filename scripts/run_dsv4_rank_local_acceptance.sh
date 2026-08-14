#!/usr/bin/env bash
# Short-context production acceptance matrix for rank-local TP2 decode.
# Three centralized/rank-local pairs are interleaved. The first decode step is
# reported as warm-up; the remaining per-step walls define each repetition's
# steady median. Full-context sparse selection is a later, separate gate.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner=${RUNNER:-"${repo_root}/build-landing-nccl/strata-deepseek-run"}
model_dir=${MODEL_DIR:-"${repo_root}/models/dsv4f"}
result_dir=${RESULT_DIR:-"${repo_root}/results/dsv4-rank-local-main-landing/step2-acceptance-r2"}
rank_local_oracle=${RANK_LOCAL_ORACLE:-"${repo_root}/results/dsv4-rank-local-main-landing/step2-sequential-oracle-32/sequential.json"}
prompt=${PROMPT:-"what is the closes star to us, and how far is it?"}
devices=${DEVICES:-1,2}
reuse_existing_arms=${REUSE_EXISTING_ARMS:-0}

mkdir -p "${result_dir}"

if [[ ! -f "${rank_local_oracle}" ]]; then
    printf 'missing rank-local sequential oracle: %s\n' "${rank_local_oracle}" >&2
    exit 2
fi
if ! jq -e '(.generated_token_ids | length) == 32' \
    "${rank_local_oracle}" >/dev/null; then
    printf 'rank-local sequential oracle does not contain 32 tokens: %s\n' \
        "${rank_local_oracle}" >&2
    exit 2
fi

validate_arm() {
    local stem=$1
    jq -e '.decode_steps == 31 and
           (.generated_token_ids | length) == 32 and
           (.decode_step_seconds | length) == 31 and
           .decode_checkpoint_read_bytes == 0' \
        "${stem}.json" >/dev/null
}

run_arm() {
    local repetition=$1
    local arm=$2
    local topology=$3
    local stem="${result_dir}/rep-${repetition}-${arm}"
    set +e
    "${runner}" \
        --model "${model_dir}" \
        --prompt "${prompt}" \
        --max-new 32 \
        --devices "${devices}" \
        --max-context 4096 \
        --host-memory 216G \
        --vram-fraction 0.95 \
        --device-resident-runtime \
        --decode-topology "${topology}" \
        --detailed-timing --quiet --json \
        >"${stem}.json" 2>"${stem}.log"
    local status=$?
    set -e
    printf '%d\n' "${status}" >"${stem}.exit"
    if [[ ${status} -ne 0 ]]; then
        return "${status}"
    fi
    validate_arm "${stem}"
}

for repetition in 0 1 2; do
    if [[ ${reuse_existing_arms} == 1 ]]; then
        validate_arm "${result_dir}/rep-${repetition}-centralized"
        validate_arm "${result_dir}/rep-${repetition}-rank-local"
    else
        run_arm "${repetition}" centralized centralized
        run_arm "${repetition}" rank-local rank-local-tp2
    fi
    if ! cmp -s \
        <(jq -c '.generated_token_ids' \
            "${result_dir}/rep-${repetition}-rank-local.json") \
        <(jq -c '.generated_token_ids' "${rank_local_oracle}"); then
        printf 'rank-local sequential-oracle mismatch in repetition %d\n' \
            "${repetition}" \
            >"${result_dir}/first-mismatch.txt"
        exit 4
    fi
done

jq -s '
    def median:
        sort as $s |
        if ($s | length) % 2 == 1 then
            $s[($s | length) / 2 | floor]
        else
            ($s[($s | length) / 2 - 1] +
             $s[($s | length) / 2]) / 2
        end;
    [range(0; length) as $index |
        .[$index] as $run |
        $run + {
            arm: (if $index % 2 == 0 then "centralized"
                  else "rank-local" end),
            repetition: ($index / 2 | floor),
            token_hash: ($run.generated_token_ids | @json),
            cold_ms: ($run.decode_step_seconds[0] * 1000),
            steady_median_ms:
                ([$run.decode_step_seconds[1:][] * 1000] | median),
            steady_min_ms:
                ([$run.decode_step_seconds[1:][] * 1000] | min),
            steady_max_ms:
                ([$run.decode_step_seconds[1:][] * 1000] | max)
        }
    ] as $runs |
    [$runs[] | select(.arm == "centralized") |
        .steady_median_ms] as $central_medians |
    [$runs[] | select(.arm == "rank-local") |
        .steady_median_ms] as $rank_medians |
    {
        runs: [$runs[] | {
            arm, repetition, cold_ms, steady_median_ms,
            steady_min_ms, steady_max_ms, decode_checkpoint_read_bytes,
            rss_bytes, device_vram_used_bytes
        }],
        deterministic_tokens_within_each_topology:
            (([$runs[] | select(.arm == "centralized") | .token_hash] |
                 unique | length) == 1 and
             ([$runs[] | select(.arm == "rank-local") | .token_hash] |
                 unique | length) == 1),
        cross_topology_token_identity_required: false,
        rank_local_oracle: "sequential rank-local full 32-token sequence",
        rank_local_oracle_exact: true,
        zero_decode_checkpoint_io:
            ([$runs[].decode_checkpoint_read_bytes] | all(. == 0)),
        centralized_repetition_medians_ms: $central_medians,
        rank_local_repetition_medians_ms: $rank_medians,
        centralized_median_ms: ($central_medians | median),
        rank_local_median_ms: ($rank_medians | median),
        rank_local_tokens_per_second:
            (1000 / ($rank_medians | median)),
        strict_target_ms: 125.0,
        accepted_variance_ms: 2.0,
        review_ceiling_ms: 127.0,
        strict_throughput_gate:
            (($rank_medians | median) <= 125.0),
        review_variance_gate:
            (($rank_medians | median) <= 127.0),
        step2_pass:
            (([$runs[] | select(.arm == "centralized") | .token_hash] |
                  unique | length) == 1 and
             ([$runs[] | select(.arm == "rank-local") | .token_hash] |
                  unique | length) == 1 and
             ([$runs[].decode_checkpoint_read_bytes] | all(. == 0)) and
             (($rank_medians | median) <= 127.0))
    }
' \
    "${result_dir}/rep-0-centralized.json" \
    "${result_dir}/rep-0-rank-local.json" \
    "${result_dir}/rep-1-centralized.json" \
    "${result_dir}/rep-1-rank-local.json" \
    "${result_dir}/rep-2-centralized.json" \
    "${result_dir}/rep-2-rank-local.json" \
    >"${result_dir}/summary.json"

jq -e '.step2_pass' "${result_dir}/summary.json" >/dev/null || exit 3
