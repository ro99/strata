#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_root=${RESULT_ROOT:-"${repo_root}/results/glm52-observability/baseline"}
rounds=${ROUNDS:-3}

mkdir -p "${result_root}"
counters_files=()
full_files=()

run_mode() {
    local round=$1
    local mode=$2
    local result_dir="${result_root}/round-$(printf '%02d' "${round}")-${mode}"
    local trace=0
    local detailed=0
    if [[ "${mode}" == full ]]; then
        trace=1
        detailed=1
    fi
    RESULT_DIR="${result_dir}" REPETITIONS=1 TRACE_ROUTES="${trace}" \
        DETAILED_TIMING="${detailed}" "${repo_root}/scripts/run_glm52_baseline.sh"
    if [[ "${mode}" == counters ]]; then
        counters_files+=("${result_dir}/run-01.json")
    else
        full_files+=("${result_dir}/run-01.json")
    fi
}

for ((round = 1; round <= rounds; ++round)); do
    if ((round % 2 == 1)); then
        run_mode "${round}" counters
        run_mode "${round}" full
    else
        run_mode "${round}" full
        run_mode "${round}" counters
    fi
done

jq -s \
    --slurpfile full <(jq -s '.' "${full_files[@]}") '
    def median:
        sort as $s | length as $n |
        if ($n % 2) == 1 then $s[$n / 2 | floor]
        else (($s[$n / 2 - 1] + $s[$n / 2]) / 2) end;
    . as $counters | $full[0] as $full_runs |
    ($counters | map(.generation_tok_s) | median) as $counters_median |
    ($full_runs | map(.generation_tok_s) | median) as $full_median |
    {
        rounds: ($counters | length),
        order: "odd counters/full; even full/counters",
        counters_generation_tok_s: ($counters | map(.generation_tok_s)),
        counters_generation_tok_s_median: $counters_median,
        full_generation_tok_s: ($full_runs | map(.generation_tok_s)),
        full_generation_tok_s_median: $full_median,
        full_overhead_percent: (($counters_median - $full_median) /
                                $counters_median * 100.0),
        counters_accounting_reconciles: ($counters | map(
            .checkpoint_read_bytes ==
            (.phases.prefill.checkpoint_read_bytes + .phases.decode.checkpoint_read_bytes))),
        full_accounting_reconciles: ($full_runs | map(
            .checkpoint_read_bytes ==
            (.phases.prefill.checkpoint_read_bytes + .phases.decode.checkpoint_read_bytes))),
        unique_generated_sequences: ([$counters[], $full_runs[]] |
                                     map(.generated_token_ids) | unique | length),
        generated_token_ids: $counters[0].generated_token_ids
    }
' "${counters_files[@]}" >"${result_root}/overhead-summary.json"

cat "${result_root}/overhead-summary.json"
