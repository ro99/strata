#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_dir=${RESULT_DIR:-"${repo_root}/results/glm52-flash-attention-correctness"}
maximum_new_tokens=${MAX_NEW_TOKENS:-2}
compare_only=${COMPARE_ONLY:-0}

case "${compare_only}" in
    0|1) ;;
    *) echo "error: COMPARE_ONLY must be 0 or 1" >&2; exit 2 ;;
esac
if [[ "${compare_only}" == 0 && -d "${result_dir}" ]] &&
   [[ -n $(find "${result_dir}" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    echo "error: RESULT_DIR is not empty; choose a fresh deterministic path: ${result_dir}" >&2
    exit 1
fi
mkdir -p "${result_dir}"

common_environment=(
    REPETITIONS=1
    MAX_NEW_TOKENS="${maximum_new_tokens}"
    TRACE_ROUTES=1
    DETAILED_TIMING=1
)

if [[ "${compare_only}" == 0 ]]; then
    env "${common_environment[@]}" \
        RESULT_DIR="${result_dir}/scalar" FLASH_ATTENTION=0 \
        "${repo_root}/scripts/run_glm52_baseline.sh" \
        >"${result_dir}/scalar.log" 2>&1
    env "${common_environment[@]}" \
        RESULT_DIR="${result_dir}/flash" FLASH_ATTENTION=1 \
        "${repo_root}/scripts/run_glm52_baseline.sh" \
        >"${result_dir}/flash.log" 2>&1
fi

tokens_equal=false
if jq -e --slurpfile candidate "${result_dir}/flash/run-01.json" \
    '.generated_token_ids == $candidate[0].generated_token_ids' \
    "${result_dir}/scalar/run-01.json" >/dev/null; then
    tokens_equal=true
fi
route_coefficient_tolerance=0.000004
jq -s --argjson tolerance "${route_coefficient_tolerance}" \
    --slurpfile candidate "${result_dir}/flash/run-01.routes.jsonl" '
    def routes: map(select(has("experts")));
    routes as $reference |
    ($candidate | routes) as $candidate_routes |
    ($reference | length) as $count |
    ([range(0; $count) as $route |
      range(0; ($reference[$route].coefficients | length)) as $coefficient |
      (($reference[$route].coefficients[$coefficient] -
        $candidate_routes[$route].coefficients[$coefficient]) | fabs)] |
      max // 0) as $maximum_error |
    {
        route_count: $count,
        expert_routes_equal:
            (($candidate_routes | length) == $count and
             all(range(0; $count);
                 ($reference[.].token_position ==
                      $candidate_routes[.].token_position and
                  $reference[.].layer == $candidate_routes[.].layer and
                  $reference[.].experts == $candidate_routes[.].experts))),
        coefficient_tolerance: $tolerance,
        maximum_coefficient_absolute_error: $maximum_error,
        coefficients_within_contract: ($maximum_error <= $tolerance)
    }
' "${result_dir}/scalar/run-01.routes.jsonl" \
    >"${result_dir}/route-comparison.json"
routes_equal=$(jq -r \
    '.expert_routes_equal and .coefficients_within_contract' \
    "${result_dir}/route-comparison.json")

jq -n \
    --argjson tokens_equal "${tokens_equal}" \
    --argjson routes_equal "${routes_equal}" \
    --slurpfile route_comparison "${result_dir}/route-comparison.json" \
    --slurpfile scalar "${result_dir}/scalar/run-01.json" \
    --slurpfile flash "${result_dir}/flash/run-01.json" \
    '{
        tokens_equal: $tokens_equal,
        routes_equal: $routes_equal,
        route_comparison: $route_comparison[0],
        scalar_generation_tok_s: $scalar[0].generation_tok_s,
        flash_generation_tok_s: $flash[0].generation_tok_s,
        scalar_flash_attention: ($scalar[0].flash_attention // false),
        flash_flash_attention: ($flash[0].flash_attention // false),
        scalar_generated_token_ids: $scalar[0].generated_token_ids,
        flash_generated_token_ids: $flash[0].generated_token_ids
    }' >"${result_dir}/summary.json"

cat "${result_dir}/summary.json"
if [[ "${tokens_equal}" != true || "${routes_equal}" != true ]]; then
    exit 1
fi
