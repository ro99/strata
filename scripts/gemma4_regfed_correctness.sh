#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
binary="${repo_root}/build-release/strata-gemma4-run"
model="${repo_root}/models/gemma4"
result_dir=${1:-"${repo_root}/results/gemma4-regfed/correctness"}
visible_device=${GEMMA4_CUDA_VISIBLE_DEVICES:-1}

if [[ ! -x "${binary}" ]]; then
    echo "error: ${binary} is missing; build the Release tree first" >&2
    exit 1
fi
if [[ $(sed -n 's/^CMAKE_BUILD_TYPE:STRING=//p' \
             "${repo_root}/build-release/CMakeCache.txt") != Release ]]; then
    echo "error: build-release is not a Release build" >&2
    exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
    echo "error: jq is required for the token comparison" >&2
    exit 1
fi

mkdir -p "${result_dir}"
run_arm() {
    local enabled=$1
    local name=$2
    CUDA_VISIBLE_DEVICES="${visible_device}" STRATA_REGFED_MATMUL="${enabled}" \
        "${binary}" \
        --model "${model}" \
        --devices 0 \
        --prompt "The capital of France is" \
        --max-new 8 \
        --temperature 0 \
        --seed 33377335 \
        --route-census "${result_dir}/${name}.census.json" \
        --json >"${result_dir}/${name}.json" \
        2>"${result_dir}/${name}.log"
}

run_arm 0 scalar
run_arm 1 register-fed

jq -n \
    --slurpfile scalar "${result_dir}/scalar.json" \
    --slurpfile candidate "${result_dir}/register-fed.json" \
    --slurpfile scalar_census "${result_dir}/scalar.census.json" \
    --slurpfile candidate_census "${result_dir}/register-fed.census.json" '
    def first_divergence($left; $right):
      ([range(0; ([$left|length, $right|length] | max)) |
         select(($left[.] // null) != ($right[.] // null))] | first) // null;
    ($scalar[0].generated_token_ids) as $left |
    ($candidate[0].generated_token_ids) as $right |
    {
      scalar_text: $scalar[0].answer,
      register_fed_text: $candidate[0].answer,
      scalar_tokens: $left,
      register_fed_tokens: $right,
      first_divergence_index: first_divergence($left; $right),
      scalar_census: $scalar_census[0],
      register_fed_census: $candidate_census[0]
    }' | tee "${result_dir}/summary.json"

if [[ $(jq '.routes.fp4_register_fed' \
             "${result_dir}/register-fed.census.json") == 0 ]]; then
    echo "error: register-fed arm recorded zero fp4_register_fed routes" >&2
    exit 1
fi
if [[ $(jq '.first_divergence_index' "${result_dir}/summary.json") != null ]]; then
    echo "error: scalar and register-fed token streams diverged" >&2
    exit 1
fi
