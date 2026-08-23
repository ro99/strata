#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
binary="${repo_root}/build-release/strata-gemma4-run"
model="${repo_root}/models/gemma4"
output=${1:-"${repo_root}/results/gemma4-regfed/scalar-g1.json"}
census=${2:-"${output%.json}.census.json"}

if [[ ! -x "${binary}" ]]; then
    echo "error: ${binary} is missing; build the Release tree first" >&2
    exit 1
fi

mkdir -p "$(dirname "${output}")"
CUDA_VISIBLE_DEVICES=1 STRATA_REGFED_MATMUL=0 "${binary}" \
    --model "${model}" \
    --devices 0 \
    --prompt "The capital of France is" \
    --max-new 8 \
    --temperature 0 \
    --seed 33377335 \
    --route-census "${census}" \
    --json >"${output}"
