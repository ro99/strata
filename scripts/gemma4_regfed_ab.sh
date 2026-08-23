#!/usr/bin/env bash
# Gemma 4 MXFP4 register-fed end-to-end A/B at the production capped point.
#
# Hypothesis: replacing scalar FP4 projection kernels with register-fed FP4
# reduces the measured argmax term, GPU/HBM kernel service, and therefore
# steady batch-1 decode seconds per token.
# Primary metric: steady decode seconds/token with the first batch-1 step
# excluded. Correctness: greedy token IDs identical across every arm. Memory:
# one 19.5 GB resident model plus bounded workspaces on one 24 GB RTX 3090.
# Rollback: STRATA_REGFED_MATMUL=0 keeps canonical weights and W8A16 is never
# prepacked.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
outdir=${1:?usage: gemma4_regfed_ab.sh OUTDIR [REPS] [MAX_NEW]}
reps=${2:-3}
max_new=${3:-32}
binary=${BINARY:-"${repo_root}/build-release/strata-gemma4-run"}
model=${MODEL:-"${repo_root}/models/gemma4"}
visible_device=${GEMMA4_CUDA_VISIBLE_DEVICES:-1}
prompt=${PROMPT:-"The following sequence contains the integers from 1 through 100: 1,"}

if [[ ! -x "${binary}" ]]; then
    echo "error: ${binary} is missing; build the Release tree first" >&2
    exit 2
fi
build_dir=$(dirname "${binary}")
build_type=$(sed -n 's/^CMAKE_BUILD_TYPE:STRING=//p' \
                 "${build_dir}/CMakeCache.txt")
if [[ "${build_type}" != Release ]]; then
    echo "error: ${build_dir} is CMAKE_BUILD_TYPE=${build_type:-unset}, not Release" >&2
    exit 2
fi

mkdir -p "${outdir}"
nvidia-smi --query-gpu=index,pci.bus_id,name,power.limit,clocks.sm,clocks.max.sm \
    --format=csv,noheader >"${outdir}/gpu-operating-point.txt"

run_arm() {
    local arm=$1
    local switch=$2
    local rep=$3
    local stem="${outdir}/${arm}-rep${rep}"
    echo "=== ${arm} rep ${rep} STRATA_REGFED_MATMUL=${switch} $(date -Is)"
    if ! CUDA_VISIBLE_DEVICES="${visible_device}" \
         STRATA_REGFED_MATMUL="${switch}" \
         "${binary}" \
            --model "${model}" --devices 0 \
            --prompt "${prompt}" --max-new "${max_new}" \
            --temperature 0 --seed 33377335 \
            --route-census "${stem}.census.json" --json \
            >"${stem}.json" 2>"${stem}.log"; then
        echo "error: ${arm} rep ${rep} failed" >&2
        tail -40 "${stem}.log" >&2
        return 1
    fi
}

echo "Gemma 4 register-fed A/B: ${reps} reps/arm, ${max_new} generated tokens"
echo "Expected wall time: about 45-60 seconds/arm, $((reps * 2 * 45 / 60))-$((reps * 2)) minutes total."
echo "The rejected cheaper probe was the kernel microbenchmark: it already proves"
echo "the mechanism, but cannot prove full-model decode. The prompt is only 31 tokens"
echo "because scalar prefill is separately known not to amortize its weight reads."

for rep in $(seq 1 "${reps}"); do
    # Counterbalanced interleaving shares thermal and clock drift between arms.
    if ((rep % 2 == 1)); then
        run_arm register-fed 1 "${rep}"
        run_arm scalar 0 "${rep}"
    else
        run_arm scalar 0 "${rep}"
        run_arm register-fed 1 "${rep}"
    fi
done

python3 "${repo_root}/scripts/gemma4_regfed_ab_summary.py" "${outdir}"
