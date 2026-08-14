#!/usr/bin/env bash
# Step 5 gate: centralized decode on this branch must be bit-identical to main.
#
# Three arms, identical flags. Only flags that exist on BOTH revisions are used,
# so the comparison is not confounded by a branch-only option:
#
#   main     main's runner, Release, NCCL off      the reference
#   branch   this branch, Release, NCCL off        the bit-identity claim
#   nccl     this branch, Release, NCCL on         proves the NCCL build does
#                                                  not perturb the default path
#
# Generation oracle is the generated token IDs and the answer text. Teacher
# forcing oracle is --logit-trace, which reports the top-k logits at every
# prompt position, so a divergence anywhere in the forward pass shows up even
# when sampling happens to agree.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
main_runner=${MAIN_RUNNER:?set MAIN_RUNNER to main-revision strata-deepseek-run}
branch_runner=${BRANCH_RUNNER:-"${repo_root}/build-landing/strata-deepseek-run"}
nccl_runner=${NCCL_RUNNER:-"${repo_root}/build-landing-nccl/strata-deepseek-run"}
model_dir=${MODEL_DIR:-"${repo_root}/models/dsv4f"}
result_dir=${RESULT_DIR:-"${repo_root}/results/dsv4-rank-local-main-landing/step5-centralized-main-parity"}
prompt=${PROMPT:-"what is the closes star to us, and how far is it?"}
devices=${DEVICES:-1,2}
max_new=${MAX_NEW:-32}

mkdir -p "${result_dir}"

run_arm() {
    local name=$1
    local runner=$2
    local stem="${result_dir}/${name}"
    if [[ ! -x "${runner}" ]]; then
        printf 'missing runner for arm %s: %s\n' "${name}" "${runner}" >&2
        return 2
    fi
    set +e
    "${runner}" \
        --model "${model_dir}" \
        --prompt "${prompt}" \
        --max-new "${max_new}" \
        --devices "${devices}" \
        --max-context 4096 \
        --host-memory 216G \
        --vram-fraction 0.95 \
        --device-resident-runtime \
        --logit-trace \
        --quiet --json \
        >"${stem}.json" 2>"${stem}.log"
    local status=$?
    set -e
    printf '%d\n' "${status}" >"${stem}.exit"
    printf '%s exit %d\n' "${name}" "${status}"
    return 0
}

for arm in "main:${main_runner}" "branch:${branch_runner}" "nccl:${nccl_runner}"; do
    run_arm "${arm%%:*}" "${arm#*:}"
done

printf '\n== parity ==\n'
python3 "${repo_root}/scripts/compare_dsv4_centralized_parity.py" \
    "${result_dir}/main.json" \
    "${result_dir}/branch.json" \
    "${result_dir}/nccl.json"
