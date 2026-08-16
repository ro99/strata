#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_dir=${RESULT_DIR:-"${repo_root}/results/dsv4-0107-promotion-campaign"}
model_dir=${MODEL_DIR:-"/home/rodrigo/Developer/strata/models/dsv4f"}
runner=${RUNNER:-"${result_dir}/strata-deepseek-run"}

mkdir -p "${result_dir}"
if [[ ! -x "${runner}" ]]; then
    echo "preserved runner is missing or not executable: ${runner}" >&2
    exit 2
fi

words=(alpha bravo charlie delta echo foxtrot golf hotel india juliet
       kilo lima mike november oscar papa quebec romeo sierra tango)
prompt=""
for ((index=0; index<420; ++index)); do
    prompt+="${words[index % ${#words[@]}]} "
done

run_arm() {
    local arm=$1
    local repetition=$2
    local stem="${arm}-${repetition}"
    local attention_flag tensor_flag
    case "${arm}" in
        baseline)
            attention_flag=--no-dsv4-batched-page-attention
            tensor_flag=--no-dsv4-fp8-tensor-page
            ;;
        candidate)
            attention_flag=--dsv4-batched-page-attention
            tensor_flag=--dsv4-fp8-tensor-page
            ;;
        *)
            echo "unknown arm: ${arm}" >&2
            exit 2
            ;;
    esac

    if [[ -e "${result_dir}/${stem}.json" ||
          -e "${result_dir}/${stem}.log" ]]; then
        echo "refusing to overwrite preserved ${stem} result" >&2
        exit 2
    fi

    {
        date --iso-8601=seconds
        printf 'arm=%s\nrepetition=%s\n' "${arm}" "${repetition}"
        printf 'attention_flag=%s\ntensor_flag=%s\n' \
            "${attention_flag}" "${tensor_flag}"
        git -C "${repo_root}" rev-parse HEAD
        git -C "${repo_root}" status --short --branch
        sha256sum "${runner}"
        nvidia-smi --query-gpu=index,name,pci.bus_id,memory.free,memory.total \
            --format=csv
    } >"${result_dir}/${stem}-system.txt"

    /usr/bin/time -v "${runner}" \
        --model "${model_dir}" \
        --devices 1,2 \
        --host-memory 216G \
        --vram-fraction 0.95 \
        --max-context 4096 \
        --device-resident-runtime \
        --decode-topology rank-local-tp2 \
        --prefill-page-tokens 8192 \
        --max-new 4 \
        --prompt "${prompt}" \
        --detailed-timing \
        --quiet \
        --json \
        "${attention_flag}" \
        "${tensor_flag}" \
        >"${result_dir}/${stem}.json" \
        2>"${result_dir}/${stem}.log"
}

# Interleave complete arms so time-dependent machine state affects both
# mechanisms at the same cadence.
for repetition in 1 2 3; do
    run_arm baseline "${repetition}"
    run_arm candidate "${repetition}"
done

printf 'campaign complete: %s\n' "${result_dir}"
