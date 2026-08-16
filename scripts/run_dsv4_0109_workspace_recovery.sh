#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_dir=${RESULT_DIR:-"${repo_root}/results/dsv4-0109-workspace-recovery"}
model_dir=${MODEL_DIR:-"/home/rodrigo/Developer/strata/models/dsv4f"}
runner=${RUNNER:-"${repo_root}/build/strata-deepseek-run"}

mkdir -p "${result_dir}"
if [[ ! -x "${runner}" ]]; then
    echo "runner is missing or not executable: ${runner}" >&2
    exit 2
fi

words=(alpha bravo charlie delta echo foxtrot golf hotel india juliet
       kilo lima mike november oscar papa quebec romeo sierra tango)

run_arm() {
    local name=$1
    local prompt_words=$2
    local prompt=""
    for ((index=0; index<prompt_words; ++index)); do
        prompt+="${words[index % ${#words[@]}]} "
    done
    if [[ -e "${result_dir}/${name}.json" ||
          -e "${result_dir}/${name}.log" ]]; then
        echo "refusing to overwrite preserved ${name} result" >&2
        exit 2
    fi
    {
        date --iso-8601=seconds
        printf 'arm=%s\nprompt_words=%s\n' "${name}" "${prompt_words}"
        git -C "${repo_root}" rev-parse HEAD
        git -C "${repo_root}" status --short --branch
        sha256sum "${runner}"
        nvidia-smi --query-gpu=index,name,pci.bus_id,memory.free,memory.total \
            --format=csv
    } >"${result_dir}/${name}-system-before.txt"

    /usr/bin/time -v "${runner}" \
        --model "${model_dir}" \
        --devices 1,2 \
        --host-memory 216G \
        --vram-fraction 0.95 \
        --max-context 8192 \
        --device-resident-runtime \
        --decode-topology rank-local-tp2 \
        --prefill-page-tokens 8192 \
        --max-new 4 \
        --prompt "${prompt}" \
        --detailed-timing \
        --quiet \
        --json \
        >"${result_dir}/${name}.json" \
        2>"${result_dir}/${name}.log"

    {
        date --iso-8601=seconds
        nvidia-smi --query-gpu=index,name,pci.bus_id,memory.free,memory.total \
            --format=csv
    } >"${result_dir}/${name}-system-after.txt"
}

if [[ ${RUN_677:-1} == 1 ]]; then
    run_arm 677-w420 420
fi
if [[ ${RUN_2612:-1} == 1 ]]; then
    run_arm "${LONG_NAME:-2612-w1630}" 1630
fi

printf '0109 arms complete: %s\n' "${result_dir}"
