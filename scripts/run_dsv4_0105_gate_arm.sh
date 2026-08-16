#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
arm=${1:?usage: run_dsv4_0105_gate_arm.sh baseline|candidate}
result_dir=${RESULT_DIR:-"${repo_root}/results/dsv4-0105-fp8-tensor-projections"}
model_dir=${MODEL_DIR:-"/home/rodrigo/Developer/strata/models/dsv4f"}
runner=${RUNNER:-"${result_dir}/strata-deepseek-run"}

case "${arm}" in
    baseline)
        tensor_flag=--no-dsv4-fp8-tensor-page
        ;;
    candidate)
        tensor_flag=--dsv4-fp8-tensor-page
        ;;
    *)
        echo "unknown arm: ${arm}" >&2
        exit 2
        ;;
esac

mkdir -p "${result_dir}"
if [[ -e "${result_dir}/${arm}.json" || -e "${result_dir}/${arm}.log" ]]; then
    echo "refusing to overwrite preserved ${arm} result" >&2
    exit 2
fi
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

{
    date --iso-8601=seconds
    printf 'arm=%s\n' "${arm}"
    printf 'tensor_flag=%s\n' "${tensor_flag}"
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" status --short --branch
    sha256sum "${runner}"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.free,memory.total \
        --format=csv
} >"${result_dir}/${arm}-system.txt"

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
    "${tensor_flag}" \
    >"${result_dir}/${arm}.json" \
    2>"${result_dir}/${arm}.log"
