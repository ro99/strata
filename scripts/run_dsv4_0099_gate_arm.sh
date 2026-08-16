#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
arm=${1:?usage: run_dsv4_0099_gate_arm.sh baseline|candidate}
result_dir=${RESULT_DIR:-"${repo_root}/results/dsv4-0099-prepared-selection-gate"}
model_dir=${MODEL_DIR:-"/home/rodrigo/Developer/strata/models/dsv4f"}

case "${arm}" in
    baseline)
        runner=${BASELINE_RUNNER:-"/tmp/strata-0099-baseline/build/strata-deepseek-run"}
        ;;
    candidate)
        runner=${CANDIDATE_RUNNER:-"${repo_root}/build/strata-deepseek-run"}
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

words=(alpha bravo charlie delta echo foxtrot golf hotel india juliet
       kilo lima mike november oscar papa quebec romeo sierra tango)
prompt=""
for ((index=0; index<420; ++index)); do
    prompt+="${words[index % ${#words[@]}]} "
done

{
    date --iso-8601=seconds
    printf 'arm=%s\n' "${arm}"
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
    >"${result_dir}/${arm}.json" \
    2>"${result_dir}/${arm}.log"
