#!/usr/bin/env bash
set -euo pipefail

model_dir="${1:-models/laguna}"
result_dir="${2:-results/laguna-device-kv-ab}"
mkdir -p "${result_dir}"

run_arm() {
    local repetition="$1"
    local arm="$2"
    local -a arm_args=()
    if [[ "${arm}" == "host" ]]; then
        arm_args+=(--host-kv)
    fi
    ./build-release/strata-laguna-profile \
        --model "${model_dir}" \
        --devices 0,1,2 \
        --context 256 \
        --max-new 80 \
        --repetitions 2 \
        --vram-fraction 0.85 \
        --no-detailed-cuda-timing \
        "${arm_args[@]}" \
        >"${result_dir}/${repetition}-${arm}.log" 2>&1
}

run_arm 1 host
run_arm 1 device
run_arm 2 device
run_arm 2 host
run_arm 3 host
run_arm 3 device
