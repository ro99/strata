#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID

if [[ "${1:-}" == "--help" ]]; then
    cat <<'EOF'
usage: scripts/check_glm52_determinism.sh

Runs the same greedy GLM-5.2 request repeatedly and reports the first token
position at which the generated token sequences differ.

Environment overrides:
  MODEL_DIR, RESULT_DIR, REPETITIONS, MAX_NEW_TOKENS, MAX_CONTEXT_TOKENS,
  CUDA_DEVICES, VRAM_FRACTION, STRATA_RUN, DIAGNOSTIC_TRACE
EOF
    exit 0
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/glm52"}
result_dir=${RESULT_DIR:-"${repo_root}/results/glm52-determinism"}
repetitions=${REPETITIONS:-3}
maximum_new_tokens=${MAX_NEW_TOKENS:-128}
maximum_context_tokens=${MAX_CONTEXT_TOKENS:-256}
devices=${CUDA_DEVICES:-0,1,2}
vram_fraction=${VRAM_FRACTION:-0.85}
prompt='What is the closer start to sun, and how distant it is from it?'
strata_run=${STRATA_RUN:-"${repo_root}/build/strata-run"}
diagnostic_args=()
if [[ "${DIAGNOSTIC_TRACE:-0}" == "1" ]]; then
    diagnostic_args+=(--diagnostic-trace)
fi

if ! command -v jq >/dev/null 2>&1; then
    echo "error: jq is required" >&2
    exit 1
fi
if [[ ! -x "${strata_run}" ]]; then
    echo "error: strata-run not found or not executable: ${strata_run}" >&2
    echo "build it first with: cmake --build build --parallel" >&2
    exit 1
fi
if [[ ! -d "${model_dir}" ]]; then
    echo "error: model directory does not exist: ${model_dir}" >&2
    exit 1
fi
if ! [[ "${repetitions}" =~ ^[2-9][0-9]*$ ]]; then
    echo "error: REPETITIONS must be at least 2" >&2
    exit 1
fi

mkdir -p "${result_dir}"
run_files=()
for ((run = 1; run <= repetitions; ++run)); do
    run_name=$(printf 'run-%02d' "${run}")
    output_file="${result_dir}/${run_name}.json"
    log_file="${result_dir}/${run_name}.log"
    echo "[determinism] ${run_name}/${repetitions}" >&2
    "${strata_run}" \
        --model "${model_dir}" \
        --prompt "${prompt}" \
        --devices "${devices}" \
        --vram-fraction "${vram_fraction}" \
        --max-context "${maximum_context_tokens}" \
        --max-new "${maximum_new_tokens}" \
        "${diagnostic_args[@]}" \
        --json \
        >"${output_file}" \
        2>"${log_file}"
    jq empty "${output_file}"
    run_files+=("${output_file}")
done

jq -s '
    map(.generated_token_ids) as $runs |
    ($runs | map(length) | min) as $limit |
    ([range(0; $limit) as $i |
        select(any($runs[]; .[$i] != $runs[0][$i])) |
        {index: $i, position: ($i + 1), tokens: [$runs[] | .[$i]]}
    ][0] // null) as $first |
    {
        repetitions: ($runs | length),
        generated_token_counts: ($runs | map(length)),
        unique_sequences: ($runs | unique | length),
        identical: (($runs | unique | length) == 1),
        first_divergence: $first
    }
' "${run_files[@]}" | tee "${result_dir}/summary.json"

echo >&2
echo "Artifacts: ${result_dir}" >&2
