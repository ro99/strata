#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/DeepSeek-V4-Flash-DSpark"}
result_dir=${RESULT_DIR:-"${repo_root}/results/deepseek-v4-parallel-warmup-ab"}
candidate=${CANDIDATE_BINARY:-"${repo_root}/build/strata-deepseek-run"}
reference=${REFERENCE_BINARY:-"/tmp/strata-dsv4-parallel-warmup-baseline/build/strata-deepseek-run"}
warmup_workers=${SPINE_WARMUP_WORKERS:-3}

mkdir -p "${result_dir}"
for binary in "${candidate}" "${reference}"; do
    if [[ ! -x "${binary}" ]]; then
        echo "error: DeepSeek runner is not executable: ${binary}" >&2
        exit 1
    fi
done

{
    date --iso-8601=seconds
    git -C "${repo_root}" rev-parse HEAD
    sha256sum "${candidate}" "${reference}"
    free -b
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.free,memory.total \
        --format=csv
} >"${result_dir}/system-before.txt"

run_one() {
    local variant=$1
    local binary=$2
    shift 2
    local stem="${result_dir}/pair-${variant}"

    free -b >"${stem}-memory-before.txt"
    /usr/bin/time -v "${binary}" \
        --model "${model_dir}" \
        --devices 0,1,2 \
        --host-memory 216G \
        --vram-fraction 0.85 \
        --max-context 2048 \
        --max-new 1 \
        --prompt Hi \
        --json \
        "$@" \
        >"${stem}.json" \
        2>"${stem}.log"
    free -b >"${stem}-memory-after.txt"
}

run_one candidate "${candidate}" --spine-warmup-workers "${warmup_workers}"
run_one reference "${reference}"

jq -n \
    --slurpfile candidate "${result_dir}/pair-candidate.json" \
    --slurpfile reference "${result_dir}/pair-reference.json" '
    {
        candidate: $candidate[0],
        reference: $reference[0],
        initialization_speedup: ($reference[0].initialization_seconds /
                                 $candidate[0].initialization_seconds),
        post_staging_speedup:
            (($reference[0].initialization_seconds -
              $reference[0].resident_staging_seconds) /
             ($candidate[0].initialization_seconds -
              $candidate[0].resident_staging_seconds)),
        tokens_match: ($candidate[0].generated_token_ids ==
                       $reference[0].generated_token_ids),
        staged_bytes_match: ($candidate[0].resident_stage_bytes ==
                             $reference[0].resident_stage_bytes),
        generation_reads_match: ($candidate[0].generation_checkpoint_read_bytes ==
                                 $reference[0].generation_checkpoint_read_bytes),
        decode_reads_match: ($candidate[0].decode_checkpoint_read_bytes ==
                             $reference[0].decode_checkpoint_read_bytes)
    }
' >"${result_dir}/summary.json"

cat "${result_dir}/summary.json"
