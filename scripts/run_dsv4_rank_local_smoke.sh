#!/usr/bin/env bash
# First end-to-end run of the rank-local TP2 decode path.
#
# This is a completion gate, not a performance gate. The question is whether
# the layer chain produces tokens at all; ms/token at an 18-token prompt does
# not describe the 1M operating point and must not be quoted as if it did.
#
# Operating point: devices 1,2 (the symmetric 24 GiB pair -- device 0 is a
# 16 GiB card and cannot hold the 22.5 GiB per-device set), 18-token chat
# prompt, 4,096-token context ceiling. Active tokens stay far below the 2,048
# sparse-indexer threshold, so this exercises the dense candidate path. The
# sparse regime is a separate arm.
#
# Budget: two arms, about three minutes of setup each plus a few seconds of
# decode. The rank-local arm additionally loads its own 12.9 GiB per-device
# weight set, so expect it to be the slower of the two to start.
#
# Usage: scripts/run_dsv4_rank_local_smoke.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner=${RUNNER:-"${repo_root}/build-landing-nccl/strata-deepseek-run"}
model_dir=${MODEL_DIR:-"${repo_root}/models/dsv4f"}
result_dir=${RESULT_DIR:-"${repo_root}/results/dsv4-rank-local-smoke"}

devices=${DEVICES:-1,2}
max_new=${MAX_NEW:-8}
max_context=${MAX_CONTEXT:-4096}
host_memory=${HOST_MEMORY:-216G}
vram_fraction=${VRAM_FRACTION:-0.95}
prompt=${PROMPT:-"what is the closes star to us, and how far is it?"}

mkdir -p "${result_dir}"

run_arm() {
    local arm=$1
    shift
    local log="${result_dir}/${arm}.log"
    printf 'arm=%s\ndevices=%s\nmax_new=%s\nmax_context=%s\nextra=%s\n' \
        "${arm}" "${devices}" "${max_new}" "${max_context}" "$*" \
        >"${result_dir}/${arm}.arm.txt"
    echo "=== ${arm} ==="
    set +e
    "${runner}" \
        --model "${model_dir}" \
        --prompt "${prompt}" \
        --max-new "${max_new}" \
        --devices "${devices}" \
        --max-context "${max_context}" \
        --host-memory "${host_memory}" \
        --vram-fraction "${vram_fraction}" \
        --device-resident-runtime \
        "$@" \
        >"${log}" 2>&1
    local status=$?
    set -e
    printf 'exit=%d\n' "${status}" >>"${result_dir}/${arm}.arm.txt"
    echo "${arm} exit=${status} log=${log}"
    tail -25 "${log}"
    echo
}

# Centralized first: it is the reference the rank-local arm is read against,
# and a failure here means the operating point is wrong, not the feature.
run_arm centralized --decode-topology centralized
run_arm rank-local --decode-topology rank-local-tp2

echo "results in ${result_dir}"
