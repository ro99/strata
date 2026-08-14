#!/usr/bin/env bash
# Runs the sparse-path in-chain selection gate: the 2,685-active-token arm on
# both topologies, above the 2,048-token indexer threshold.
#
# The arm proves nothing unless the indexer engaged, so the caller must check
# prompt_tokens (2,673) before reading any other number. Compare against the
# recorded baseline with compare_dsv4_sparse_gate.py.
#
# Usage: scripts/run_dsv4_rank_local_sparse_gate.sh <tag> [build-dir]
set -euo pipefail

TAG="${1:?usage: run_dsv4_rank_local_sparse_gate.sh <tag> [build-dir]}"
BUILD="${2:-build-landing-nccl}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/results/dsv4-rank-local-main-landing/step4-sparse-path"
PROMPT_FILE="${OUT}/prompt-long.txt"
RUN="${ROOT}/${BUILD}/strata-deepseek-run"

test -x "${RUN}"
test -f "${PROMPT_FILE}"
mkdir -p "${OUT}"

PROMPT="$(cat "${PROMPT_FILE}")"

common=(
  --model "${ROOT}/models/dsv4f"
  --prompt "${PROMPT}"
  --max-new 12
  --devices 1,2
  --max-context 8192
  --host-memory 216G
  --vram-fraction 0.95
  --device-resident-runtime
  --detailed-timing
  --quiet
  --json
)

run_arm() {
  local name="$1"; shift
  echo "== ${name} =="
  set +e
  "${RUN}" "${common[@]}" "$@" \
    >"${OUT}/${name}.json" 2>"${OUT}/${name}.log"
  local status=$?
  set -e
  echo "${status}" >"${OUT}/${name}.exit"
  echo "exit ${status}; json ${OUT}/${name}.json"
}

# ARMS selects which topologies run; each is about eight minutes, of which six
# are prefill. Default is both.
ARMS="${ARMS:-rank-local centralized}"
for arm in ${ARMS}; do
  case "${arm}" in
    rank-local) run_arm "rank-local-${TAG}" --decode-topology rank-local-tp2 ;;
    centralized) run_arm "centralized-${TAG}" ;;
    *) echo "unknown arm ${arm}" >&2; exit 2 ;;
  esac
  sha256sum "${OUT}/${arm}-${TAG}.json"
done
