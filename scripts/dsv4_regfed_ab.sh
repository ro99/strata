#!/usr/bin/env bash
# MIX-2: does the register-fed FP4/FP8 substitution pay at the production
# operating point? Two arms, interleaved, on one DeepSeek V4 process each.
#
#   regfed  STRATA_REGFED_MATMUL=1  register-fed skinny kernels (the default)
#   scalar  STRATA_REGFED_MATMUL=0  the incumbent scalar routes (the control)
#
# HYPOTHESIS: substituting the accepted QPN skinny kernels for the scalar FP8
# routes reduces decode wall time.
# PRIMARY METRIC: median steady-state decode seconds per token, first step
# excluded (the lazy fragment prepack lands there and is reported separately).
# CORRECTNESS GATE: identical greedy token ids across arms, or the divergence
# position reported. Both arms run temperature 0 on the same prompt.
# MEMORY CEILING: unchanged. The prepack permutes in place through transient
# scratch, so no weight has two device copies.
# ROLLBACK: STRATA_REGFED_MATMUL=0 restores the incumbent with no rebuild.
#
# BOTTLENECK ACCOUNTING, stated before the run so the result can falsify it.
# The change reduces CUDA kernel time on the FP8 projections and the shared
# expert. It does NOT reduce host MoE time, NVMe, or attention. It INCREASES
# per-token launch count on the shared expert -- nine against the incumbent's
# five -- so it adds to the serial dispatch term. If argmax_r in the decode
# phase is not CUDA kernel time, this cannot improve tau however fast the
# kernels are, and the summary will say so.
#
# ARM BUDGET. Each arm is one process: ~1 min of load and staging, a short
# prefill, then MAX_NEW decode steps. Six arms (3 reps x 2) is roughly 10-15
# minutes total. The prompt is deliberately short and MAX_CONTEXT small: the
# hypothesis is about decode, and a long prompt would spend the whole run in
# prefill measuring nothing. The rejected cheaper experiment was a standalone
# kernel benchmark -- experiments 0148 and 0159 already did that, and it is
# precisely the number that cannot answer an end-to-end question.
#
# Usage:  scripts/dsv4_regfed_ab.sh OUTDIR [REPS] [MAX_NEW] [MAX_CONTEXT]
# Run it inside a named tmux session; it is minutes long and holds both GPUs.
#   tmux new -s regfed-ab 'scripts/dsv4_regfed_ab.sh results/regfed-ab 2>&1 | tee results/regfed-ab/run.log'
set -euo pipefail

OUTDIR="${1:?usage: dsv4_regfed_ab.sh OUTDIR [REPS] [MAX_NEW] [MAX_CONTEXT]}"
REPS="${2:-3}"
MAX_NEW="${3:-32}"
MAX_CONTEXT="${4:-256}"

MODEL="${MODEL:-models/dsv4f}"
BINARY="${BINARY:-./build-release/strata-deepseek-run}"
HOST_MEMORY="${HOST_MEMORY:-216G}"
VRAM_FRACTION="${VRAM_FRACTION:-.95}"
HOST_ATTENTION_THREADS="${HOST_ATTENTION_THREADS:-28}"
# Empty selects the invocation experiment 0162 is known to run; set to
# rank-local-tp2 for the production decode topology.
DECODE_TOPOLOGY="${DECODE_TOPOLOGY:-}"
PROMPT="${PROMPT:-Explain tensor parallelism briefly.}"

mkdir -p "$OUTDIR"

if [ ! -x "$BINARY" ]; then
  echo "error: $BINARY is missing or not executable." >&2
  echo "       Build it first: cmake --build build-release --parallel" >&2
  exit 2
fi
# A Debug build has cost this campaign 45 minutes of A/B runs and two wrong
# conclusions. Refuse rather than measure one.
BUILD_DIR="$(dirname "$BINARY")"
if [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
  BUILD_TYPE="$(sed -n 's/^CMAKE_BUILD_TYPE:STRING=//p' "$BUILD_DIR/CMakeCache.txt")"
  if [ "$BUILD_TYPE" != "Release" ] && [ "$BUILD_TYPE" != "RelWithDebInfo" ]; then
    echo "error: $BUILD_DIR is CMAKE_BUILD_TYPE=${BUILD_TYPE:-unset}, not Release." >&2
    exit 2
  fi
fi

TOPOLOGY_FLAGS=()
if [ -n "$DECODE_TOPOLOGY" ]; then
  TOPOLOGY_FLAGS=(--decode-topology "$DECODE_TOPOLOGY")
fi

run_arm() {
  local arm="$1" switch="$2" rep="$3"
  local stem="$OUTDIR/${arm}-rep${rep}"
  echo "=== ${arm} rep ${rep} (STRATA_REGFED_MATMUL=${switch}) $(date -Is)"
  # strata-deepseek-run pins CUDA_DEVICE_ORDER=PCI_BUS_ID, so devices 1,2 are
  # the RTX 3090 rank pair matching nvidia-smi; after the remap they are
  # --devices 0,1 to this process. --device-resident-runtime already implies
  # --host-routed-moe: without it the device weight arena exhausts at
  # layers.2.ffn.shared_experts.w1, which is an explicit refusal, not a
  # fallback.
  if ! env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 \
        STRATA_REGFED_MATMUL="$switch" \
        "$BINARY" --model "$MODEL" \
        --devices 0,1 --host-memory "$HOST_MEMORY" \
        --vram-fraction "$VRAM_FRACTION" \
        --max-context "$MAX_CONTEXT" --max-new "$MAX_NEW" \
        --prompt "$PROMPT" \
        --device-resident-runtime \
        --host-attention-threads "$HOST_ATTENTION_THREADS" \
        "${TOPOLOGY_FLAGS[@]}" \
        --detailed-timing --json \
        --route-census "${stem}.census.json" \
        > "${stem}.json" 2> "${stem}.log"; then
    echo "error: ${arm} rep ${rep} failed. Last 40 lines of ${stem}.log:" >&2
    tail -40 "${stem}.log" >&2
    return 1
  fi
}

echo "register-fed A/B: ${REPS} interleaved reps, ${MAX_NEW} decode steps,"
echo "context ${MAX_CONTEXT}, topology ${DECODE_TOPOLOGY:-default}, into ${OUTDIR}"
echo "expected wall time: roughly $((REPS * 2 * 2)) to $((REPS * 2 * 3)) minutes"

for rep in $(seq 1 "$REPS"); do
  # Interleaved, so any drift in cache state, clocks or thermals is shared
  # between the arms rather than assigned to whichever ran second.
  run_arm regfed 1 "$rep"
  run_arm scalar 0 "$rep"
done

echo
python3 scripts/dsv4_regfed_ab_summary.py "$OUTDIR"
