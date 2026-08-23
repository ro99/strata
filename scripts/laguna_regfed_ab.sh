#!/usr/bin/env bash
# Does the register-fed fused MXFP4 MoE pay end to end on Laguna S 2.1?
#
# HYPOTHESIS: substituting the register-fed MoE kernels reduces decode wall time.
# PRIMARY METRIC: per-step wall, plus the matmul-kernel term the substitution
# actually targets.
# CORRECTNESS GATE: both arms generate coherent text; compared side by side.
# ROLLBACK: STRATA_REGFED_MATMUL=0, no rebuild.
#
# PREDICTION, stated before the run so it can be falsified (experiment 0168):
#   matmul kernels   ~14.70 ms/step -> ~3 ms, a 5x effect on that term;
#   step wall        ~236.98 ms/step -> ~229 ms, about 1.035x.
# The first is unmissable at one repetition. The second is not: a 3.5% effect
# cannot be separated from run variance without a spread estimate, and one
# repetition per arm gives none. Read the matmul term as the result and the wall
# clock as an indication only.
#
# ARM BUDGET: 2 arms x ~17 s (2.6 s load, 10.3 s prefill, 3.6 s decode) plus
# process start. Under a minute.
#
# Usage: scripts/laguna_regfed_ab.sh OUTDIR
set -euo pipefail
OUTDIR="${1:?usage: laguna_regfed_ab.sh OUTDIR}"
MODEL="${MODEL:-models/laguna}"
BINARY="${BINARY:-./build-release/strata-laguna-profile}"
DEVICES="${DEVICES:-0,1,2}"
CONTEXT="${CONTEXT:-128}"
MAX_NEW="${MAX_NEW:-16}"
PROMPT="${PROMPT:-The capital of France is}"
mkdir -p "$OUTDIR"

BUILD_DIR="$(dirname "$BINARY")"
if [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
  TYPE="$(sed -n 's/^CMAKE_BUILD_TYPE:STRING=//p' "$BUILD_DIR/CMakeCache.txt")"
  case "$TYPE" in Release|RelWithDebInfo) ;; *)
    echo "error: $BUILD_DIR is CMAKE_BUILD_TYPE=${TYPE:-unset}, not Release." >&2; exit 2;; esac
fi

run_arm() {
  local arm="$1" switch="$2"
  echo "=== ${arm} (STRATA_REGFED_MATMUL=${switch}) $(date -Is)"
  env STRATA_REGFED_MATMUL="$switch" "$BINARY" --model "$MODEL" \
    --devices "$DEVICES" --context "$CONTEXT" --max-new "$MAX_NEW" \
    --repetitions 1 --prompt "$PROMPT" > "$OUTDIR/${arm}.log" 2>&1 \
    || { echo "error: ${arm} failed:" >&2; tail -30 "$OUTDIR/${arm}.log" >&2; return 1; }
}

# Page-cache state dominates this measurement: the same copy is 0.52 GB/s cold
# and 6.09 GB/s warm (experiment 0169). A first run therefore warms the cache for
# whichever arm follows it, which on the first attempt handed the candidate a
# spurious 1.89x. Warm the checkpoint before either arm, and discard a warm-up
# run, so both arms see the same state.
echo "warming the page cache (both arms must see the same state)"
cat "$MODEL"/*.safetensors > /dev/null 2>&1 || true
run_arm warmup 0
run_arm scalar 0
run_arm regfed 1

echo
printf '%-34s %14s %14s\n' 'metric' 'scalar' 'regfed'
printf '%s\n' '-----------------------------------------------------------------'
# Read the DECODE section only. The profile prints prefill first, and prefill on
# this model is a separate open defect (experiment 0166) with a 0% cache hit rate
# -- reading its numbers as decode is how the first attempt went wrong.
decode() { sed -n '/^== decode ==/,$p' "$OUTDIR/$1.log"; }
extract() { decode "$1" | grep -iE "$2" | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1; }
for pair in \
  "per-step wall (ms)|per-step=" \
  "matmul kernels (ms)|matmul kernels" \
  "weight memcpy (ms)|weight memcpy" \
  "moe routed (ms)|moe routed" \
  "cache hit rate (%)|hit rate"; do
  label="${pair%%|*}"; pat="${pair#*|}"
  printf '%-34s %14s %14s\n' "$label" "$(extract scalar "$pat")" "$(extract regfed "$pat")"
done
echo
echo "matmul route census (proves which path served the run):"
for arm in scalar regfed; do
  echo "  --- $arm"
  sed -n '/-- matmul route census --/,$p' "$OUTDIR/${arm}.log" | tail -n +2 | sed 's/^/  /'
done
echo
echo "Full breakdowns: $OUTDIR/scalar.log and $OUTDIR/regfed.log"
