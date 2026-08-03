#!/usr/bin/env bash
# Interleaved A/B for Laguna stage 1: zero-copy view() upload + weight arena.
# Hypothesis: decode argmax_r is routed-expert miss staging; removing the heap
# copy and the per-weight cudaMalloc reduces it without touching any other
# resource.
# Primary metric: decode ms/token at 512 context, median of 3 interleaved reps.
# Correctness gate: identical greedy token ids to the baseline arm.
# Memory ceiling: unchanged admitted budgets (19.31/19.31/12.54 GiB).
# Rollback: revert if decode ms/token regresses or greedy tokens diverge.
set -euo pipefail
cd "$(dirname "$0")/.."
SC="$1"
OUT="${2:-results}"
PROMPT="What is a hash map?"
for rep in 1 2 3; do
  for arm in baseline stage1; do
    "$SC/profile-$arm" --model models/laguna-s-21 --context 512 --max-new 12 \
      --repetitions 1 --prompt "$PROMPT" > "$OUT/ab-$arm-$rep.log" 2>&1 \
      || echo "ARM $arm rep $rep FAILED"
    echo "done $arm rep $rep"
  done
done
