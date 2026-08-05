#!/usr/bin/env bash
# Interleaved A/B for Laguna stage 2: incremental F32 KV cache.
# Hypothesis: the KV restage decodes the whole BF16 cache to F32 every layer
# every token, which is O(context) host work to add one row. Storing rows
# already rounded through BF16 removes it without changing the arithmetic.
# Primary metric: decode ms/token at 512 context, median of 3 interleaved reps.
# Correctness gate: identical greedy token ids to the stage-1 arm.
# Rollback: revert if tokens diverge or decode regresses.
set -euo pipefail
cd "$(dirname "$0")/.."
SC="$1"
OUT="${2:-results}"
PROMPT="What is a hash map?"
for rep in 1 2 3; do
  for arm in stage1 stage2; do
    "$SC/profile-$arm" --model models/laguna-s-21 --context 512 --max-new 12 \
      --repetitions 1 --prompt "$PROMPT" > "$OUT/s2-$arm-$rep.log" 2>&1 \
      || echo "ARM $arm rep $rep FAILED"
    echo "done $arm rep $rep"
  done
done
