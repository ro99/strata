#!/usr/bin/env bash
# Interleaved A/B over two prebuilt profile binaries, with the checkpoint
# faulted into the page cache first so both arms start from the same state.
set -euo pipefail
cd "$(dirname "$0")/.."
SC="$1"; ARM_A="$2"; ARM_B="$3"; TAG="$4"
./scripts/laguna_warm_cache.sh models/laguna-s-21
for rep in 1 2 3; do
  for arm in "$ARM_A" "$ARM_B"; do
    "$SC/profile-$arm" --model models/laguna-s-21 --context 512 --max-new 12 \
      --repetitions 1 --prompt "What is a hash map?" \
      > "results/$TAG-$arm-$rep.log" 2>&1 || echo "ARM $arm rep $rep FAILED"
    echo "done $arm rep $rep"
  done
done
