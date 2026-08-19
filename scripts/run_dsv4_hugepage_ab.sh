#!/usr/bin/env bash
# A/B: transparent hugepages on the tiled routed-expert arena.
# One binary, two selectors, interleaved. Decode is the hypothesis.
set -euo pipefail
ROOT=results/dsv4-hugepage-ab
for rep in 1 2; do
  for arm in no-arena-hugepages arena-hugepages; do
    OUT="$ROOT/$arm-$rep"; mkdir -p "$OUT"
    echo "=== $arm rep $rep $(date +%T) ==="
    bash scripts/run_dsv4_decode_arm.sh "$OUT" --vram-fraction 0.95 "--$arm" || true
    echo "EXIT=$?" >> "$OUT/run.log"
  done
done
echo "AB_COMPLETE"
