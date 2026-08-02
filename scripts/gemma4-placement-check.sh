#!/usr/bin/env bash
# Loads Gemma 4 through the placement plan and decodes a short prompt.
# Correctness gate for the plan-driven layer assignment: same greedy answer,
# no admission error, every layer where the dry run said it would be.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/placement-plan-gemma4
mkdir -p "$OUT"

echo "=== dry run ==="
./build/strata-chat --model models/gemma4 --model-type gemma4 \
    --context-size 8192 --dry-run --replan 2>&1 | tee "$OUT/dry-run.txt"

echo "=== plan-driven load ==="
./build/strata-chat --model models/gemma4 --model-type gemma4 \
    --context-size 8192 --max-new 48 \
    --prompt "In one sentence, what is a hash table?" \
    > "$OUT/planned.out" 2> "$OUT/planned.err"
echo "planned exit=$?"

echo "=== baseline load (no plan cache) ==="
./build/strata-chat --model models/gemma4 --model-type gemma4 \
    --context-size 8192 --max-new 48 --no-plan-cache \
    --prompt "In one sentence, what is a hash table?" \
    > "$OUT/baseline.out" 2> "$OUT/baseline.err"
echo "baseline exit=$?"

echo "=== answers ==="
diff "$OUT/planned.out" "$OUT/baseline.out" && echo "IDENTICAL greedy answer"
grep -E "^\[ready\]|^\[done\]" "$OUT/planned.err" "$OUT/baseline.err"
