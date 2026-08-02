#!/usr/bin/env bash
# Confirms the placement plan resolution does not disturb the GLM runtime.
# GLM's plan is descriptive: the load must behave exactly as before.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/placement-plan-glm
mkdir -p "$OUT"
./build/strata-chat --model models/glm52 --model-type glm \
    --context-size 2048 --max-new 4 \
    --prompt "In one sentence, what is a hash table?" \
    > "$OUT/planned.out" 2> "$OUT/planned.err"
echo "exit=$?"
tail -3 "$OUT/planned.out"
grep -E "^\[ready\]|^\[done\]|^\[placement\]" "$OUT/planned.err"
