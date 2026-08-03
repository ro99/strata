#!/usr/bin/env bash
# First end-to-end Laguna S 2.1-NVFP4 generation.
# Hypothesis: the runtime graph (SWA + per-head softplus gating + QK-norm +
# sigmoid-routed NVFP4 MoE) produces coherent text from the pinned checkpoint.
# Correctness gate: output is fluent, on-topic English, not repetition or noise.
# Memory ceiling: the dry-run plan (19.81/19.81/13.04 GiB device budgets).
# Rollback: revert feat/laguna-s21-nvfp4 if output is incoherent.
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./build/strata-chat \
  --model models/laguna-s-21 \
  --model-type laguna \
  --context-size "${CONTEXT:-512}" \
  --max-new "${MAX_NEW:-24}" \
  --temperature 0 \
  --prompt "${PROMPT:-Write one sentence explaining what a hash map is.}"
