#!/usr/bin/env bash
# Decode-throughput arm for the routed-expert VRAM cache work.
# Short prompt, many generated tokens: decode is the hypothesis, so prefill is
# minimised rather than inherited from a prefill-shaped script.
set -euo pipefail
OUT="${1:?usage: run_dsv4_decode_arm.sh OUTDIR [extra flags...]}"; shift
mkdir -p "$OUT"
PROMPT="Explain how a tiered memory hierarchy serves a mixture-of-experts model whose weights exceed local VRAM, covering residency, admission and the cost of a cache miss."
exec ./build-release/strata-deepseek-run \
  --model models/dsv4f \
  --prompt "$PROMPT" \
  --max-new 64 \
  --devices 1,2 \
  --decode-topology rank-local-tp2 \
  --device-resident-runtime \
  --flash-attention \
  --prefill-page-tokens 8192 \
  --max-context 16384 \
  --detailed-timing --json "$@" \
  > "$OUT/generation.json" 2> "$OUT/run.log"
