#!/usr/bin/env bash
# Decode arm past the sequential-driver band (decode position > ~2052), so the
# queued/chain driver that production actually uses is the one measured.
set -euo pipefail
OUT="${1:?usage: OUTDIR [flags...]}"; shift
mkdir -p "$OUT"
exec ./build-release/strata-deepseek-run \
  --model models/dsv4f \
  --prompt "$(cat /home/rodrigo/.claude/jobs/4bfde57a/tmp/prompt2600.txt)" \
  --max-new 64 --devices 1,2 --decode-topology rank-local-tp2 \
  --device-resident-runtime --flash-attention \
  --prefill-page-tokens 8192 --max-context 16384 --vram-fraction 0.95 \
  --detailed-timing --json "$@" > "$OUT/generation.json" 2> "$OUT/run.log"
