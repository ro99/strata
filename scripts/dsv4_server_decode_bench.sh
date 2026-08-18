#!/usr/bin/env bash
# Repeatable decode baseline against a running strata server.
# Usage: dsv4_server_decode_bench.sh PORT LABEL [REPS]
set -euo pipefail
PORT="${1:?port}"; LABEL="${2:?label}"; REPS="${3:-3}"
BODY='{"model":"strata-deepseek-v4","messages":[{"role":"user","content":"Explain in detail how a tiered memory hierarchy serves a mixture-of-experts model whose weights exceed local VRAM."}],"max_tokens":64,"temperature":0}'
for i in $(seq 1 "$REPS"); do
  curl -s -m 900 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$BODY" \
    | python3 scripts/dsv4_timings.py "$LABEL" "$i"
done
