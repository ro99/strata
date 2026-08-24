#!/usr/bin/env bash
# Decode/prefill arm against a running strata-server, saving the raw response
# JSON of every rep so greedy output can be compared byte-for-byte across arms.
# Usage: dsv4_decode15_bench.sh PORT LABEL REPS OUTDIR [PROMPT_FILE] [MAX_TOKENS]
set -euo pipefail
PORT="${1:?port}"; LABEL="${2:?label}"; REPS="${3:-3}"
OUTDIR="${4:?outdir}"; PROMPT_FILE="${5:-}"; MAX_TOKENS="${6:-64}"
mkdir -p "$OUTDIR"
if [ -n "$PROMPT_FILE" ]; then
  BODY=$(python3 -c '
import json,sys
print(json.dumps({"model":"strata-deepseek-v4","messages":[{"role":"user","content":open(sys.argv[1]).read()}],"max_tokens":int(sys.argv[2]),"temperature":0}))' "$PROMPT_FILE" "$MAX_TOKENS")
else
  BODY="{\"model\":\"strata-deepseek-v4\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain in detail how a tiered memory hierarchy serves a mixture-of-experts model whose weights exceed local VRAM.\"}],\"max_tokens\":${MAX_TOKENS},\"temperature\":0}"
fi
for i in $(seq 1 "$REPS"); do
  curl -s -m 1800 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$BODY" > "$OUTDIR/${LABEL}-rep${i}.json"
  python3 scripts/dsv4_timings.py "$LABEL" "$i" < "$OUTDIR/${LABEL}-rep${i}.json"
done
