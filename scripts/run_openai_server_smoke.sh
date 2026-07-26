#!/usr/bin/env bash
set -euo pipefail

result_dir=${RESULT_DIR:-results/openai-server-smoke}
port=${PORT:-18080}
mkdir -p "$result_dir"

build/strata-server \
  --model models/glm52 --model-type glm --model-id glm52 \
  --devices 0,1,2 --context-size 2048 --max-new 3 --port "$port" \
  >"$result_dir/server.stdout" 2>"$result_dir/server.log" &
server_pid=$!
trap 'kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT

for _ in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:$port/v1/health" \
      >"$result_dir/health.json" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid"
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:$port/v1/models" >"$result_dir/models.json"

request='{"model":"glm52","messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"Say hello."}],"temperature":1,"top_p":0.9,"max_tokens":3,"stop":"NEVER_EMIT_THIS","seed":17}'
for repetition in 1 2 3; do
  curl -fsS -H 'Content-Type: application/json' -d "$request" \
    "http://127.0.0.1:$port/v1/chat/completions" \
    >"$result_dir/response-$repetition.json"
  jq -e '.object == "chat.completion" and .choices[0].message.role == "assistant" and .usage.total_tokens > 0' \
    "$result_dir/response-$repetition.json" >/dev/null
done
jq -e -s '.[0].choices[0].message.content == .[1].choices[0].message.content and .[1].choices[0].message.content == .[2].choices[0].message.content' \
  "$result_dir"/response-{1,2,3}.json >/dev/null

curl -fsS -N -H 'Content-Type: application/json' \
  -d '{"model":"glm52","messages":[{"role":"user","content":"Say hello."}],"temperature":0,"max_tokens":3,"stream":true}' \
  "http://127.0.0.1:$port/v1/chat/completions" >"$result_dir/stream.txt"
grep -q '^data: \[DONE\]$' "$result_dir/stream.txt"

kill -TERM "$server_pid"
wait "$server_pid"
trap - EXIT
