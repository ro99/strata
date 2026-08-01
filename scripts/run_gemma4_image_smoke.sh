#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_dir=${RESULT_DIR:-"${repo_root}/results/gemma4-image-smoke"}
port=${PORT:-18084}
mkdir -p "${result_dir}"

"${repo_root}/build/strata-server" \
    --model "${repo_root}/models/gemma4" \
    --model-type gemma4 \
    --model-id gemma4 \
    --devices 0,1,2 \
    --context-size 512 \
    --max-new 1 \
    --port "${port}" \
    >"${result_dir}/server.out" \
    2>"${result_dir}/server.log" &
server_pid=$!
cleanup() {
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 90); do
    if curl --silent --fail "http://127.0.0.1:${port}/v1/models" \
        >"${result_dir}/models.json"; then
        break
    fi
    sleep 1
done
curl --silent --fail "http://127.0.0.1:${port}/v1/models" >/dev/null

image='iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAIAAADYYG7QAAAAPklEQVRYw+3OMQEAIBAAIbV/57eF5wAJ2LP+cuqAkJCQUB0QEhISqgNCQkJCdUBISEioDggJCQnVASEhodcuSeEBX2tqyt4AAAAASUVORK5CYII='
body='{"model":"gemma4","messages":[{"role":"user","content":[{"type":"text","text":"What color is this image?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,'"${image}"'"}}]}],"max_tokens":1,"temperature":0}'
curl --silent --fail \
    -H 'Content-Type: application/json' \
    --data-binary "${body}" \
    "http://127.0.0.1:${port}/v1/chat/completions" \
    >"${result_dir}/response.json"

jq . "${result_dir}/response.json" >"${result_dir}/summary.json"
