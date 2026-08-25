#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: check_model_router.sh STRATA_SERVER TEST_CHILD" >&2
    exit 2
fi

server=$1
child=$2
test_root=$(mktemp -d)
port=$((20000 + $$ % 20000))
server_pid=

cleanup() {
    if [[ -n ${server_pid} ]]; then
        kill -TERM "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
    rm -rf "${test_root}"
}
trap cleanup EXIT

mkdir -p "${test_root}/alpha" "${test_root}/beta"
cat >"${test_root}/models.ini" <<EOF
version = 1

[*]
context-size = 128
max-new = 8
temperature = 0.7

[alpha]
name = Alpha Writer
model = ${test_root}/alpha
model-type = glm

[beta]
name = Beta Editor
model = ${test_root}/beta
model-type = gemma4
temperature = 0.4
EOF

STRATA_ROUTER_CHILD_EXECUTABLE="${child}" "${server}" \
    --models-preset "${test_root}/models.ini" --models-max 1 --port "${port}" \
    >"${test_root}/server.out" 2>"${test_root}/server.log" &
server_pid=$!

for _ in $(seq 1 100); do
    if curl -fsS "http://127.0.0.1:${port}/v1/health" >"${test_root}/health.json"; then
        break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        cat "${test_root}/server.log" >&2
        exit 1
    fi
    sleep 0.05
done

jq -e '.role == "router"' "${test_root}/health.json" >/dev/null
rss_kb=$(ps -o rss= -p "${server_pid}" | tr -d ' ')
[[ ${rss_kb} -lt 196608 ]]

curl -fsS "http://127.0.0.1:${port}/v1/models" >"${test_root}/models-before.json"
jq -e '.data | length == 2 and all(.status.value == "unloaded")' \
    "${test_root}/models-before.json" >/dev/null
jq -e '.data[] | select(.id == "alpha") | .name == "Alpha Writer" and .defaults.temperature == 0.7' \
    "${test_root}/models-before.json" >/dev/null

curl -fsS -N -H 'Content-Type: application/json' \
    -d '{"model":"alpha","messages":[{"role":"user","content":"hello"}],"stream":true}' \
    "http://127.0.0.1:${port}/v1/chat/completions" >"${test_root}/alpha.sse"
grep -q 'alpha ready' "${test_root}/alpha.sse"
grep -q '^data: \[DONE\]$' "${test_root}/alpha.sse"

curl -fsS -N -H 'Content-Type: application/json' \
    -d '{"model":"beta","messages":[{"role":"user","content":"hello"}],"stream":true}' \
    "http://127.0.0.1:${port}/v1/chat/completions" >"${test_root}/beta.sse"
grep -q 'beta ready' "${test_root}/beta.sse"

curl -fsS "http://127.0.0.1:${port}/v1/models" >"${test_root}/models-after.json"
jq -e '.data[] | select(.id == "alpha") | .status.value == "unloaded"' \
    "${test_root}/models-after.json" >/dev/null
jq -e '.data[] | select(.id == "beta") | .status.value == "loaded"' \
    "${test_root}/models-after.json" >/dev/null

curl -fsS -H 'Content-Type: application/json' -d '{"model":"beta"}' \
    "http://127.0.0.1:${port}/models/unload" | jq -e '.success == true' >/dev/null
curl -fsS "http://127.0.0.1:${port}/v1/models" | \
    jq -e '.data | all(.status.value == "unloaded")' >/dev/null

kill -TERM "${server_pid}"
wait "${server_pid}"
server_pid=
