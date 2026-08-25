# HTTP server

`strata-server` exposes the runtime over an OpenAI-compatible API. It is a thin
layer on top of the C++ runtime, not a second inference path — an endpoint the
loaded model has no exact implementation for returns an explicit error rather
than a degraded answer.

See [`cli.md`](cli.md) for shared flags, [`sampling.md`](sampling.md) for
sampler semantics.

```bash
./build/strata-server --model models/glm52 --model-type glm \
  --model-id glm52 --context-size 2048 --devices 0,1,2

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm52","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

It serves `/v1/models`, `/v1/health`, `/v1/chat/completions`, `/v1/completions`, and `/v1/tokenize`, with streaming over SSE. An endpoint the loaded model has no exact implementation for returns an explicit error. Measured serving overhead is about 0.04% of a decode step.

For DeepSeek V4 on two GPUs, see
[the rank-local TP2 route](#fastest-deepseek-v4-route-rank-local-tp2) above for
the fastest supported topology and what it requires.

## Model router mode

Instead of one model fixed at startup, `strata-server` can run as a router over
a declarative catalog and spawn an isolated single-model child per selection.
The router itself is GPU-free: it never interprets weights and never substitutes
another catalog entry.

```bash
cat > models.ini <<'EOF'
version = 1

[*]
context-size = 8192
max-new = 256

[alpha]
name = Alpha Writer
model = glm52
model-type = glm
load-on-startup = true

[beta]
name = Beta Editor
model = kimi-k3
model-type = kimi-k3
EOF
./build/strata-server --models-preset models.ini --models-max 1 --port 8080
```

Relative `model` paths resolve against the directory containing the catalog.
Entries accept extra launch arguments verbatim plus optional sampler defaults
(`max-new`, `temperature`, `top-p`, `seed`) that apply when a request does not
set its own value.

Behavior:

- Standard OpenAI requests route by the top-level `model` field; an unloaded
  model autoloads on first use (`--no-models-autoload` rejects instead).
- With `--models-max N` (default 1), the least-recently-used idle child is
  evicted to admit a new one. A child serving a request is never evicted;
  a fully busy slot returns `503` with `all model slots are busy`.
- `/v1/models` lists every catalog entry with `status`
  (`unloaded`/`loading`/`loaded`/`failed`) and its sampler defaults.
- `POST /models/load {"model":"ID"}` and `POST /models/unload {"model":"ID"}`
  manage children explicitly; unloading a busy model returns `409`.
- Children are ordinary `strata-server` processes on loopback ports, so a child
  crash surfaces as `failed` status with its exit code, and streamed responses
  pass through byte-for-byte.

Router-mode steady-state RSS currently includes the full monolithic binary
(~158 MiB mapped) even though no CUDA context is created; shrinking this to a
lightweight dispatcher binary is tracked in
[issue #38](https://github.com/ro99/strata/issues/38).

