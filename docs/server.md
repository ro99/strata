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

