# GLM-5.2: operator runbook

GLM-5.2 is a 78-layer, 256-expert W4A16 model. On the reference workstation
the checkpoint exceeds aggregate resident memory, so Strata reports it as
I/O-dependent; caches do not turn that dense storage obligation into sparsity.

## Build and preflight

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel --target strata-chat strata-server

./build-release/strata-chat \
  --model models/glm52 --model-type glm \
  --devices 0,1,2 --context-size 4096 --max-new 256 \
  --vram-fraction 0.85 --dry-run
```

Read the placement report before loading weights. If it reports storage-backed
weights, decode is I/O-dependent by construction; do not interpret cache-hit
warm-up as a steady-state guarantee.

## Chat and server

```bash
./build-release/strata-chat \
  --model models/glm52 --model-type glm \
  --devices 0,1,2 --context-size 4096 --max-new 256

./build-release/strata-server \
  --model models/glm52 --model-type glm \
  --devices 0,1,2 --context-size 4096 --port 8080
```

Use the same device ordering and placement flags for comparisons. GLM's model
token remains `glm` for CLI compatibility; source ownership and filenames use
the unambiguous `glm52` name.
