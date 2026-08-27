# Kimi-K3: operator runbook

Kimi-K3 is a 1.45 TiB hybrid KDA/MLA MoE checkpoint. On the reference machine,
SATA traffic is the measured bottleneck by six to eight times: a batch-1 decode
step moves 15.6–20.7 GiB of expert misses at roughly 400 MB/s. The honest rate
is therefore about 0.02 tok/s. See [`../kimi-k3-runtime.md`](../kimi-k3-runtime.md)
for the exact architecture, write guard, correctness gates, and cost model.

## Build and preflight

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel --target strata-chat strata-server

./build-release/strata-chat \
  --model /data/kimi-k3 --model-type kimi-k3 \
  --devices 0,1,2 --context-size 2048 --max-new 64 \
  --vram-fraction 0.85 --dry-run
```

Kimi's write guard fails closed if model-derived pages could reach a protected
NVMe or swap device. Satisfy the memory-lock preflight before attempting a
load; do not disable the guard to make admission pass.

## Run

```bash
./build-release/strata-chat \
  --model /data/kimi-k3 --model-type kimi-k3 \
  --devices 0,1,2 --context-size 2048 --max-new 64 \
  --vram-fraction 0.85 --prompt "Explain mixture-of-experts routing."
```

Keep the process alive if running more than one request so the admitted host
arena can be reused. This does not change the SATA-dependent steady-state
classification when the working set exceeds resident memory.
