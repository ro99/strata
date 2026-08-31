# Strata

**A C++20/CUDA inference engine for models that don't fit in your VRAM — or your RAM.**

Strata splits one checkpoint across GPU VRAM, host RAM, and read-only NVMe, and
runs it at the precision, router semantics, expert count and top-k the
checkpoint actually declares. No Python and no ML framework in the runtime:
Strata is a C++ application with command-line and HTTP entry points.

## The problem it solves

Most people who want to run a frontier model don't have a rack of data-center
GPUs. They have one or a few consumer cards and a reasonable amount of system
RAM. Fitting a large model into that is easy if you're willing to degrade it —
drop below four bits, cut experts, reduce top-k, or silently fall back to a
cheaper path. Strata is built against doing that:

- **Never below four bits.** Not in weights, caches, KV storage, drafts, or
  predictors.
- **Precision, routing, expert count and top-k come from the checkpoint** and
  never change silently.
- **A predictor may only reorder work**, never change what is computed.
- **Exact mode completes or fails.** It does not quietly substitute something
  cheaper.
- **A model too large for combined memory is reported as I/O-dependent**, not
  hidden behind a cache.

These are enforced in code. It's also why some numbers here are unglamorous:
the honest figure for a checkpoint that doesn't fit in memory is a slow one.

## Quick start

```bash
make check                       # build and run the test suite
./build/strata-chat --model /path/to/checkpoint --model-type gemma4
```

`--model-type` is one of `gemma4`, `deepseek`, `glm`, `glm53`, `laguna`,
`inkling`, `kimi-k3`. Add `--devices 0,1` to pin GPUs, `--admission-only --json` to print
the placement plan without loading anything.

There is also an OpenAI-compatible server (`strata-server`).
**Full flag reference: [`docs/cli.md`](docs/cli.md).**

## Supported models

| Model | Layout | Native precision | On a 64 GiB-VRAM / 251 GiB-RAM workstation |
|---|---|---|---|
| **Gemma 4 31B-IT** | 60 dense hybrid-attention layers + 27-layer vision tower | INT8 group-32 or MXFP4 text, BF16 vision/KV | Fully resident in VRAM; text and image input. **668.99 tok/s at 128-token prefill; 29.747 tok/s decode** |
| **DeepSeek-V4-Flash-0731** | 43 layers, 256 experts, top-6 | FP4 E2M1 experts, FP8 E4M3 spine | Resident in RAM; zero checkpoint reads during decode |
| **Laguna S 2.1** | 48 layers 1:3 global/sliding, 256 experts + 1 shared, top-10 | NVFP4 or MXFP4 experts, BF16 elsewhere | Spine in VRAM; experts stream from RAM |
| **Inkling Small** | 42 layers, 256 experts + 2 sinks, top-6, no rotary | NVFP4 or MXFP4 experts, BF16 elsewhere | Experts stream from RAM |
| **GLM-5.2** | 78 layers, 256 experts, top-8 | INT4 group-128, W4A16 | Exceeds combined memory; I/O-dependent |
| **GLM-5.3-Flash** | 45 layers, 3 KDA : 1 sparse MLA, 288 experts + 1 shared, top-8 | FP8 E4M3 block-128, or MXFP4 routed experts with BF16 corrections | Text-only; automatic release selection, exact through 2,048 tokens; **3.88 tok/s MXFP4 decode** |
| **Kimi-K3** | 93 layers, 3 KDA : 1 gated MLA, 896 experts, top-16 | MXFP4 experts, BF16 elsewhere | 1.45 TB; I/O-dependent, 38.6 s/step. Vision not implemented |

Each runs its declared semantics as-is — hybrid compressed attention and
manifold-constrained hyper-connections for DeepSeek, Kimi Delta Attention and
latent-space MoE for Kimi-K3, per-head softplus output gating and YaRN-on-global
rotary for Laguna, and so on.

Each model runbook states the hardware and operating point for its measured
figures. A number from one context length does not transfer to another.

## Precision, as declared

Strata reads a quantized checkpoint in the representation it ships in. There is
no requantization pass, and no second copy of the weights in a wider format.

- **MXFP4 is read as MXFP4.** `mxfp4-pack-quantized` — two E2M1 codes per byte,
  one E8M0 exponent per group of 32 — is decoded from the bytes on disk. No
  FP4-to-INT4 conversion, no FP4-to-BF16 expansion held resident beside it.
- **Mixed representations stay mixed.** DeepSeek-V4-Flash-0731 runs its FP4
  E2M1 experts and its FP8 E4M3 spine as published, in one model, without
  normalizing either format to the other.
- **Four bits is the floor**, and it is enforced in code rather than promised in
  a README — weights, caches, KV storage, drafts and predictors alike.
- **Scales are admitted, not clamped.** E8M0 codes 0 and 255 decode to `+0` and
  `+inf` in a BF16 exponent field, so a checkpoint carrying them fails admission
  at load instead of producing a wrong value at decode.

### FP4 on Ampere, from software

Ada and Blackwell get FP4 in silicon. Ampere does not: SM86 has no FP4 tensor
op, and its `mma.sync` takes BF16 operands.

Strata has an SM86 kernel that decodes E2M1 codes and E8M0 group-32 scales
directly into MMA operand registers with `PRMT` and a lookup table. The codes
stay four bits in HBM; the prepack into `m16n8k16` fragment order is a pure
permutation of the same byte count, so no widened copy is ever held and the
weight never occupies more memory than the checkpoint gave it.

Measured on one RTX 3090 at batch 1, on the two DeepSeek V4 production shapes:
**742.9 and 749.9 GB/s, about 88% of that card's measured read ceiling**, with
the four-bit decode itself costing no measurable time — the kernel is bound by
reading the weights, which is the floor for any weight-stationary matmul.

On **Gemma 4 31B-IT**, whose 19.5 GB MXFP4 checkpoint is fully resident on one
24 GB card, the first register-fed substitution made decode **3.367x faster**
than the scalar route. The current runtime uses one FP32-output Marlin layout
for both M=128 pages and M=1 decode: it reaches **668.99 prefill tok/s** at 128
tokens and **29.747 steady decode tok/s**. Experiments 0165 and 0186 contain the
respective arms, spreads, and correctness gates.

### When it helps, and when it does not

The kernel reads weights faster. That only moves the wall clock if reading
weights is what the step is spending its time on, and whether it is depends on
whether the weights fit on the card:

| Workload | `argmax_r` of a decode step | GPU matmul share | Result |
|---|---|---:|---|
| Gemma 4 — dense, fully resident | GPU kernel / HBM service | 99.1% | **32.18x M=128 page; 29.747 decode tok/s** |
| Laguna S 2.1 — MoE, 63.7 GiB | routed-expert staging | 6.2% | not built; ceiling 1.066x |
| Inkling Small — MoE | cache-miss staging / H2D | 5.5% | not built; ceiling ~1.06x |
| DeepSeek V4 — MoE, 147 GB experts | host-side expert compute | ~2% | measured 0.98x |

For a model whose experts stream from RAM, decode time is moving weights across
PCIe, not multiplying them. Two things then work against the substitution: the
fragment prepack is a per-staging cost that lands on the bottleneck term rather
than a one-off at load, and at batch 1 across many small per-expert matrices the
dispatch is launch-bound rather than bandwidth-bound — DeepSeek V4 measured
2.027 ms against 2.038 ms of device MoE kernel time, which is no difference at
all.

So the register-fed route is opt-in per model rather than global, and the
streaming-MoE models keep the scalar MXFP4 path. Experiments 0164, 0166 and 0167
record those as negatives with their measurements; 0165 records the positive.

## Documentation

| | |
|---|---|
| [`docs/cli.md`](docs/cli.md) | every flag and the dry-run planner |
| [`docs/sampling.md`](docs/sampling.md) | sampler semantics and reproducibility |
| [`docs/server.md`](docs/server.md) | the OpenAI-compatible HTTP API |
| [`docs/models/`](docs/models/) | copy-paste build, chat, server, and measured-speed runbooks by model |
| [`docs/current-architecture.md`](docs/current-architecture.md) | how the code is organised and what is enforced |
| [`docs/model-bringup-guide.md`](docs/model-bringup-guide.md) | adding another model |
| [`docs/architecture.md`](docs/architecture.md) | the target scheduler design, not yet built |
| [`docs/README.md`](docs/README.md) | product documentation index |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | repository layout, architecture, and change hygiene rules |

## Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before changing the repository.
Start from a measured bottleneck using `τ = max_r W_r/B_r + Σ_serial`, not from
a guess about what's slow. State a hypothesis and a kill criterion before
building, run `make check` before claiming a result, and don't call something a
win if it's inside run-to-run variance. Keep accepted and rejected research
records in the local Git-ignored experimentation workspace. The headers under
`include/strata/` are internal application interfaces, not an installed SDK or
a stable C++ ABI.

## License

Apache-2.0.
