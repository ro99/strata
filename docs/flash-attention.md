# Generic FlashAttention forward backend

Strata provides a ground-up, dependency-free CUDA forward-attention primitive
behind the model-neutral contract in `include/strata/attention.hpp`. It is
opt-in while the pinned full-model correctness and replicated performance gates
remain under evaluation.

## Ownership boundary

The shared primitive computes scaled `QK^T`, stable tiled online softmax, and
the weighted `V` accumulation. It does not choose sparse positions or change a
model graph. Architecture adapters retain ownership of projections,
normalization, RoPE/YaRN, compression, indexing, cache commit, and declared
rounding boundaries.

DeepSeek supplies two logical regions: the exact 128-token ring window and the
declared compressed history (all rows or the learned top-512 rows). Attention
sinks are virtual normalization terms with zero values. The adapter applies the
declared BF16 output boundary and inverse RoPE after the CUDA result.

GLM supplies its per-head non-RoPE key concatenated with the shared RoPE key,
per-head values, and a causal visible-row count for every query. This covers the
currently admitted full-attention region through 2,048 tokens. DSA selection and
IndexShare remain model operations; when the long-context adapter is enabled,
its exact selected rows can use the same indexed-segment contract.

## Supported tensor contract

- Queries: `[query_rows, query_heads, query_key_dim]` in F32 host storage.
- Segment keys: `[source_rows, key_value_heads, query_key_dim]`.
- Segment values: `[source_rows, key_value_heads, value_dim]`.
- Empty value storage aliases keys when `query_key_dim == value_dim`.
- Empty row indices select every source row in storage order; otherwise indices
  gather exact source rows in descriptor order.
- Logical segments are packed without a persistent score tensor or host score
  transfer. The DeepSeek exact-compatibility specialization uses bounded
  transient device score/probability scratch for one decode query.
- `query_heads` must be divisible by `key_value_heads`, covering MHA, GQA, and
  MQA/shared-KV mappings.
- Optional causal counts select a visible prefix of the concatenated rows for
  each query.
- Optional finite per-head sinks contribute to the softmax denominator with a
  zero value numerator.
- CUDA dimensions are bounded to 1,024 for Q/K and V, query rows to 65,535 per
  call, and logical rows to the U32 range.
- CUDA support is explicit for the repository targets SM86 and SM120. Other
  devices fail with an unsupported error when the backend is requested.

The descriptor carries a byte ceiling. Validation accounts queries, gathered
K/V rows, sinks, causal limits, outputs, numerical status, and exact-contract
score scratch before CUDA allocation. Device buffers grow geometrically within
that ceiling and are reused. Pinned staging combines each request into one H2D
and one D2H transfer. No score tensor is stored persistently or copied to the
host.

## Numerical and failure contract

The request declares one of three numerical contracts; dispatch never changes
it implicitly:

- `tiled_online_f64` is the normal FlashAttention-2 path. The scalar and CUDA
  implementations use a tiled online-softmax recurrence with F64 dot products,
  exponentials, running statistics, and output accumulation before returning
  F32.
- `f64_dot_f32_score_f32_accum` preserves DeepSeek's established scalar order:
  sequential F64 dot, F32 score, global F64 softmax, F32 probability, and F32
  value accumulation. One thread computes each row's sequential dot, thread
  zero preserves the ordered softmax reduction, and value lanes preserve the
  original row accumulation order using bounded transient scratch.
- `f32_dot_f32_softmax_f32_accum` similarly preserves GLM's all-F32 dot,
  softmax, probability, and value-accumulation contract. Cross-device
  libm/libdevice exponential differences are accepted up to `4e-6` absolute at
  the routed-coefficient gate; selected experts must still match exactly.

Model adapters apply their existing RoPE and BF16 boundaries unchanged. The
compatibility paths deliberately trade some arithmetic throughput for exact
model behavior; they still eliminate host score buffers and host per-head
attention loops.

CUDA reports non-finite scores, invalid denominators, and non-finite output as
errors. It also rejects incompatible shapes, indices, head mappings, workspace
ceilings, and devices. Explicit CUDA enablement validates the device during
runtime initialization, even when a shape-aware adapter retains its scalar
kernel below a declared crossover. Output is copied into the caller span only
after successful CUDA completion and numerical-status validation. There is no
error-triggered or numerical fallback.

## Current dispatch

Use `--flash-attention` with `strata-deepseek-run`, `strata-run`, or
`strata-chat`. Use `--scalar-attention` in the concrete benchmark runners to
pin the reference path. DeepSeek's production adapter keeps its 28-worker
scalar kernel below 256 logical KV rows and dispatches CUDA at or above that
measured crossover. `--flash-attention-minimum-rows 0` forces CUDA for
diagnostics. JSON graph metrics report `attention_cuda_dispatches` and
`attention_scalar_dispatches` independently.

The original unconditional implementation failed promotion at `0.520x`
median short-context decode throughput. The production decode redesign and its
replacement gates are recorded in
[`docs/experiments/0015-production-flash-decode-2026-07-19.md`](experiments/0015-production-flash-decode-2026-07-19.md); the replicated gate shows
attention/prefill improvement but no end-to-end decode win within observed
variance, so the scalar backend remains the global default. Use
`--flash-attention` for the shape-aware hybrid policy.

Detailed CUDA JSON includes FlashAttention calls, launches, H2D/D2H transfers
and bytes, useful/wasted staging bytes, transfer time, kernel time, total
service time, workspace allocation, and synchronization counters for aggregate
and per-device scopes. Indexed gathers stage only selected rows, so wasted
staging is zero for the current adapters.
