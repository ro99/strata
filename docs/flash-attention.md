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
- Logical segments are concatenated without materializing attention scores.
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
K/V rows, sinks, causal limits, and outputs before CUDA allocation. Device
buffers grow only to the validated request and are reused. No score tensor is
stored persistently or copied to the host.

## Numerical and failure contract

The request declares one of three numerical contracts; dispatch never changes
it implicitly:

- `tiled_online_f64` is the normal FlashAttention-2 path. The scalar and CUDA
  implementations use a tiled online-softmax recurrence with F64 dot products,
  exponentials, running statistics, and output accumulation before returning
  F32.
- `f64_dot_f32_score_f32_accum` preserves DeepSeek's established scalar order:
  sequential F64 dot, F32 score, global F64 softmax, F32 probability, and F32
  value accumulation. The fused CUDA compatibility specialization recomputes
  QK across bounded passes instead of storing a score tensor.
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
ceilings, and devices. Output is copied into the
caller span only after successful CUDA completion and numerical-status
validation. There is no silent fallback when FlashAttention is explicitly
enabled.

## Current dispatch

Use `--flash-attention` with `strata-deepseek-run`, `strata-run`, or
`strata-chat`. Use `--scalar-attention` in the concrete benchmark runners to
pin the reference path. Operation fixtures and short full-model traces pass,
but the required three-repetition DeepSeek matrix completed with a median
candidate/scalar decode-throughput ratio of `0.520x`; the scalar default is
therefore retained under the rollback condition. The measured result is
recorded in
[`docs/experiments/0014-generic-flash-attention-performance-2026-07-19.md`](experiments/0014-generic-flash-attention-performance-2026-07-19.md).

Detailed CUDA JSON includes FlashAttention calls, launches, H2D/D2H transfers
and bytes, useful/wasted staging bytes, transfer time, kernel time, total
service time, workspace allocation, and synchronization counters for aggregate
and per-device scopes. Indexed gathers stage only selected rows, so wasted
staging is zero for the current adapters.
