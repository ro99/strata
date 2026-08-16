# Experiment 0100 — DSV4 page projection residency screen

Status: **re-scoped to attribution after the required source and counter audit
falsified the original dispatch-collapse premise.** The original mechanism was
not implemented. Experiment 0099 remains preserved at `abded32` and is not
relitigated or rolled back.

## Proposed contract

The proposed mechanism was to batch the dense query and KV projections over a
prefill page, keep intermediate activations device-resident within the layer,
and leave decode untouched. The stated structural gate required total CUDA
`matmul_calls` and `synchronization_calls` to fall by roughly an order of
magnitude. The value metric was total 677-token prefill at page 8192, from the
0099 point of 60.195 s / 11.247 tok/s, with a keep threshold of at most 55 s /
at least 12.3 tok/s. Correctness required bit-exact page versus row projections,
generated IDs `2107, 8777, 1277, 440`, zero decode checkpoint reads,
unregressed decode, and the existing 384 MiB per-attention-device workspace
ceiling.

The measured target was the 28.794 s attention term, specifically 10.338 s of
query projection, 3.908 s of KV projection, and any same-shaped work in the
6.234 s paged-attention backend host remainder. The intended resource reduction
was host dispatch and synchronization, not activation-transfer volume: the
entire prefill reported only 0.894 s activation H2D and 1.214 s activation D2H.
The expected signs were fewer dispatches and synchronizations, unchanged
projection arithmetic and weight bytes, reduced or unchanged activation bytes,
unchanged MoE work, unchanged exact semantics, and no workspace growth beyond
384 MiB per attention device.

## Cheapest falsifying audit

The required source audit found that bounded page projection batching was
already implemented and accepted in experiment 0016. In
`DeepSeekV4Runtime::Impl::attention_page`, each page invokes:

- `linear_rows(..., "wq_a", ..., rows, ...)` once;
- `linear_rows(..., "wq_b", ..., rows, ...)` once;
- `linear_rows(..., "wkv", ..., rows, ...)` once.

`linear_rows` forwards the complete `[rows, columns]` activation to one
`Dsv4WeightCache::matmul` call. The CUDA backend then uses
`bf16_matvec_rows_kernel` for multi-row BF16 weights or the row-tiled
`plain_matmul_kernel`; it does not loop over rows on the host. Git blame traces
this page-major projection path to accepted commit `4e93545`.

The 0099 counters independently prove the same shape:

| Counter / derivation | Value | Reading |
|---|---:|---|
| Prefill pages | 2 | one 676-row page plus one single-row page |
| Core query/KV projected rows | `3 * 43 * 677` = 87,333 | unchanged arithmetic |
| Core query/KV page calls | `3 * 43 * 2` = 258 | already page-batched |
| Reported attention projection rows | 87,415 | core rows plus 82 single-row compressor rows |
| Reported attention projection calls | 340 | core calls plus the same 82 compressor calls |
| Calls if the core were per-row | about 87,333 | contradicted by the measured 340 |

Experiment 0016 measured this exact transition at a 2,054-token operating
point: page 1 to page 64 reduced projection calls from 264,966 to 4,257
(-98.4%), query time by 37.9%, KV time by 30.4%, and total prefill by 8.4%.
The dispatch-collapse mechanism therefore landed previously; the remaining
query/KV time is a residual after that collapse, not evidence that batching is
absent.

## Why the proposed global structural gate is invalid

CUDA `matmul_calls` is a semantic-work counter, not a kernel-dispatch counter.
The page-batched routed-expert backend explicitly adds
`3 * (expert groups + shared expert)` to it even though those experts execute
inside a small number of page commands. At 0099 the complete prefill reported:

- 100,721 semantic `matmul_calls`;
- only 340 attention-projection calls;
- 129 DeepSeek MoE commands and 688 MoE kernel launches;
- 58,899 mHC commands and 176,020 mHC kernel launches;
- 143,995 global synchronization calls.

Removing every reported attention-projection call could reduce global
`matmul_calls` by at most 0.34%, not an order of magnitude. The three core
page projections account for only 258 synchronous generic matmul commands;
even deleting all of their synchronization boundaries could reduce the global
synchronization count by at most 0.18%. A `wq_a -> q_norm -> wq_b` device chain
could at most merge a page-local boundary after adding an exact device norm;
`wkv` is already one projection and has no second KV projection to collapse in
Strata's declared V4 layout. That is materially different from the proposed
per-row-to-per-page mechanism and cannot satisfy the stated structural gate.

The reference stack confirms the desired tensor shape but does not expose a
missing Strata row batch: `DeepseekAttention.forward` applies `q_a_proj`,
`q_b_proj`, and `kv_a_proj_with_mqa` directly to `hidden_states`, whose leading
token dimension is preserved by each linear. Strata's `linear_rows` calls
already have that same leading page dimension.

The prior `exp/dsv4-device-activations` branch is reusable only as a bounded,
generation-checked two-buffer command pattern. Experiment 0018 chained the two
attention output projections during decode, removed one host round trip and 43
synchronizations per step, remained exact, and produced no material throughput
win. It does not supply evidence that the current 10.338 s query bucket is
dispatch-bound, and its ABI predates the current page attention backend.

## Decision point

The brief requires stopping when its premise appears wrong. Implementing the
requested mechanism would duplicate accepted page batching while the stated
structural counters could not move by the required order of magnitude. No code,
fixture, build-specific checkpoint, baseline arm, candidate arm, repetition,
2,612-token run, or fixed/marginal fit was started.

The value metric change remains accepted for future work: total 677-token
prefill seconds and tok/s are the decision metric, and a structural partial win
must be preserved and discussed rather than automatically discarded. Before a
new mechanism is selected, the unresolved 10.338 s query and 3.908 s KV buckets
need attribution between batched kernel execution, host normalization/RoPE,
matmul issue/finish, and the three page-local synchronization waits. That would
be a new measurement hypothesis and needs an explicit revised contract because
it no longer targets per-row dispatch collapse.

Raw evidence is the preserved 0099 JSON under
`results/dsv4-0099-prepared-selection-gate/candidate.json`, accepted experiment
0016, rejected experiment 0018, current runtime/backend source, and the local
Lvllmds4-x checkout. The pre-implementation `make check` passed 2/2 tests.

## Authorized attribution revision

The user accepted the falsification and authorized continuing experiment 0100
as measurement only. There is no candidate mechanism and no value gate. The
deliverable is one untraced 677-token, page-8192 arm that attributes:

- the 10.3379 s query and 3.9079 s KV buckets among batched matmul device
  execution, host RMS/RoPE, matmul issue, matmul finish and exact stream wait;
- the prior 143,995 synchronization calls and 10.599 s critical-path recorded
  synchronization time by attention, projection, mHC, MoE, weight and other
  subsystems;
- mHC device, host and synchronization time after preventing a negative CUDA
  event interval from wrapping to approximately 18.4 billion seconds.

Success requires every counter to be traceable to a source site and no query,
KV or synchronization residual larger than approximately 1 s to be silently
distributed. Any larger residual must be reported explicitly and sized.
Instrumentation must preserve the executed graph and the 384 MiB/device page
attention workspace ceiling.

The instrumentation adds no commands or stream synchronization. Generic
matmul reports its existing issue, stream-wait, pinned-output-finish and CUDA
event intervals to the page caller. Query RMS and RoPE CPU work are timed
inside the existing per-row worker task; KV norm and RoPE retain their existing
boundaries. Every existing backend synchronization site is assigned exactly
one subsystem, and per-device category sums are required to reproduce the
historical global total.

Source inspection found a necessary reporting distinction: generic matmul's
historical `synchronization_nanoseconds` stops after the pinned output is copied
to the caller, so it contains both the true `cudaStreamSynchronize` wait and
host finish. That historical definition is preserved so the prior 10.599 s can
be attributed without changing the baseline metric. New projection-specific
counters separately stop at stream completion and after host finish.

The mHC fix converts only finite, non-negative CUDA event intervals to unsigned
nanoseconds. Invalid intervals become zero and increment
`dsv4_mhc_timing_clamped_samples`; they can no longer silently wrap. Direct mHC
commands additionally report device event time, host-exclusive call time and
their already-existing synchronization wait.

Arm budget: one untraced 677-token page-8192 arm, approximately four minutes of
model time, with no baseline, repetition, 2,612-token arm or fixed/marginal fit.
The arm is not authorized until the build, focused CUDA fixtures, full
`make check`, experiment-record amendment and instrumentation commit pass.

Instrumentation checkpoint: the focused native CUDA fixture suite passed, and
the required full `make check` passed 2/2 CTest targets before commit.
