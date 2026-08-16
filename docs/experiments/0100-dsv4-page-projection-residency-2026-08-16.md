# Experiment 0100 — DSV4 page projection residency screen

Status: **attribution arm complete after the required source and counter audit
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

## Attribution result

The sole authorized arm ran untraced at 677 tokens and page 8192 from
instrumentation commit `7c5fd1d`. It completed successfully in 2:52.89 process
wall time, below the approximately four-minute model-time budget. Raw output is
preserved under `results/dsv4-0100-attribution/`.

Correctness and operating-point checks passed:

- generated IDs were exactly `2107, 8777, 1277, 440`;
- `decode_checkpoint_read_bytes` was zero;
- decode was 6.773 tok/s versus 0099's one-arm 6.819 tok/s, a 0.7% difference
  with no material regression claim from one run;
- page attention retained 86 calls, 1,655 launches and 30,564,224 page bytes;
- RSS was 158,858,625,024 bytes (147.949 GiB), and GPU 1/GPU 2 VRAM was
  22,994,354,176 / 22,916,759,552 bytes (21.415 / 21.343 GiB), exactly the
  same VRAM byte counts as 0099;
- the maximum page workspace was 44,302,336 bytes, below the declared
  384 MiB/device ceiling.

This was instrumentation, not an A/B candidate. Total prefill was 64.3285 s or
10.5241 tok/s. The 0099 arm was 60.1949 s or 11.2468 tok/s, but no timing claim
is made from that non-interleaved comparison; the current arm's phase split was
29.9014 s attention, 30.0876 s MoE and 2.8978 s graph mHC pre+post.

### Query attribution

The 10.634712 s query bucket is accounted to 0.015787 s:

| Query component | Seconds | Interpretation |
|---|---:|---|
| page vector allocation/zeroing | 2.337584 | host allocation and initialization before projection |
| weight acquisition | 0.000579 | demand guard and resident-weight lookup |
| matmul host issue | 7.298632 | submission interval; device work overlaps it |
| exact stream wait | 0.704693 | time inside `cudaStreamSynchronize` only |
| host finish | 0.022637 | pinned output copy plus CUDA event reads |
| query-rank RMS norm wall | 0.023763 | host wall time |
| query RMS/RoPE worker wall | 0.231037 | existing page worker dispatch |
| unexplained residual | 0.015787 | subtraction residual |

CUDA event service within the matmuls was 0.088536 s H2D, 7.159487 s kernel
and 0.651140 s D2H, or 7.899163 s total. Those intervals overlap host issue and
stream wait and therefore must not be added to the wall decomposition. The
worker tasks accumulated 4.485757 CPU-seconds of RMS and 1.500354 CPU-seconds
of RoPE across threads while consuming only 0.231037 s wall time.

The large finding is that the query residual is not dispatch count or PCIe:
the batched kernel itself consumes 7.159 s of device service, and repeatedly
allocating/zeroing page-sized host vectors consumes another 2.338 s.

### KV attribution

The projection/normalization portion accounts for only 0.601260 s of the
4.360476 s historical KV bucket:

| KV component | Seconds |
|---|---:|
| page vector allocation/zeroing | 0.025732 |
| weight acquisition | 0.000293 |
| matmul host issue | 0.128879 |
| exact stream wait | 0.406586 |
| host finish | 0.010974 |
| host norm | 0.012576 |
| host RoPE/rounding | 0.016221 |
| sliding-KV append plus learned compressor | 3.759216 |

CUDA event service for the actual KV projection was 0.069887 s H2D,
0.324564 s kernel and 0.015820 s D2H. Source inspection explains the 3.759216 s
remainder: `attention_append_prepared` deliberately adds sliding-cache append
and `compressor(...)` to `attention_kv_nanoseconds` for every row after the
page projection. The compressor includes its own two row projections, pooling,
norm, RoPE and cache publication. The arm does not split those operations
internally, so 3.759216 s is the explicitly sized unresolved sub-split; it is
not assigned to the batched KV projection.

### Synchronization attribution

The subsystem counters reconcile exactly on the critical device:

| Subsystem | Calls | Critical-path recorded seconds |
|---|---:|---:|
| weights | 170 | 0.057705 |
| attention | 43 | 3.941541 |
| generic projections | 84,630 | 1.598877 |
| mHC | 58,813 | 0.789410 |
| MoE | 87 | 4.176965 |
| other | 252 | 0.002243 |
| **total** | **143,995** | **10.566741** |

Thus the synchronization term is principally MoE collect/wait (4.177 s),
attention (3.942 s), generic matmul (1.599 s), and mHC (0.789 s). It is not a
projection-dispatch-count problem.

Generic matmul's historical projection synchronization value includes its
pinned output copy. On critical GPU 1, exact generic-matmul host finish over all
41,978 calls was only 0.133939 s. Therefore at least 10.432802 s of the
historical 10.566741 s global total was genuine stream wait; host finish could
account for at most 0.133939 s (1.3%). The exact query+KV subset separately
measured 1.111278 s stream wait and 0.033611 s host finish. A global exact
matmul-wait accumulator was not persisted, so the remaining global split is a
0.133939 s bound rather than an invented point estimate.

### mHC attribution and counter correction

mHC reported 58,899 commands and 176,020 launches. On the critical device its
event service was 1.693445 s: 0.248781 s H2D, 1.054103 s kernels and 0.390561 s
D2H. Host-exclusive backend time was 1.726688 s and exact mHC stream wait was
0.789410 s. Device service overlaps host issue/wait and is not additive wall
time. Graph mHC pre+post was 2.897822 s; host-exclusive plus exact wait accounts
for 2.516098 s, leaving a 0.381724 s boundary/validation residual.

`maximum_device_dsv4_mhc_kernel_seconds` is now finite at 1.054103 s. One
negative event sample on GPU 2 was clamped and reported explicitly rather than
wrapping to 18,446,744,070 s; the clamp counter is one.

## Attribution verdict

The measurement succeeds in locating query and synchronization time with no
unexplained residual above 0.382 s. It also falsifies the label on the old KV
bucket: only 0.601 s is page projection/norm/RoPE, while 3.759 s belongs to the
row-local sliding append and learned compressor. That 3.759 s path is named and
source-traceable but not internally split by this arm, and is recorded as the
one material unresolved sub-bucket rather than distributed across projection
terms.

The cost model at this operating point now names the largest serial waits as
MoE synchronization (4.177 s) and attention synchronization (3.942 s), while
the largest query resources are device kernel service (7.159 s) and host page
allocation/zeroing (2.338 s). Selecting a mechanism or adding a finer
compressor split is a new decision and is outside this measurement-only scope.

The final required `make check` passed 2/2 CTest targets before the result
record commit.
