# Experiment 0102 — DSV4 production query encoding falsifier

Status: **stopped at the prerequisite gate before implementation or model
timing.** The alleged 8x slowdown compared different weight encodings and
different kernels: experiment 0101 measured plain BF16, while the production
DeepSeek V4 query/KV tensors are FP8 E4M3 block-128 and dispatch the native FP8
kernel. The authorized plain-BF16 cuBLAS swap cannot reach these tensors.

## Predeclared contract

The proposed experiment first had to explain why 0101's isolated per-layer
plain-BF16 query kernel time implied about 0.82 s over 43 layers while 0100
measured 7.159487 s of production query matmul device service. The proposed
hypothesis was concurrent expert H2D contention. The cheapest planned test was
an upper-bound concurrent-H2D arm in the standalone probe, derived from 0100's
74,083,499,520 demand bytes rather than from an assumed transfer rate.

Only if that prerequisite held was the runtime scope authorized: multi-row
plain-BF16 page projections through the existing cuBLAS handle, with single-row
decode and every other encoding unchanged. Correctness was to compare current
and cuBLAS results with deterministic FP64 scalar-oracle samples spanning the
full production M/N/K shapes; cuBLAS had to be no worse in declared maximum and
RMS error. Generated IDs had to remain `2107, 8777, 1277, 440`, decode
checkpoint reads zero, decode unregressed and `make check` green. Runtime
workspace could not grow beyond the existing 384 MiB/device page ceiling.

The corrected primary value metric was same-build
`attention_query_matmul_kernel_seconds`, not one-arm total prefill. The 0100
reference was 7.159487 s. Total prefill, tok/s and query/KV/score buckets were
directional only because identical code had shown roughly 16 s of MoE-upload
NUMA spread. No total-prefill movement within that range could promote or
reject the candidate.

Budget was at most ten minutes for source/counter audit and contention screen,
twenty minutes for oracle plus implementation/build/tests, then two untraced
677-token page-8192 model arms at about four minutes each. No 2,612-token arm,
repetition or fixed/marginal fit was authorized.

## Cheapest prerequisite audit

Before modifying the H2D probe, the checkpoint headers and manifest dispatch
were checked. All three production tensors contradict the premise:

| Tensor | Shape | Checkpoint dtype | Runtime encoding | Production kernel |
|---|---|---|---|---|
| `layers.0.attn.wq_a.weight` | `[1024,4096]` | `F8_E4M3` | FP8 E4M3 block-128 | `native_fp8_matmul_kernel` |
| `layers.0.attn.wq_b.weight` | `[32768,1024]` | `F8_E4M3` | FP8 E4M3 block-128 | `native_fp8_matmul_kernel` |
| `layers.0.attn.wkv.weight` | `[512,4096]` | `F8_E4M3` | FP8 E4M3 block-128 | `native_fp8_matmul_kernel` |

Each tensor has a paired `.scale`; `deepseek_manifest.cpp` consequently assigns
every non-routed quantized pair `Fp8E4m3Block128`. In `matmul_impl`, that
encoding first quantizes each activation row to E4M3 and then launches
`native_fp8_matmul_kernel` on a grid with one block per `(output row, batch
row)`. The plain-BF16 multi-row branch containing
`bf16_matvec_rows_kernel<16>` is reachable only when encoding is `Plain` and
dtype is BF16. None of these checkpoint projections satisfies that predicate.

The 0101 probe got M/N/K right but synthesized BF16 weights and invoked an exact
copy of the plain-BF16 kernel. Its 13.19--20.06x cuBLAS ratio and reassociation
comparison are valid only for that isolated plain-BF16 mechanism. They do not
measure the production FP8 decode/scaling arithmetic and cannot be divided into
0100's 7.159487 s production counter. The apparent 8x runtime slowdown is
therefore not a resource-contention observation at all; it is a cross-kernel,
cross-encoding comparison defect.

Source ordering also makes expert-upload contention an especially poor next
explanation: the page graph executes attention before the layer's FFN/MoE, and
generic `matmul_impl` rejects execution while a DeepSeek MoE command is in
flight. A saturated concurrent-H2D probe would answer a hypothetical overlap,
not the production comparison that triggered this experiment. It was not run.

## Gate verdict

The prerequisite fails before the contention test. The authorized production
mechanism targets `Plain + BF16`, while the measured 7.159487 s target term is
`Fp8E4m3Block128`. Routing that term through cuBLAS would require a materially
different mechanism, such as converting FP8 weights and block scales to a
cuBLAS-supported operand format. That would change weight bytes, conversion
work, cache capacity/eviction pressure and the numerical oracle, and is outside
0102's authorization.

No probe arm, FP64 oracle, CUDA/runtime implementation, model arm, generated
token check or timing comparison was performed after the premise failed. The
only code-tree change is the experiment documentation correcting 0101's
production relevance. The preserved 0098/0099 attention work remains supported
only by its deterministic structural/device counters and bit-exactness, not by
a one-arm total-prefill claim.

## Decision point

Experiment 0102 does not reject tensor-core projection work in general; it
rejects this plain-BF16 integration as a mechanism for the production query
term. A future experiment would first need to screen the actual FP8 block-scaled
operation, including the signs of expanded weight residency and cache pressure,
before any runtime design. No replacement mechanism is proposed or started
here.
