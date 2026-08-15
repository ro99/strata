# Experiment 0094 — TP2 CPU expert row batching

Status: **rejected for runtime integration.** The isolated CPU mechanism is
exact and materially faster, but the layer-major physical prefill graph needed
to expose multiple prompt rows fails the full-model exactness gate before its
first MoE. No runtime flag is promoted.

## Hypothesis and gate

The rank-local TP2 prompt path executes the NUMA-local routed expert once per
`(token, layer)` callback. Grouping rows which selected the same expert should
decode each transformed FP4/E8M0 tile once for up to four rows while preserving
each row's column/FMA order.

The production operating-point baseline was a 502-token prompt:

| term | measured value |
| --- | ---: |
| prefill | 78.4126 s (6.402 tok/s) |
| routed expert invocations | 129,516 |
| CPU routed work | 40.130 s |
| host callbacks | 21,586 |
| demand H2D | 0.529 GB |

Under `tau = max_r W_r/B_r + sum_serial`, CPU routed work is the largest named
prefill term and the callback chain is serial work. The target was therefore
CPU weight-decode volume and callback serialization. The proposed primitive
does not change routed weight residency, H2D, NVMe reads, model precision,
routes, or output accumulation order. Its extra resource is one page of gate,
up, activation, and routed-output scratch, bounded by the admitted host-memory
ceiling of 216 GiB and the existing 95% VRAM cap on each RTX 3090.

The binding correctness gate was exact equality of routes, operation hashes,
layer hashes, logits, and generated token IDs, plus zero decode checkpoint
reads. The rollback condition was any mismatch, increased I/O, or no material
throughput improvement.

## Cheapest mechanism screen

`strata-dsv4-expert-row-batch-probe` reproduces one production-shape TP shard:
4096 hidden columns, 2048 expert intermediate columns split over two NUMA
shards, and a 24-worker node-local pool. Five interleaved repetitions compare
the scalar row loop with a four-row tile decode. Every result is bit-identical.

| selected rows | scalar median | batched median | speedup |
| ---: | ---: | ---: | ---: |
| 4 | 1.935649 ms | 0.836272 ms | 2.315x |
| 8 | 3.774375 ms | 2.420360 ms | 1.559x |
| 12 | 5.648815 ms | 3.262551 ms | 1.731x |
| 16 | 7.531003 ms | 3.991433 ms | 1.887x |

The confirmation screen passes its predeclared `>1.05x` materiality threshold
in every arm. The initial five-repetition screen was also exact and measured
3.330x, 2.219x, 1.711x, and 2.283x at 4, 8, 12, and 16 rows respectively. The
screen establishes
that shared weight decode reduces the intended CPU term; it does not establish
that TP2 can legally present several rows to the primitive.

Reproduce with:

```bash
cmake --build build-nccl --target strata-dsv4-expert-row-batch-probe -j2
scripts/run_dsv4_cpu_expert_row_batch_screen.sh
```

Ignored raw results are in `results/dsv4-cpu-expert-row-batch/summary.json`.

## Full-model falsification

Physical TP2 currently owns mHC state in a token-major fused CUDA state
machine. CPU expert row batching requires a layer-major page. The candidate
therefore used the existing host page mHC calculation and standalone physical
attention, then compared page 64 against the accepted page-1 path on an
8-token prompt and one decode token.

Both arms completed with zero decode checkpoint reads and selected the same
generated token. The exact gate failed:

| check | result |
| --- | --- |
| generated token IDs | equal |
| logits | different |
| layer hashes | different |
| operation hashes | different |
| routes | different |

The first operation mismatch is position 0, layer 0, `attn_mhc_post`:

| operation | page 1 | page 64 |
| --- | --- | --- |
| `attn_mhc_pre` | exact | exact |
| `attn_norm` | exact | exact |
| `attn_output` | exact | exact |
| `attn_mhc_post` | `9795903614f4597e` | `67b422071cf99346` |

This happens before the first routed expert, so it neither implicates nor
validates the row-batched expert primitive. The device path retains the exact
pre/post/combination state produced by its fused transition; matching only the
BF16 reduced vector and BF16 branch vector does not reproduce that hidden mix
state on the host.

The 8-token prefill times were 1.8715 s for page 1 and 2.1019 s for page 64.
They are correctness-probe observations, not throughput results: fixed model
setup was about 119 seconds per arm and dwarfed the measured window.

Ignored evidence is in
`results/dsv4-cpu-prefill-correctness/{summary.json,page1,page64}`. Its harness
was superseded by `scripts/run_dsv4_page_major_prefill_correctness.sh`, which
runs the same comparison at any page size.

## Decision

Stop at the negative gate. The failed physical page runtime code was removed,
and `--prefill-page-tokens` is not advertised as a TP2 speed setting in the
Quick Start.

The isolated exact primitive and its unit/probe coverage remain useful input
for a future design, but runtime integration is gated on an exact multi-row
device mHC state representation (or an equivalently exact way to save and
restore one fused state per prompt row). That prerequisite must be measured and
validated before retrying CPU expert row batching.
