# Experiment 0095 — page-major TP2 prefill

Status: **accepted.** Physical-device prompt processing executes layer-major
over a page of rows, each row owning its own fused device mHC state, and the
page's rows are grouped by expert before the routed FP4 weights are decoded.
The result is bit-equal to the accepted page-1 path and 1.204x faster at the
production operating point. `--prefill-page-tokens 64` is now a supported TP2
prompt setting.

This closes the prerequisite that rejected experiment 0094.

## The bottleneck, instantiated

`results/dsv4-rank-local-phase/rank-local-tp2.json.log`, an 18-token TP2
physical prefill taking 3.7517 s, already carried the breakdown:

| term | seconds | share |
| --- | ---: | ---: |
| CPU routed expert (`device_moe_runtime.routed_cpu_seconds`) | 1.9929 | 53.1% |
| attention | 1.4174 | 37.8% |
| MoE router and enqueue | 0.3212 | 8.6% |
| mHC pre, embedding, norms | 0.011 | 0.3% |

`argmax_r` is host CPU routed-expert execution. The mechanism reduces the
FP4/E8M0 weight-decode volume of that term and touches no other resource:
routes, precision, expert residency, NVMe reads and H2D are unchanged, and its
extra cost is one page of activation scratch inside the admitted host ceiling.

**`mhc_post_seconds` is not mHC work and must not be read as such.** The same
run reports 1.9941 s of it against 7.7e-05 s of measured mHC kernel time. The
transition blocks on the stream, and what the stream is waiting for is the
enqueued MoE host callback, so `mhc_post` and `routed_cpu` are the same
seconds counted from opposite sides. Read naively the profile names mHC as the
bottleneck and sends the work at the wrong term.

## Arithmetic screen before the build

Grouping only pays if a page's rows collide on experts. Uniform top-6-of-256
routing over 64 rows would give 384 draws over about 199 distinct experts, or
1.9 rows per expert, which would not have been worth building. Real prefill
routing is skewed. From the 4,096-token prefill trace in
`results/deepseek-v4-lightning-indexer-4k/runs/reference/run-01/routes.jsonl`,
43 layers, 1,056,768 selections:

| page | distinct groups | rows per touched expert |
| ---: | ---: | ---: |
| 4 | 562,327 | 1.879 |
| 16 | 258,331 | 4.091 |
| 64 | 108,041 | 9.781 |
| 128 | 66,196 | 15.964 |

At page 64, 77% of selections fall in groups larger than ten rows. The
available weight-read reduction is 9.78x; the primitive from 0094 decodes each
tile once per four rows, so it can capture at most about 2.3x of it, which is
what the confirmation screen measured and what the runtime went on to deliver.

## What the prerequisite actually was

0094 concluded that runtime integration was gated on "an exact multi-row device
mHC state representation", and proposed either multi-row kernels or saving and
restoring fused state per row. Neither was needed.

The complete fused state is one device workspace plus three host scalars —
stage, residual index, and branch-ready. Making that tuple a **slot** and
swapping which slot every later mHC, attention and MoE command binds to costs
no device copy and changes no kernel shape, so a row's command sequence stays
byte-identical to the token-major sequence it replaces. 64 slots are 6.1 MB.

`dsv4_mhc_select_slot` and `dsv4_mhc_reserve_slots` are the whole mechanism. A
CUDA test advances three rows through begin, two transitions and finish,
interleaved a step at a time, and requires bit equality against the same rows
run to completion one after another.

## Two structural blockers the handover did not know about

Both were found by the cheapest screen that could see them: one untraced
581-token run at page 64, about seven minutes.

**The KV block admits one operation at a time.** A block refuses a reservation
or a mutation while any device lease is outstanding, and a page sweeps every
row of one layer before any of them is collected. The deferred append failed
first (`cannot reserve an in-flight device block`); switching it off moved the
failure to the synchronous append (`cannot mutate an in-flight device block`),
because the fused attention command retains the layer's KV read leases until
the collect. A page therefore uses the unfused attention command, which
releases its leases before returning. That is what `attention_page` already
drives, so the page passes each row's mHC slot into it and the branch still
lands in that row's workspace.

**The traced arm does not exercise the production attention path.** Enabling
the layer-hash trace disables the fused branch in both arms, so the exactness
gate cannot see anything that only the fused path does. Both blockers above are
invisible to it. Any page-major candidate must be run untraced before it is
believed.

## What had to stay per row

Two things resisted batching and are marked in the code:

- **The router.** Its logits are not rounded to BF16, and the row-batched
  projection reassociates the accumulation. The expert selection was identical
  everywhere, but the coefficients moved by a ULP — 321 of 344 traced routes
  differed, while every `ffn_router_weights` hash matched, because that hash
  rounds to BF16 first. The route trace's decimal round-trip is what caught it.
  The attention projections keep their row-batched form only because their
  BF16-rounded outputs absorb the same reassociation.
- **Operation-record order.** The page must be routed before any of it is
  executed, so each row's `ffn_mhc_pre` and `ffn_norm` records are emitted
  alongside its route. Without that the hashes were all present and all equal
  but permuted within a layer, and the gate failed on ordering alone.

The 0094 primitive also gained a production-shape test. Its original case is 64
input columns, which cannot see a reduction that only differs once the column
loop is long enough to be blocked; the runtime drives it at 4,096.

## Exactness

`scripts/run_dsv4_page_major_prefill_correctness.sh` compares a page arm
against the accepted page-1 path on generated token ids, logits, layer hashes,
operation hashes, the route trace including coefficients, and decode checkpoint
reads.

| arm | prompt | maximum page | result |
| --- | ---: | ---: | --- |
| page 4 | 8 tokens | 4 | all equal, 0 checkpoint reads |
| page 64 | 52 tokens | 51 | all equal, 0 checkpoint reads |
| page 64 | 144 tokens | 64 | all equal, 0 checkpoint reads |

Evidence is under `results/dsv4-page-major-prefill-correctness{,-64}/`.

## Throughput

`scripts/run_dsv4_page_major_prefill_ab.sh`, 581-token prompt, devices 1,2,
three interleaved repetitions, no diagnostic trace.

| term | page 1 | page 64 | ratio |
| --- | ---: | ---: | ---: |
| prefill | 75.720 s | 62.886 s | **1.204x** |
| prefill throughput | 7.673 tok/s | 9.239 tok/s | 1.204x |
| CPU routed expert | 46.152 s | 21.170 s | 2.180x |
| routed gate/up | 28.877 s | 13.888 s | 2.079x |
| routed down | 14.544 s | 6.790 s | 2.142x |
| attention | 18.726 s | 31.941 s | 0.586x |
| mHC post | 54.394 s | 2.211 s | — |
| MoE outside the CPU term | 2.342 s | 5.358 s | — |

Per-run prefill was 75.720, 75.635, 76.066 s against 62.886, 62.927, 62.757 s.
The spread within each arm is under 0.6%, so the difference is far outside run
variance. `mhc_post` and `moe` move against each other because the MoE wait is
attributed to whichever call blocks on the stream; their sum plus attention
accounts for the whole phase in both arms.

Every other resource is unchanged or negligible:

| resource | page 1 | page 64 |
| --- | ---: | ---: |
| demand H2D | 572,554,752 B | 572,554,752 B |
| prefill checkpoint reads | 572,554,752 B | 572,554,752 B |
| decode checkpoint reads | 0 | 0 |
| cache evictions | 0 | 0 |
| demand wait | 0.326 s | 0.424 s |
| host callbacks | 24,983 | 24,983 |
| routed expert invocations | 149,898 | 149,898 |
| RSS | 158.362 GB | 158.411 GB |
| per-GPU VRAM | 22.380 / 22.378 GB | 22.397 / 22.388 GB |

Both arms generate the same token.

## Served path

`strata-server` already defaulted `prefill_page_tokens` to 64 and the physical
path ignored it, so accepting this turns page-major prompt processing on for
the server and for `strata-chat` without a flag. `--prefill-page-tokens` is now
parsed by the server as the escape hatch. Measured on the same box against a
580-token chat request, model `strata-deepseek-v4`, devices 1,2:

| server arm | prompt throughput | prompt time |
| --- | ---: | ---: |
| default (page 64) | 9.173 tok/s | 63.23 s |
| `--prefill-page-tokens 1` | 7.651 tok/s | 75.81 s |

1.199x, matching the runner A/B, with identical generated text.

## What is left, and how large it is

**Attention is now the largest term**, 31.941 s of 62.886 s, and 13.2 s of that
is the price of the unfused command. Recovering it would put prefill near
50 s, about 1.5x over page 1. It needs the KV block to accept a page of rows
under one reservation — one reserve, a commit per row, one account — so the
fused asynchronous attention command can be used with slots. That is a bounded
next experiment with a measured 13.2 s prize.

**The CPU term is close to done.** Measured 2.18x against the primitive's 2.315x
four-row ceiling, while the available dedup is 9.78x. The gap is not weight
reads: with four rows per decode the routed work is much closer to
compute-bound, so widening the tile has little left to win, and the AVX2
accumulator budget does not have room for eight rows at sixteen outputs anyway.

## Reproduce

```bash
cmake -S . -B build-pagemajor -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_NCCL=ON
cmake --build build-pagemajor --target strata-deepseek-run strata-tests -j4

PAGE_TOKENS=64 PROMPT_REPETITIONS=140 \
  scripts/run_dsv4_page_major_prefill_correctness.sh
REPETITIONS=3 scripts/run_dsv4_page_major_prefill_ab.sh
```

Validation on this branch: `make check` 100%, CUDA suite 305/306 with one
opt-in skip, `git diff --check` clean.
