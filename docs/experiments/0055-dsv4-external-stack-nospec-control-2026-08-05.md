# Experiment 0055 — the external stack's missing control arm, and what it changes

Status: **investigation; no runtime code changed.** Runs the external
DeepSeek-V4-Flash stack on *this* machine with and without its speculative
decoder, which experiment 0050 never had, and re-prices four standing decisions
against the result.

## Result

**DSpark depth 5 is a 0.72x regression on this machine.** The external stack
decodes at **10.2 tok/s** with speculation off and **7.3 tok/s** with it on. The
two arms differ by one line of the launch command.

The external stack's advantage over Strata is therefore **2.51x and entirely in
the base forward pass** — 97.6 ms/row against Strata's 245.5 ms/row — not in
speculation, not in CPU expert placement as such, and not in anything Strata
cannot reach without new hardware.

Three standing conclusions change:

- **DSpark is not the surviving lever.** Both engines measure a row-marginal
  cost near 0.6 where the depth-5 break-even needs 0.36. Strata's own rejection
  in experiment 0029 was correct and now has an independent confirmation on the
  same hardware, from an engine that ships the mechanism and loses 28% to it.
- **Experiment 0054's "do not re-open under 32 GB/s" is contradicted by
  measurement.** The external stack moves all 3.449 GB/token of routed-expert
  weights out of host DRAM *and* completes attention, dense and router in
  97.6 ms — 35.3 GB/s on the expert term alone, before crediting overlap.
- **The largest gap between the two stacks is prefill, not decode**: ~180
  ms/token against ~5 ms/token, and it is a re-execution defect, not a transfer
  one.

## Contract

- Hypothesis: the external stack's measured 7.3 tok/s on this machine is
  produced by speculative decoding, as experiment 0050 assumed when it priced
  DSpark at 2.6x from the author's reported numbers.
- Primary metric: median single-stream decode tok/s over three runs, same
  client, same prompt, same 512 output tokens, speculation the only variable.
- Correctness gate: none applicable — no Strata code is changed and the
  external stack is not under a correctness claim here.
- Memory ceiling: unchanged; the external stack was already resident.
- Kill criterion: none. This is a control arm for an existing measurement.
- Rollback: the external server is restored to the user's original
  speculation-enabled configuration in tmux session `ds4serve`.

## Method

The user's `bench/launch.sh` was copied and one line replaced:

```
-    --speculative-config '{"method":"dspark","num_speculative_tokens":5,...}'
+    --seed 0
```

Everything else is byte-identical: same checkpoint at
`models/dsv4f`, TP=2 on the two 3090s, `--kv-cache-dtype fp8_ds_mla`,
`--compilation_config.cudagraph_mode FULL_DECODE_ONLY`, `--max-num-seqs 4`,
`LVLLM_MOE_NUMA_ENABLED=1`, `LK_THREADS=28`, `LK_THREAD_BINDING=CPU_CORE`. The
benchmark is the stack's own `bench/bench.py --decode-only --tg-runs 3`.

Forward-pass rates come from vLLM's own counters. With speculation on, each
running request drafts exactly `num_speculative_tokens` per pass, so
`passes/s = drafted_throughput / (5 × running_reqs)` and each pass verifies
6 rows. With it off, `passes/s = generation_throughput / running_reqs` at 1 row
per pass.

## Measured — the two arms

| arm | decode, single-stream | runs | ms/forward pass | rows/pass |
|---|---:|---|---:|---:|
| DSpark depth 5 | 7.3 tok/s | 6.5, 7.3, 14.6 | 385-400 | 6 |
| **speculation off** | **10.2 tok/s** | 10.2, 10.2, 10.2 | **97.6** | 1 |

The speculative arm's 2.2x run-to-run spread is entirely acceptance variance:
its 14.6 tok/s run logged `Mean acceptance length: 6.00, Per-position acceptance
rate: 1.000, 1.000, 1.000, 1.000, 1.000` — a degenerate repeating output, not a
faster machine. The non-speculative arm reports 10.2 three times.

Prefill and time-to-first-token move the same way:

| prompt tokens | prefill t/s, DSpark | prefill t/s, no spec | TTFT DSpark | TTFT no spec |
|---:|---:|---:|---:|---:|
| 669 | 20.3 | 66.6 | 33.0 s | 10.0 s |
| 2,630 | 161.7 | 197.9 | 16.3 s | 13.3 s |
| 5,182 | 269.6 | 320.6 | 19.2 s | 16.2 s |
| 10,326 | 175.9 | 318.7 | 58.7 s | 32.4 s |

**Decode is flat in context length**: 10.3 tok/s at 669, 2,630, 5,182 and
10,326-token prompts. Nothing context-scaling is on the critical path.

## Measured — the row-marginal cost, which decides everything multi-row

Write per-pass cost as `c(k) = 1 + m(k-1)` in units of a 1-row pass. Speculation
at depth 5 commits `a` tokens per pass and wins only if `a > c(6) = 1 + 5m`,
i.e. **`m < 0.36`** at the measured `a ≈ 2.9` — before paying for the draft.

| measurement | rows | ms/pass | derived `m` |
|---|---:|---:|---:|
| no spec, 1 request | 1 | 97.6 | — |
| no spec, 4 requests | 4 | 318 | 0.753 |
| DSpark 5, 1 request | 6 | 385 | **0.586** |
| DSpark 5, 4 requests | 24 | 1250 | — |

The within-sequence figure, 0.586, is the one that governs speculation, and it
is lower than the cross-request 0.753 because consecutive tokens of one sequence
share experts while independent requests do not.

**The arithmetic reproduces the measurement.** Predicted speculative throughput
is `a / c(6) = 2.9 / 3.93 = 0.738×` base, i.e. 7.5 tok/s. Measured 7.3 tok/s,
`7.3/10.2 = 0.716×`. Agreement within 3%.

Strata's own measured marginal, experiment 0029 after attention batching, is
`m = 0.659`. **The two engines are in the same regime.** Speculation fails here
for a reason neither implementation controls: routed-expert reuse across a
speculative window on this model is small. Experiment 0025 measured it on
Strata's own decode trace — window dedup 1.60 at γ=4 and 1.88 at γ=8, cold dedup
1.22 and 1.44. A depth-5 window needs something near 6 for the weight read to
amortize; it gets about 1.7.

This retires the standing assumption that DSpark raises Strata's ceiling to
75-80 tok/s. That figure assumed one weight read amortized across ~3.5 accepted
tokens. The measured dedup does not deliver it, and the only engine on this
machine that ships the mechanism pays 28% for it.

## Measured — where the external stack puts things

From its own startup log and `/proc`:

| quantity | value | source |
|---|---:|---|
| VRAM weights per GPU | 6.2 GiB (6.56 with the draft) | `Model loading took` |
| routed experts in VRAM | **none** | `Initialized lk_moe with 256 experts for layer …ffn.experts [CPU]`, every layer |
| host RSS per TP rank | 74.3 GB | `ps`, ×2 ranks = 148.6 GB |
| KV cache | 7.89 GiB, 64,693 tokens, `fp8_ds_mla` | `Available KV cache memory` |
| GPU P2P | unavailable, custom allreduce disabled | `custom_all_reduce.py:160` |

So both stacks hold the same ~147 GB expert set resident in host DRAM (Strata's
RSS is 148.9 GB). The difference is that Strata *also* mirrors 46.6 GB of it
into VRAM and demand-transfers the misses, while the external stack computes in
place and moves only activations.

## Reconciliation: the 2.51x is one term, in five pieces

Strata's 245.5 ms/step from experiment 0051, against a 97.6 ms external step:

| Strata phase | ms/step | external equivalent |
|---|---:|---|
| `moe_prepare` — PCIe demand-H2D wait | 101.15 | **0** — weights never cross a link |
| MoE execution | 49.10 | the whole step: ~3.449 GB from DRAM |
| attention (2.4 ms of it device time) | 70.17 | on GPU, inside the graph, concurrent |
| mHC pre + post | 21.10 | on GPU, concurrent |
| router / output head / branch norm | 4.89 | concurrent |
| **total** | **245.5** | **97.6** |

The external stack's step is *one* term with everything else hidden underneath
it. Strata's is five terms in series. That, and not any algorithm, is the 2.51x.

## Ranked candidates, ordered by Strata's measured `argmax_r`

### 1. Row-batched, expert-major MoE execution

`enqueue_deepseek_moe` takes `std::span<const float> hidden` — one row — and
`kMaxDeepSeekRoutedExperts = 6U` (`kernels/cuda/backend.cu:630`). `moe_page`
batches only the router and then loops:

```cpp
for (std::uint32_t row = 0U; row < rows; ++row) {
    result = execute_moe(layer, routes[row],
                         input.subspan(row * kHidden, kHidden), …);
}
```

`src/deepseek_runtime.cpp:3177`. Every row re-acquires and re-reads its own
experts. lk_moe's boundary is one call for all rows with the routing table
passed in — `cpu_decode(stream, num_tokens, top_k, hidden, topk_ids,
topk_weights, out)` — so rows are grouped by expert and each distinct expert is
read once.

**This is where prefill lives.** At a 511-token prefill, `moe_page` performs
`511 × 6 = 3,066` expert applications per layer against a distinct-expert
ceiling of 256 — up to **12x** redundant expert execution. Experiment 0030
measured prefill at 92.2 s for 511 tokens (**180 ms/token**) with prefill weight
H2D already at its 105 GB floor, so this is *not* a transfer problem. The
external stack does the same work at ~5 ms/token.

Prefill is the largest measured gap between the two stacks — roughly 35x —
against decode's 2.51x.

**Verify first:** Strata's prefill has never had a per-phase breakdown emitted.
Run one and state how much of the 180 ms/token is MoE execution, attention and
dispatch. If MoE execution is not `argmax_r` for prefill, this ranking is wrong.
Also re-measure — 0030 is from 2026-07-25 and MoE-adjacent work has landed since.

### 2. Remove the 101 ms serial PCIe demand wait

41% of Strata's step, the single largest term, and the external stack pays zero
for it. Experiment 0051 measured 693 MB/step at 6.9 GB/s.

Two mechanisms, and this experiment changes the verdict on one:

**(a) Host compute for misses.** Experiment 0051 rejected the scheduling policy
at 0.76x, and 0054 rejected static placement by arithmetic: break-even 32.3 GB/s
standalone against 26.23 measured, best case 0.939x, "do not re-open under
32 GB/s."

**That bar is now known to be clearable on this exact box and ISA.** The
external stack completes a 1-row step — all 3.449 GB of routed-expert weights
read from host DRAM, *plus* attention, dense and router — in 97.6 ms. That is
35.3 GB/s on the expert term alone before crediting overlap, against a 32.3 GB/s
break-even.

The gap is not the kernel's inner loop. Strata's kernel is 26.23 GB/s standalone
(28.62 under `numactl --interleave=all`); 35.3 is 1.35x that. The gap is the
decomposition around it, and 0051 measured exactly that: 86 barriers/step at
73.6 µs, 0.9 diverted experts per layer split across 28 threads, **11.0 GB/s in
situ against 26.23 standalone.** lk_moe issues one call per layer over all
experts against its own persistent NUMA-bound pool, not a `parallel_for` per
expert-phase.

The adjacent unexecuted item matters more here than anywhere else:
`Dsv4ResidentWeightStore::stage` still allocates `MAP_PRIVATE|MAP_ANONYMOUS`
with no NUMA policy (experiment 0026's "Next term", still item 5 of 0050).
Strata's own probe measured **23 GB/s at 28 threads** when the arena was
first-touched on one node against **76.3 GB/s node-local** — a 3.3x swing from
placement alone, larger than the entire margin being argued about.

**(b) Overlap rather than eliminate.** Per the charter, the 101 ms is
`Σ_serial` — a demand-miss stall — not volume. Fix overlap first; it is usually
cheaper and strictly larger.

### 3. Attention and mHC to the device, activations device-resident

91 ms/step of which 2.4 ms is device time. This is 0050's item 1 and 0051's
"Next, in order" item 1, still not done. Named defect: `CudaBackend::flash_attention`
host-copies each KV row into pinned staging and H2Ds it 43 times per step when
only one row per layer is new.

The context-flatness result sharpens this. The external stack decodes at
10.3 tok/s identically from 669 to 10,326 tokens; Strata's score-plus-KV-gather
went 10.8 ms/row at 12 tokens to 123.9 ms/row at ~570 (0029). This term does not
merely cost more than the external stack's — it is the one that grows while
theirs does not.

### 4. CUDA graph capture

Depends on 2 and 3. Their changelog prices it at 2x; that is unverified here and
bundles other changes, so it is unpriced.

## What is not worth taking

- **DSpark / speculative decoding.** 0.72x measured in the external stack on
  this machine; 0.669x and 0.785x in Strata's own 0029. Not an implementation
  failure in either engine — window dedup on this model is 1.6-1.9, and depth 5
  needs ~6.
- **Wholesale CPU expert placement as an architecture.** The external stack's
  10.2 tok/s *is* that design fully realized here. It is 2.51x Strata's current
  number, and Strata's 87.5% VRAM cache hit rate is an advantage that design
  gives up. Build the hybrid, not the copy.
- **lk_moe's kernel itself.** ~35 GB/s implied against Strata's 26.23 standalone
  — 1.35x — and 0051 already established that kernel rate is invisible until the
  decomposition around it is fixed.
- **The aggregate-throughput framing.** Measured here: 10.2 tok/s at one request
  to 12.4-12.8 aggregate at four, i.e. **1.2x**, not the 68 tok/s their author
  reports on a 563 GB/s box. Cross-request batching does not amortize on this
  memory subsystem either — the same `m` that sinks speculation sinks it.

## Ceiling, if items 2 and 3 land

| term | ms/token | source |
|---|---:|---|
| VRAM reads, 11.93 GB/token, summed across sequential devices | 16.22 | 0051, measured |
| host expert misses, 693 MB at 26.23 GB/s, overlappable | 26.4 | 0051, measured standalone |
| attention + mHC + router on device | 3-5 | 0050 estimate, **never measured** |

Overlapped: **~20-30 ms/token, 33-50 tok/s** — on the current placement, without
speculation, without new DIMMs, without moving to CPU-only experts. Above the
external stack's 10.2 tok/s, and the reason is the asymmetry Strata genuinely
has: 12.5% of expert bytes cross a link instead of 100% coming from DRAM.

## Risks and open items

1. Strata's prefill has no per-phase attribution. Emit one before building
   item 1.
2. The 26.23 GB/s host-kernel figure is standalone with a whole expert per
   thread. Its in-situ rate at four or more diverted experts per layer has never
   been measured; 0051 measured only the 0.9-per-layer case, at 11.0 GB/s.
3. The dedup constants come from a 127-position decode trace at one context.
   Re-measure at the target context before reusing them.
4. The external stack runs bf16 activations, `fp8_ds_mla` KV and Marlin FP8
   linears. Its 97.6 ms bounds what this hardware can do; it is not a like-for-
   like numerical contract with Strata's exact path.
5. The two 3090s have no working P2P and the 5060 Ti sat idle in every external
   arm. Strata uses all three GPUs. The comparison is not equal-hardware in
   Strata's favour on VRAM and against it on GPU count.

## Artifacts

Throwaway and in the session scratchpad, not the repository: the modified
launcher (`launch_nospec.sh`, one line different from the user's
`bench/launch.sh`), the two server logs, and the concurrency client. The
external server was restored to the user's speculation-enabled configuration in
tmux session `ds4serve`. Every Strata number above is from experiments 0025,
0029, 0030, 0050 and 0051 already in the repository.
