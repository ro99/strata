# Experiment 0027 — the decode attention dispatch crossover

## The framing that was wrong

Experiments 0023 and 0024 both closed by recording:

> Attention 197 ms/step with `attention_cuda_dispatches: 0` — all 5,461 decode
> dispatches take the host scalar path, because `flash_attention_minimum_rows`
> is 256 and decode presents one row.

Two errors in one sentence, and neither survives reading the dispatch site.

**The crossover is keyed on key rows, not query rows.**
`should_dispatch_flash_attention_cuda(enabled, logical_rows, minimum)` is called
from `src/deepseek_runtime.cpp:2166` with `logical_rows = score_stride`, and
`score_stride = window_count + attended_compressed_count` — the number of K/V
rows the query attends, not the number of query rows. Batch-1 decode presents
one *query* row and hundreds of *key* rows. The threshold was never the reason.

**Decode measured zero CUDA dispatches because `--flash-attention` was never
passed.** `enable_flash_attention` defaults false, and neither
`run_deepseek_v4_pinned_arena_ab.sh` nor `run_deepseek_v4_prefill_layer_major_ab.sh`
set it. The counter was reporting a disabled feature, not a lost crossover.

## What decode's row count actually is

At a 511-token prompt with 127 generated tokens, positions run 511..637.
`kWindow` is 128, so `window_count` is 128 throughout. The sparse indexer is
only admitted above `kIndexTopK * ratio = 2048` tokens
(`src/deepseek_runtime.cpp:1630`), so at this operating point
`attended_compressed_count = (position + 1) / ratio`, and the per-layer
compression ratios give three populations:

| layers | ratio | compressed rows | `score_stride` | dispatch at 256 |
|---|---:|---:|---:|---|
| 21 (even indices 2..42) | 4 | 128..160 | **256..288** | CUDA |
| 20 (odd indices 3..41) | 128 | 4..5 | 132..133 | scalar |
| 2 (indices 0, 1) | 0 | 0 | 128 | scalar |

So decode sits *exactly on* the threshold: the ratio-4 layers cross 256 at the
first decode step and stay above it. The prediction that enabling the flag
would produce 21 x 127 = 2,667 CUDA dispatches was confirmed to the dispatch:
the hybrid arm reports `attention_cuda_dispatches: 2667` and
`attention_scalar_dispatches: 2794`.

## Contract

- Hypothesis: decode attention is on the host scalar path for every layer, and
  moving the layers above the crossover to CUDA reduces
  `attention_score_seconds` and median decode steps/s.
- Bottleneck measured first (experiment 0026): a 317.7 ms decode step is
  86.3 ms MoE demand wait, **52.0 ms attention score**, 46.9 ms MoE compute,
  37.5 ms mHC pre, 27.0 ms attention query projection, 24.6 ms attention output
  projection. Attention score is the second-largest term and the largest host
  scalar term.
- Target term: `attention_score_seconds`. The 21 ratio-4 layers carry
  `21 x 278 / (21 x 278 + 20 x 132 + 2 x 128)` = 67% of the scalar score work.
- Sign on every other resource: the CUDA path adds per-call H2D of the packed
  segments and one D2H of the output, and adds device kernel time on a device
  whose copy engine is already the step's `argmax_r`. It changes no routing,
  precision, top-k, expert count, or numerics — `f64_dot_f32_score_f32_accum`
  is the declared contract on both paths.
- Correctness: output must be bit-identical on every arm.
- Rollback: median decode not above the scalar reference beyond run variance,
  or any output byte changing.

## Commands

```bash
scripts/run_deepseek_v4_attention_dispatch_ab.sh
```

One build, three arms, only runtime flags differ, interleaved, median of three.

- `scalar` — current default, every layer on the host scalar path.
- `hybrid` — `--flash-attention`, CUDA at >= 256 rows, so 21 of 43 layers.
- `forced` — `--flash-attention --flash-attention-minimum-rows 0`, all layers.

The third arm exists so the crossover is measured rather than assumed. Prompt
34 sentences (~512 tokens), `--max-new 128`, `--pin-resident-arena`. About
3.5 minutes per arm, ~32 minutes total.

## Result

Median of three interleaved repetitions, each metric's own median:

| | scalar | hybrid | forced |
|---|---:|---:|---:|
| Decode steps/s | 3.150 | **3.226** | 3.181 |
| against scalar | — | **1.024x** | 1.010x |
| Decode seconds | 40.313 | 39.370 | 39.929 |
| Attention seconds | 14.052 | 12.846 | 12.520 |
| Attention score seconds | 6.577 | 5.293 | 4.638 |
| score against scalar | — | 1.243x | 1.418x |
| CUDA / scalar dispatches | 0 / 5,461 | 2,667 / 2,794 | 5,461 / 0 |
| FlashAttention H2D bytes | 0 | 1,832,522,496 | 2,952,690,944 |
| Maximum device flash seconds | 0 | 1.004 | 1.256 |
| mHC seconds | 5.544 | 5.487 | **6.473** |
| MoE seconds | 18.723 | 18.768 | 18.999 |
| Critical-path synchronization | 6.319 | 7.294 | 7.544 |

Every run, not just the median:

| rep | scalar | hybrid | forced |
|---|---:|---:|---:|
| 1 | 3.150 | 3.219 | 3.221 |
| 2 | 3.135 | 3.226 | 3.181 |
| 3 | 3.160 | 3.348 | 3.168 |
| range | 3.135–3.160 | 3.219–3.348 | 3.168–3.221 |

The scalar arm spans +/-0.4%. The slowest hybrid run beats the fastest scalar
run by 1.9%, so the ranges do not overlap and **1.024x is outside run
variance**. The slowest forced run beats the fastest scalar run by 0.25%, which
is not a separation; **`forced` is not a classified win**.

Gates, all passing: `dispatch_actually_differed`, `generated_tokens_equal`,
`logits_equal`, `layer_hashes_equal`, `operation_hashes_equal`,
`zero_decode_checkpoint_reads`, plus `cmp` on `routes.jsonl` byte-for-byte
against both candidate arms. `make check` passes. Output is bit-identical on
all three arms, which is the requirement: this is a dispatch policy, not a
numerical change.

## The crossover is right end-to-end and wrong on its own metric

This is the part worth keeping. Score time falls monotonically as more layers
move to CUDA — 6.577 to 5.293 to 4.638 — so **CUDA also beats the 28-worker
host scalar path at 132 rows**, well below the 256-row threshold. Judged on
`attention_score_seconds` alone, the threshold should be lowered.

Judged on the step it does not, and the reason is the charter's third rule.
The 2,794 additional low-row calls in `forced` cost 0.252 s of device service
and save 0.655 s of score time — net positive on attention, which is why
`forced` has the lowest attention total. But they also add **1.12 GB of H2D to
the link that experiment 0026 identified as the step's `argmax_r`**, and the
bill arrives outside the attention counters:

| | hybrid | forced | delta |
|---|---:|---:|---:|
| attention seconds | 12.846 | 12.520 | **-0.326** |
| mHC seconds | 5.487 | 6.473 | **+0.986** |
| MoE seconds | 18.768 | 18.999 | +0.231 |
| decode seconds | 39.370 | 39.929 | +0.559 |

mHC and MoE both run on the same three devices whose copy engines are already
saturated by routed-expert demand loads. Forcing the low-row layers onto CUDA
reduces host scalar time, which is not the bottleneck, by inflating H2D, which
is. Under a `max` that is negative by construction, and the measurement agrees.

So `flash_attention_minimum_rows = 256` survives, but not for the reason it was
set. It is not the point where CUDA starts beating scalar at attention; it is
approximately the point where the transfer CUDA costs stops being worth paying
out of the bottleneck resource. That is a different quantity, it is
hardware-dependent, and it will move if the H2D term is ever taken off the
critical path.

## Cost accounting

1.024x is a real but small win, and it should be reported next to what it is
made of. Attention was 112.9 ms/step of a 317.7 ms step; this recovers 10.1 ms
of it. The scalar path is not eliminated — 2,794 of 5,461 decode dispatches
still take it, and the 21 layers that moved gave up only 1.28x on the work they
carried, because a 687 KB segment upload and a 0.377 ms round trip per call
consume most of the arithmetic saving.

## Operating point

This is a ~512-token prompt, where the sparse indexer is not admitted and
`score_stride` is 256..288 on the ratio-4 layers. **Neither the 1.024x nor the
256-row threshold may be transplanted to a longer context.** Above 2,048 tokens
the indexer is admitted and `attended_compressed_count` is capped at
`kIndexTopK = 512`, so those layers present 640 rows — more than twice the
arithmetic per call against the same per-call transfer overhead, which should
favour CUDA more. The ratio-128 layers grow too, and would cross 256 at about
16,400 tokens. Both are extrapolations from the model, stated here as
assumptions to be re-measured, not results.

## Decision

Promote `hybrid`, which is what `--flash-attention` already selects, and leave
`flash_attention_minimum_rows` at 256. The flag stays opt-in rather than
becoming the default until it has been measured at the long-context operating
point, for the same reason `--pin-resident-arena` did: the benefit is
context-dependent and this is the regime least favourable to it.

Reject lowering the threshold. Recording the regime that would have been
required: forcing every layer onto CUDA would pay if H2D were not `argmax_r` —
if the expert working set fit in VRAM, or if demand loads were moved off the
critical path. It is not the operating point that was asked about.

## What this closes and what it does not

The claim in experiments 0023 and 0024 that decode attention was "on the host
scalar path because decode presents one row" is withdrawn. Decode presents
128..288 key rows, the crossover reads key rows, and the counter was reporting
a flag that was never set. The correct statement is that decode attention was
scalar because `--flash-attention` was off, and enabling it is worth 1.024x
here.

Attention is no longer the second-largest opportunity. After this change the
step is ~310 ms and its largest term is unchanged: 86 ms of serial MoE demand
wait, whose remaining overlap headroom experiment 0026 measured at 1.33x, or
7% of a step. Neither of the two terms this pair of experiments was opened to
attack is worth more than that at this operating point.
