# Experiment 0186 — Gemma 4 device page closes the prefill defect

**Date:** 2026-08-24

**Branch:** `fix/gemma4-device-page-runtime`

**Origin:** experiment 0165's separate prefill defect

**Verdict:** **ACCEPTED — prefill is materially cheaper than decode**

## Hypothesis and predeclared gates

Gemma's implausible prefill cost is an orchestration defect: the host graph
serializes every projection, norm, RoPE, attention, KV commit, and residual
boundary even though all weights and the complete KV allocation fit on one RTX
3090. A bounded first-page graph using experiment 0184's exact FP32-output
Marlin primitive should reduce the measured CUDA/host handoff
`sum_serial` term and change each projected-weight read from eight skinny
passes to one page-shaped pass.

- Primary metric: complete 128-token prefill wall and tok/s.
- Architecture gate stated before the run: prefill/decode per-token cost below
  0.25. The directly measured vLLM reference in experiment 0181 was 881.67
  prefill tok/s at the comparable TP=1 short page and 36.214 decode tok/s.
- Correctness: maximum relative projection residual below 1e-4, identical
  full-generation token IDs against the serialized graph, unchanged W8A16
  dispatch, full Gemma oracle, then `make check`.
- Memory ceiling: one 24 GiB RTX 3090 at `--vram-fraction 0.95`, no widened or
  duplicate persistent weight representation. At a 16,384-token admission the
  planner totals 20.93 GiB and zero cross-device hops.
- Rollback/control: `STRATA_GEMMA4_DEVICE_PAGE=0` retains the serialized page
  graph. Any numerical divergence, OOM, ratio at or above 0.25, or a result
  within run variance rejects the integration.

The mechanism targets the baseline `sum_serial` argmax and the associated
activation handoff volume. It reduces projected HBM work from eight passes to
one. Its positive secondary signs are fewer launches, about three orders of
magnitude less H2D/D2H volume, and a persistent device KV ring. Its negative
signs are a bounded 128-row device workspace, FP64 reduction instructions for
the existing exact RMS contract, and a one-time in-place code/scale
permutation at load. Precision, scale semantics, attention visibility, and KV
storage remain unchanged.

## Mandatory cost model before design

The real operating point is the exact 19,531,513,296-byte MXFP4 shard, PCI
ordered CUDA device 1 / bus `82:00.0`, one RTX 3090 locked to 1,605 MHz at
250 W, context 512, VRAM fraction 0.95, and no cross-device hops. Experiment
0180's exact 128-token production profile instantiated:

| Baseline phase | Wall ms |
|---|---:|
| embedding | 0.479 |
| input norms | 92.954 |
| q/k/v projections | 436.385 |
| q/k norm and RoPE | 1,610.519 |
| KV prepare | 42.868 |
| attention | 1,305.041 |
| attention output projections | 180.311 |
| KV commit | 301.771 |
| post-attention norms and residual | 116.814 |
| pre-feedforward norms | 93.030 |
| gate/up projections | 888.632 |
| GeGLU | 597.662 |
| down projections | 342.803 |
| post-feedforward norms and residual | 132.829 |
| output head and KV upload | 42.510 |
| explained subtotal / residual | 6,184.924 / 169.877 |
| **complete page** | **6,354.801** |

Resource form of `tau = max_r(W_r/B_r) + sum_serial`:

| Resource/term | Work and measured service | Time |
|---|---|---:|
| MXFP4 projected weights | 18.377 GB useful, read eight times as about 147.017 GB | 293.972 ms CUDA kernels |
| activation H2D | 1.752 GB at 5.80 GB/s; Gen3 x8 rated payload about 7.88 GB/s | 302.228 ms |
| result D2H | 2.166 GB at 4.54 GB/s | 476.937 ms |
| CUDA/host handoffs | repeated blocking generic operations | **2,419.841 ms** |
| largest host phase | q/k norm and RoPE | **1,610.519 ms** |

`argmax_r` is the 2,419.841 ms serialized handoff term. PCIe achieved 58--74%
of the actual link ceiling, so the dominant term is overlap/ordering, not a
bandwidth limit. This is why the accepted mechanism owns the complete device
layer chain rather than optimizing one more host loop.

The accepted candidate's median page is 191.332 ms. Its resource accounting is:

| Resource/term | Work | Median representative time |
|---|---:|---:|
| CUDA kernel/HBM service | one page chain plus output head | **186.964 ms** |
| H2D | 2,774,016 bytes | 0.495 ms |
| D2H | 3,801,088 bytes | 0.714 ms |
| remaining host/serial wall | bookkeeping outside the timed stream regions | about 3.16 ms |

The candidate `argmax_r` is now GPU kernel/HBM service. H2D and D2H are still
within the expected Gen3 x8 range; no link-bandwidth claim is made.

## Cheap screens and budget

Experiments 0180--0185 performed the cheaper mechanism screens before runtime
integration: decode-style page ownership and conventional shared-BF16 WMMA
were rejected, upstream Marlin speed was reproduced but its BF16 epilogue was
rejected, and only the FP32-output form passed both exact shapes. The same pure
code/scale permutation then passed M=1, avoiding a duplicate decode layout.

For the final A/B, each fresh process paid about 19 seconds of load for a
0.19-second candidate window, roughly 100:1 fixed setup to measurement. A
shorter prompt was rejected because it would not test the M=128 defect. One
single-arm real-model gate first confirmed the mechanism; the required six-arm
counterbalanced matrix was budgeted at 2--2.5 minutes and completed within it.
The decode matrix used a 126-step steady window, about 4.24 seconds measured
after the same load, a roughly 4.5:1 ratio and about 75 seconds total.

## Implementation and the KV defect caught before promotion

The model loader replaces canonical MXFP4 code and scale order in place with
the GPTQ-Marlin K16/N64 permutation. One compact representation serves both
M=128 and M=1. The device executor holds hidden state, normalized branches,
Q/K/V, attention context, MLP intermediates, and all 60 layer transitions on
one CUDA stream. Exact FP64 RMS accumulation is retained; BF16 boundaries are
fused into their first consumers rather than silently removed.

The first correct device page measured 256.528 ms, but 65--70 ms remained
outside its 186.995 ms CUDA service. The runtime was allocating and zeroing
more than 100 MiB of fake host KV arrays after the real device cache was
already committed, solely so vector size could stand in for logical cache
rows. Decode then downloaded 1,971,200 KV bytes per step to keep those arrays
growing.

The release fix gives every layer explicit logical `cached_rows` and makes
host/device ownership explicit. A device-authoritative ring passes empty host
destinations to the backend; the backend then downloads no KV. Rewind restores
logical row/start metadata, and a path that needs host attention either has a
real host mirror or fails closed/re-prefills. The final page fell to the
191.332 ms median, and decode D2H fell from 1,971,200 KV bytes per step to zero;
the remaining 1,070,080 D2H bytes per step are vocabulary logits.

## Every page run and median

Order was candidate/control, control/candidate, candidate/control. All arms
used the same Release build, prompt rendering to exactly 128 tokens, one output
token, temperature zero, seed 33377335, checkpoint, clocks, power, device, and
memory budget.

| Arm | Every run, ms | Median ms | Median tok/s | Full spread |
|---|---:|---:|---:|---:|
| device page | 191.332, 191.075, 191.367 | **191.332** | **668.993** | 0.291 ms |
| serialized control | 6,157.919, 6,167.424, 5,875.770 | **6,157.919** | **20.786** | 291.654 ms |

The median improvement is **32.184x**. The 5,966.586 ms median gap is more
than 20 times the larger full observed spread, so it is not a variance call.
Every arm produced first token 3810.

Against experiment 0181, Strata reaches 75.88% of vLLM TP=1's comparable
881.67 tok/s. The remaining gap is real and tracked in
[issue #36](https://github.com/ro99/strata/issues/36); it does not invalidate
closure of the original implausible-ratio defect.

## Decode repetitions

The 31-token counting prompt requested 128 output tokens, providing 126 steady
batch-one forwards after the first decode step:

| Run | Steady seconds | Steady tok/s |
|---:|---:|---:|
| 1 | 4.238673 | 29.7263 |
| 2 | 4.232303 | 29.7710 |
| 3 | 4.235686 | **29.7472** |

Median decode is **29.747 tok/s**, 82.14% of vLLM's 36.214 reference and
1.650x experiment 0165's 18.03 tok/s baseline. All three 128-token ID vectors
have SHA-256 `03a3200b0a...b6954`.

Prefill costs 1.4948 ms/token and decode costs 33.6166 ms/token. The measured
prefill/decode per-token ratio is **0.0445**, well below the predeclared 0.25
architecture bound. Prefill is 22.49 times cheaper per token than batch-one
decode, which is the expected shape of a real page implementation.

## Rejected integration perturbations

- FP32 rather than FP64 RMS accumulation changed the real counting output to a
  degenerate separator sequence. The binding correctness gate read negative;
  it was rolled back immediately and no speed result was accepted.
- Fusing gate and up into one `N=43008` Marlin launch measured about 1.095 ms
  versus 1.087 ms for two launches. It was rejected before system integration.
- Moving fixed workspace allocation to initialization was retained as bounded
  setup hygiene but was within variance. It did not explain the 65--70 ms host
  term; the fake KV commit did.

## Correctness

The production-shaped Marlin test compares independent scalar and prepacked
uploads at Gemma's `[21504,5376]` and `[5376,21504]` shapes and requires maximum
relative residual below 1e-4 at M=1 and M=128. The device-only KV contract is
also exercised with empty host destination spans after a committed row.

The final candidate/control generation gate used the same 128-token page and
continued through decode. Both paths produced all 15 IDs identically, answer
`There are **100** "one"s in your text.`, and ID-vector SHA-256
`da4187122d...f40f49`. The two-turn chat smoke also matched forced full
re-prefill (`Paris.` then `The Seine.`); its existing template prefix detector
did not claim cache reuse, so no incremental-performance claim is made.

```text
strata-tests                  Passed  78.95 s
strata-sim-smoke              Passed   0.12 s
strata-equivalence-gemma4     Passed  23.17 s
100% tests passed, 0 failed
Total                         102.24 s
```

## Verdict and follow-up

**ACCEPTED.** Experiment 0165's open defect is closed: the prefill ratio is
explained, corrected, and outside variance. The bounded text-only first-page
executor is the default for prompts of at most 128 tokens on one GPU. Longer or
multimodal prompts retain the exact existing path; no silent fallback changes
precision or attention semantics.

The remaining work is performance parity, tracked in
[issue #36](https://github.com/ro99/strata/issues/36), not correctness of this
promotion: extend efficient page ownership beyond 128 first-page rows and
close the 668.99-to-881.67 prefill and 29.747-to-36.214 decode gaps under the
same locked operating point and numerical contract.

Raw model outputs remain ignored under:

- `results/gemma4-device-page-final/`
- `results/gemma4-decode-final/`
- `results/gemma4-device-page-generation-gate/`

Profiles and diagnostic gates remain outside Git under `/tmp/gemma4-*.log` and
`/tmp/gemma4-*.nsys-rep`.
