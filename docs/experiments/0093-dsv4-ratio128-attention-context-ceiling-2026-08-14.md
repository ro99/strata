# Experiment 0093: ratio-128 attention growth, and the 65,536-token ceiling

Date: 2026-08-14
Branch: `feat/dsv4-rank-local-decode`
Base: `db05e15`
Disposition: **CAPABILITY REJECTION — the declared 1,048,576-token context is
not executable. The supported context ceiling of the device attention path is
65,536 tokens. Tracked as issue #22.**

## Question and gates

Every other context-dependent term in a DSV4 decode step had been measured. The
attention term had not. The question was how much the 20 ratio-128 layers cost
at long context, so that `tau(1M)` could be assembled from measured parts rather
than assumed ones.

The gate was ordinary: measure the term at the production page geometry across
the contexts that matter, on an idle device, three interleaved repetitions, and
report medians. There was no throughput hypothesis to defend — this was the last
unmeasured term in the model, not a proposed mechanism.

## Bounding the question before measuring it

Reading the runtime's candidate assembly (`attention_physical`,
`src/deepseek_runtime.cpp:3552`, and the rank-local path at `:6570`) narrows the
question to one layer class before any GPU work:

```text
ratio 0    layers   0 compressed + 128 sliding                 = 128    fixed
ratio 4    layers   512 index top-k + 128 sliding              = 640    fixed
ratio 128  layers   roundup(context/128, 128) + 128 sliding    = grows
```

`sliding_window` is `128`. The ratio-4 and ratio-0 layers are therefore already
at their 1,048,576-token widths in every short-context arm ever run on this
branch, and the 20 ratio-128 layers are the entire context-dependent attention
term. That is the whole of what needed measuring.

## Measurement

`apps/strata_dsv4_attention_probe.cpp` reproduces the production shape: the
`Hca` page geometry (2 rows per block, 584 B per row), the production candidate
layout with its invalid padding region, and one sliding page. Device 1 (RTX
3090, sm86), Release `build-landing`, nine repetitions per point, three
interleaved runs, idle GPU. Medians:

```text
context    comp rows   cands   pages   ms/layer   ms/token   kernel ms   status
    4,096         32     256      17      0.159      3.188       0.043   ok
   16,384        128     256      65      0.158      3.167       0.043   ok
   32,768        256     384     129      0.170      3.400       0.049   ok
   49,152        384     512     193      0.177      3.537       0.056   ok
   65,536        512     640     257      0.186      3.729       0.064   ok
  131,072      1,024   1,152     513          -          -           -   REJECTED
  262,144      2,048   2,176   1,025          -          -           -   REJECTED
1,048,576      8,192   8,320   4,097          -          -           -   REJECTED
```

Artifacts: `results/dsv4-rank-local-main-landing/step4-ratio128-attention-growth/probe-r{1,2,3}.txt`,
SHA256 `8ac542e7a96ad85a113f56fff00f75039e9a6cc01efc882f4986e84e4b7504e6`,
`64171af7208db0888323d071434f58c5ba52d00af9ce5c5ff9ec6534f709a60d`,
`9bee399706a5e452d71d4696b0fbc4a25a80b488ce7a17dc3e3975d8dcaf396d`.

## Result

The result is the last three rows. `dsv4_paged_attention`
(`kernels/cuda/backend.cu:7545`) and `dsv4_paged_attention_to_mhc`
(`:9278`) both validate `candidates > 640U` as invalid. A ratio-128 layer
crosses 640 candidates as soon as `compressed_width > 512`, that is at
**65,537 tokens**, and needs **8,320** at the declared context.

The five passing rows are the control that identifies the cause. Queries, sinks,
scale and entry point are identical in every row; `1,152` is a valid multiple of
the required `128`; so `candidates > 640U` is the only clause of the compound
validation that can fire.

A second ceiling sits behind the first: the kernel stages KV as
`candidates * 512 * sizeof(uint16)`, and production configures a `4 MiB`
workspace (`src/deepseek_runtime.cpp:3606`). That staging is `655 KB` at 640
candidates and `8.52 MB` at 8,320. Lifting the candidate cap alone would move
the rejection rather than remove it.

A reporting defect rides along. Nothing between the configured
`maximum_context_tokens` and the attention call bounds the context, so a run at
100,000 tokens does not fail admission — it reaches the first ratio-128 layer
and reports *"DeepSeek paged attention request shape, BF16 query, scale, or sink
is invalid"*. It fails closed, which is the contract, but it names the query and
the scale for what is actually an unsupported context length.

## What this does not establish

Extrapolating the kernel's linear fit (`5.469e-5 ms/candidate`, intercept
`0.029 ms`; predicting `0.0500` and `0.0570` against `0.049` and `0.056`
measured at the interior widths) to 8,320 candidates gives `0.484 ms/layer`,
about `9.7 ms/token` over 20 layers, a growth of roughly `+8.8 ms/token`.

**That is an extrapolation 13x beyond the measured range and is not a result.**
It is also a linear fit to a kernel that cannot run at the extrapolated point:
at 8,320 candidates the staging no longer fits the workspace this kernel is
written against, so whatever eventually runs there will not be this kernel
unmodified. Two extrapolations in the landing tracker have already been
withdrawn after measurement contradicted them; this one is recorded as a
planning figure that the replacement kernel's own microbenchmark must supersede.

## Process note, recorded because the error was ours

A fourth probe pass was taken while `ctest` held the same device. It reported
`0.083 ms` at 512 candidates against `0.067` at 640 — not a monotone function of
candidate count, which a candidate-loop kernel's time must be. It was discarded
rather than averaged in. The cheapest check on a contended measurement is
whether it still has the shape the mechanism requires.

## Consequences

- The declared 1,048,576-token context is not slow; it is unexecutable, on both
  topologies. No `tau(1M)` can be measured until the kernel is replaced.
- This is unrelated to the Lightning Indexer, in-chain selection, and rank-local
  decode. The index path selects 512 candidates at any context and never
  approaches the cap; both topologies are affected identically.
- It is unrelated to prefill throughput. The rejection fires on the first
  ratio-128 layer of the first decoded token, however the context was populated.
  Prefill work does not unblock it.
- Lifting the ceiling is **necessary but, on present evidence, not sufficient**
  for a 1M throughput claim: the standing arithmetic with the extrapolation
  above lands near `149.8 ms/token` against a `127.000` review ceiling, with
  pre-leasing, the page-descriptor upload, and base-term growth charged nowhere.

## Follow-up

Issue #22. The shape of the fix is a tiled or multi-pass dense attention over an
unbounded compressed history carrying an online softmax, so the KV staging is
sized by tile width rather than candidate count. Its acceptance bar is bit
identity on generated token IDs against the current kernel inside the supported
range, screened before integration, because a tiled rescaling changes the order
in which partial sums combine and that order is part of the declared contract.
An explicit context bound that reports the real cause is worth landing even if
the kernel work is deferred.
