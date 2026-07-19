# Production FlashAttention decode optimization

Status: **exact production implementation accepted; default promotion pending
the replicated long-context gate**.

## Contract

- Hypothesis: parallelize exact QK row generation once, remove row barriers,
  reuse geometrically grown scratch, and combine pinned staging so CUDA wins at
  sufficiently large decode KV histories without taxing short decode.
- Primary metric: median decode tokens/s over three interleaved repetitions.
  Attention score/service time is the causal submetric; prefill is reported
  separately.
- Correctness gate: CUDA operation oracle, generated tokens, sequential routes,
  full logit/operation/layer hashes, finite status, and `make check`.
- Memory ceiling: existing 216 GiB host admission, per-device VRAM budgets, and
  the declared 256 MiB attention workspace. No persistent score tensor.
- Rollback: any oracle mismatch, unsupported silent path, non-finite result,
  checkpoint read, memory violation, or replicated regression.

## Implementation

The DeepSeek exact numerical contract remains sequential F64 dot, F32 score,
ordered F64 softmax, F32 probability, and ordered F32 V accumulation. CUDA now
computes independent key-row dots in parallel, stores only bounded transient
decode scratch, converts the scratch to probabilities once, and lets each
value lane accumulate without per-row block barriers.

Device upload, download, and score buffers grow at power-of-two boundaries
inside the request ceiling. Pinned host staging packs one H2D and one D2H
transfer per call. Device support is validated during initialization. The
DeepSeek adapter retains its 28-worker scalar kernel below 256 logical rows and
dispatches CUDA at or above 256; `--flash-attention-minimum-rows 0` forces CUDA
for diagnostics. Dispatch counts are explicit in JSON.

## Correctness evidence

`results/deepseek-v4-production-flash-decode-correctness-v3/` compares scalar
against forced CUDA after pinned staging. Generated tokens, route sequence,
logits, every operation/layer hash, checkpoint counters, cache leases, RSS, and
VRAM gates match. The CUDA test suite also covers the 128 + selected-512
DeepSeek layout and verifies that a 1..17-row decode sequence requires 10
geometric device allocations rather than growing on every row.

## Measurements

### Short forced-CUDA characterization

The pre-pinned three-repetition optimization matrix under
`results/deepseek-v4-production-flash-decode-performance-v2/` reported:

| Variant | Repetition tok/s | Median tok/s |
|---|---:|---:|
| Scalar | 2.011, 1.700, 2.582 | 2.011 |
| Forced CUDA | 2.115, 2.382, 2.248 | 2.248 |

The `1.118x` median is not a classified win because distributions overlap and
CUDA attention itself remained slower at 14 rows. After combined pinned
staging, one forced-CUDA correctness decode reduced maximum-device attention
service from about 3.07 ms to 2.00 ms and transfers from three H2D/two D2H to
one each, but its 12.65 ms graph attention time still exceeded the 5.91 ms
scalar path. This establishes the need for the crossover.

### Learned-index boundary

The one-pair 2,054-token boundary artifact is
`results/deepseek-v4-production-flash-boundary/`:

| Metric | Scalar | Forced CUDA | Change |
|---|---:|---:|---:|
| Prefill seconds | 499.39 | 444.71 | -10.9% |
| Prefill attention-score seconds | 129.02 | 92.68 | -28.2% |
| Decode attention-score milliseconds | 92.25 | 64.23 | -30.4% |
| Decode seconds | 1.016 | 1.276 | +25.6% |

Tokens and routes match and decode checkpoint reads remain zero. The decode
total is not an attention regression: unrelated host MoE preparation rose from
744 ms to 1,030 ms while device-MoE execution stayed at 45--46 ms. This single
pair validates the 640-row causal submetric and prefill hypothesis, not the
end-to-end promotion gate.

### Short hybrid A/A control

`results/deepseek-v4-production-flash-hybrid-performance/` executed zero CUDA
attention calls in both variants (602 scalar dispatches per run), yet measured
1.553 versus 2.317 median tok/s. The nominal `1.492x` difference is therefore
system/MoE variance, not an optimization win. It demonstrates why attention
counters and the replicated long-context matrix are required for attribution.

## Final replicated matrix and promotion decision

The completed three-repetition matrix used the same 2,054-token prompt, 10
requested output tokens, three GPUs, and identical reference/candidate binaries
with only the runtime flag changed. All repetitions exited successfully;
generated tokens and route traces matched, and decode checkpoint reads remained
zero.

| metric (median) | scalar reference | hybrid CUDA | change |
| --- | ---: | ---: | ---: |
| prefill seconds | 489.705 | 436.186 | -10.9% |
| prefill attention score seconds | 128.687 | 90.942 | -29.3% |
| decode attention score seconds | 0.829 | 0.537 | -35.3% |
| decode tokens/s | 1.288 | 1.148 | -10.9% |

The attention-only objective passes, while end-to-end decode does not: MoE
preparation variance dominates the short nine-step decode and the candidate
median is lower. The scalar global default remains in place; `--flash-attention`
enables the production hybrid policy (CUDA at 256+ logical rows, scalar below).
