# DeepSeek V4 exact compact KV/index storage

Status: **correctness accepted; performance not classified**.

## Contract

- Hypothesis: storing DeepSeek's already-declared FP8/BF16 KV values and FP4
  learned-index values in their native physical encodings reduces resident
  cache bytes without changing the scalar oracle's numerical values.
- Primary metric: physical cache bytes per context token. Correctness is the
  release gate; throughput is descriptive until three interleaved repetitions
  are available.
- Correctness gate: independent codec round trips, rejection of any lossy
  compact write, versioned block-header and corruption checks, `make check`,
  and a full-model learned-index-boundary scalar/block comparison.
- Memory ceiling: physical payload plus 64-byte block metadata and alignment,
  bounded by the admitted host cache budget.
- Rollback: scalar F32-backed mode remains the default; reject compact mode on
  any numerical mismatch, accounting error, or variance-confirmed regression.

## Implementation

Sliding, CSA, and HCA rows use FP8 E4M3 per-64 non-RoPE values, E8M0 scales,
and BF16 RoPE values. Learned-index rows use packed FP4 E2M1 per-32 values and
E8M0 scales. The encoder verifies that every decoded value is bit-identical
to the input and refuses values that would require an additional rounding.

Blocks carry a versioned 64-byte header, physical/payload byte counts, and
shape metadata. Row reads decode only requested rows; selected attention rows
are materialized for the existing scalar attention oracle, while learned-index
scoring decodes one row at a time. Device promotion transfers the versioned
physical block bytes.

`--scalar-kv-cache` (also the default) retains the F32-backed oracle.
`--block-kv-cache` opts into compact physical storage.

## Validation

`make check` passed both CTest targets: 96/96 unit cases and the simulator
smoke. The 2,054-token boundary A/B used the same frozen runner, model,
devices, budgets, prompt, and deterministic decode:

- generated tokens, logits, layer hashes, operation hashes, and route traces
  matched exactly;
- scalar/block prefill was 461.669/609.507 seconds and decode was
  1.034/1.277 seconds (one pair; descriptive only);
- compact block mode allocated 1,817 blocks, retained 527, peaked at
  13,467,712 bytes under its 19,611,136-byte host cap, and recorded zero
  misses;
- compact gather bytes were 13,030,377,744 and host writes were 58,716,578;
- the 1,048,576-token compact admission plan reports 3,707,490,816 bytes of
  KV/index state versus 14,451,884,032 bytes for the F32-backed plan.

Evidence is under
`results/deepseek-v4-compact-cache-index-boundary/` and
`results/deepseek-v4-compact-cache-correctness/`. These are one-pair A/B
measurements; no performance win is claimed.
