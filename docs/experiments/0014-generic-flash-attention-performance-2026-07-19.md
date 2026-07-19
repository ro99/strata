# Generic FlashAttention replicated performance gate

Status: **gate completed; promotion rejected by measured regression**.

## Measurement contract

- Same pinned DeepSeek-V4-Flash-DSpark checkpoint and binary revision.
- Scalar and FlashAttention variants both used device MoE, 28 resident-read
  workers, 28 host-attention workers, identical devices `0,1,2`, VRAM fraction
  `0.85`, host ceiling 216 GiB, context ceiling 2,048, detailed timing, and
  greedy decoding.
- Three repetitions were interleaved ABBAAB by the reusable
  `scripts/run_deepseek_v4_device_moe_ab.sh` harness, with nine decode steps
  per run and zero decode checkpoint-read bytes.
- Primary metric: median decode tokens/s. Correctness gates required identical
  prompt/generated tokens and route traces, balanced cache leases, bounded
  host memory, and successful exact execution.

Artifact: `results/deepseek-v4-flash-attention-performance-3rep/`.

## Result

| Variant | Repetition tok/s | Median tok/s |
|---|---:|---:|
| Scalar | 2.675, 2.809, 2.319 | 2.675 |
| FlashAttention | 1.763, 1.391, 1.258 | 1.391 |

The candidate/scalar median ratio was `0.520x` (approximately 48% slower).
All three pairs had identical tokens and route traces. The candidate also
reported zero wasted staging bytes and zero decode checkpoint reads. In one
representative candidate decode, FlashAttention issued 387 calls, transferred
59,566,848 H2D bytes and 50,726,412 D2H bytes, and used 0.1915 seconds of
kernel time; host packing, synchronization, and per-call transfer overhead
dominated the end-to-end result.

## Decision

The remaining requirement is closed as a failed promotion gate, not as a
performance win. The scalar path remains the default. FlashAttention remains
available explicitly for further optimization work; any future promotion must
first remove the per-call overhead and repeat this same interleaved matrix with
a median beyond observed variance.
