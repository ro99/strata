# Experiment 0194: DeepSeek pinned query page rejected

Date: 2026-08-25  
Branch: `fix/dsv4-pinned-query-page`  
Depends on: experiment 0190 cost model

## Decision

Do not register the reusable host query-page scratch. At the production
252 MiB transfer extent, registered D2H was slower than pageable D2H on the
critical physical GPU: 3.546 GB/s versus 5.606 GB/s. The cheapest mechanism
probe therefore falsified the hypothesis before runtime code was changed.

## Predeclared hypothesis and gates

- Hypothesis: query pages wider than 512 rows exceed the generic matmul's
  64 MiB pinned-staging ceiling, making `cudaMemcpyAsync` to the pageable
  runtime scratch block during issue. Register that one reused scratch so the
  same D2H remains asynchronous without a second bounce buffer.
- Target term: the 5.504 seconds of query-projection D2H inside the 8.068-second
  query phase at 1,925 prompt tokens. That attention phase was 42.248 seconds,
  almost tied with the 44.340-second MoE `argmax`.
- Primary metric: median D2H service at 252 MiB, followed by 619-token total
  prefill only if registration reduced the isolated term materially.
- Correctness: unchanged bytes and then `make check` plus the full DeepSeek
  oracle only if the mechanism screen passed.
- Memory ceiling: reuse the admitted 1 GiB host page workspace; add no bounce
  allocation and change no VRAM allocation.
- Rollback: registration failure, any transfer corruption, or less than a 2x
  isolated service improvement. Under the measured end-to-end model, a small
  link-rate movement could not clear the later 5% production gate.

The copy probe had no model setup and measured three repetitions in less than
ten seconds. A full 1,925-token model A/B was rejected as the first experiment:
it would spend about 106 seconds loading and 90--100 seconds in unrelated
attention, MoE, and mHC work per arm to answer a copy-mechanism question.

## Probe

`strata-topology-probe` used physical CUDA device 1 under
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, NUMA node 0, a 264,241,152-byte warm reused
buffer (1,925 rows x 64 heads x 512 columns x four bytes, rounded to 252 MiB),
and three repetitions per memory type and direction. The GPU remained at the
production 1,605 MHz / 250 W operating point. Unrelated topology-probe stages
were reduced to one 4 KiB sample.

| memory | direction | run 1 GB/s | run 2 GB/s | run 3 GB/s | median GB/s |
|---|---|---:|---:|---:|---:|
| pageable | D2H | 5.6062 | 5.6069 | 5.6011 | **5.6062** |
| registered | D2H | 3.5431 | 3.5461 | 3.5470 | **3.5461** |
| pageable | H2D | 5.6975 | 5.7304 | 5.7203 | 5.7203 |
| registered | H2D | 5.9009 | 5.8899 | 5.8942 | 5.8942 |

Every checksum verified. Registering the D2H source/destination pair cost
22.64 ms for the destination measurement. Raw output is ignored at
`results/0194-dsv4-pinned-query-page-probe.json`.

## Cost-model interpretation

The sign is negative on the target resource: registration reduces achieved
D2H throughput by 36.8%. H2D improves only 3.0%, but H2D is not the mechanism's
target and cannot compensate for slower D2H under the phase maximum. Transfer
volume, query kernel work, MoE, KV, mHC, routes, precision, and VRAM would all
remain unchanged.

The long-prompt problem is therefore not fixed by accelerating the existing
host boundary. Across 43 layers, the query projection returns roughly 10.8 GB
to host at 1,925 rows, performs host RMS/RoPE, and the attention command uploads
the BF16 image again. The next credible mechanism must remove that volume by
chaining query projection, exact normalization/RoPE, and attention on-device.

## Outcome

Rejected before implementation. No runtime or backend code changed, and no
full-model arm was launched after the microbenchmark gate failed.
