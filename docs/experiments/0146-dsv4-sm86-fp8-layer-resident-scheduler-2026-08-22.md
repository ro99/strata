# Experiment 0146: SM86 FP8 layer-resident projection scheduler

## Result

The Ampere-native successor clears its first aggregate M=1 performance gate:
**699.67 GB/s, 83.27%** of the same-session local ruler across three
independent processes. One resident kernel consumes **115,350,400 useful
compact checkpoint bytes** through the real attention projection shapes and
performs real BF16 `q_a -> q_b` and `wo_a -> wo_b` dependencies. It overlaps
the dependency-independent `wkv` work with the ready `wq_b + indexer.wq_b`
stage. This is not synthetic matrix repetition.

F8-1 remains open on numerical acceptance. The random full-output fixture has
small deterministic BF16 publication differences from the FP64 decoded oracle;
the contract requires a real-fixture no-worse comparison to the incumbent
before production dispatch. The protected M curve also remains unmeasured.

## Contract and cost model

- **Hypothesis:** experiment 0145's compact kernel bodies are fast enough, but
  separate projection launches and underfilled grids prevent the small shapes
  from reaching equal-roofline efficiency. A resident grid with only true
  dependency barriers reduces `Sigma_serial` and schedules independent ready
  projections together.
- **Primary metric:** total useful compact FP8 checkpoint bytes divided by the
  complete one-launch scheduler time and same-session ruler.
- **Correctness:** unchanged E4M3/E8M0 block-128 storage, BF16 activation and
  publication boundaries, real intermediate BF16 dependencies, and an FP64
  decoded oracle at every emitted boundary.
- **Memory ceiling:** 512 MiB; measured allocation **522,338,568 bytes** versus
  the 536,870,912-byte ceiling.
- **Rollback:** no production dispatch and no F8-2 verdict if throughput is
  below 82%, any format/boundary changes, or the no-worse numerical gate fails.

The isolated projection sum contains four launch floors. The scheduler targets
that serial term and wave eligibility, not weight volume. It adds three global
dependency barriers, persistent CTA residency, and heterogeneous scheduling;
weight bytes, scale bytes, activation precision, and publication precision are
unchanged. Each complete process arm takes about three seconds including host
fixture/oracle work, so no longer system run was needed.

## Execution graph

The scheduler launches 492 CTAs: six resident 128-thread blocks per each of the
82 SMs. Its four stages are:

1. `(1024,4096)` `wq_a`, split-K 16, publishing the BF16 query-rank vector;
2. `(40960,1024)` main-plus-indexer `wq_b`, split-K 1, consuming that published
   vector, concurrently scheduled with `(512,4096)` `wkv`, split-K 16, which
   consumes the original layer input;
3. an explicit attention boundary followed by `(8192,4096)` `wo_a`, split-K 4;
4. `(4096,8192)` `wo_b`, split-K 8, consuming the published BF16 `wo_a` result.

The attention operation itself is outside this kernel probe, so its BF16 output
is injected at the declared stage boundary. The projection dependency ordering
and both representable projection-to-projection carriers are real.

## Three-process result

| Metric | Median / stable value |
|---|---:|
| Same-session ruler | 840.21 GB/s |
| Scheduler time | 164.864 us |
| Useful compact bytes | 115,350,400 B |
| Effective throughput | **699.67 GB/s** |
| Local-roofline efficiency | **83.27%** |
| Process ruler range | 840.21--841.05 GB/s |

All three processes returned identical numerical counts:

| Boundary | BF16 differences / outputs | maximum error / sum(abs(products)) |
|---|---:|---:|
| `wq_a` | 2 / 1,024 | 8.1e-8 |
| `wq_b + indexer` | 21 / 40,960 | 1.433e-3 |
| `wkv` | 1 / 512 | 9.7e-8 |
| `wo_a` | 1 / 8,192 | 5.99e-7 |
| `wo_b` | 4 / 4,096 | 5.053e-5 |

These are reduction-order differences, not decoder or scale-binding failures,
but that explanation does not waive the correctness gate.

The scheduler uses 72 registers/thread, 20 bytes static shared memory, zero
stack/local spill, and no widened weight tile. Nsight reports 78.55% DRAM
throughput, 49.41% active warps, 24.85% issue active, and 68.21% long-scoreboard
stall. The compact DRAM stream is now the largest resource term; the remaining
serial terms are the three required stage barriers.

## Verdict and next action

The persistent/fused architecture is performance-feasible on Ampere and the
unchanged 82% threshold did not move. Experiment 0144 remains a valid rejection
of isolated launches; experiment 0146 is the materially broader successor it
required.

Next, run the actual checkpoint weight/activation fixtures through both this
reduction order and the exact incumbent, compare maximum and RMS error at every
immediate BF16 production boundary, and stop if the candidate is worse. Only a
clean numerical verdict authorizes the M `{2,3,4,8,16}` scheduler curve.
