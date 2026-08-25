# Experiment 0195: DeepSeek device page-query chain accepted

Date: 2026-08-25  
Branch: `fix/dsv4-device-page-query`  
Origin: experiment 0190's production prefill cost model and experiment 0194's
rejection of a faster host boundary

## Decision

Keep a batched physical page's DeepSeek query projection on its owning GPU,
apply the declared BF16-boundary RMS normalization and forward RoPE there, and
feed the prepared query directly to physical-page attention. This removes the
full FP32 query-page D2H, host transform, and BF16 re-upload without changing
the single-row decode path.

At 1,925 prompt tokens, the candidate median was **26.231 prefill tok/s**
against **20.882 tok/s** for the branch's disabled-path control, a **1.256x**
median improvement. One candidate arm overlapped the fastest control because
its routed-expert demand wait was anomalously high; the record therefore does
not pretend the end-to-end distributions are disjoint. The deterministic
mechanism result is the removal of 10.84 GB D2H and 5.42 GB H2D per forward,
with bit-exact operation and full-model parity.

This is production-worthy progress, not closure of the vLLM gap. The freshly
measured vLLM median at the same long-prompt class was 127.490 tok/s.

## Predeclared hypothesis and gates

- Hypothesis: the 43 query projections' host boundary is an overlap and volume
  defect. Chaining query projection, exact normalization/RoPE, and attention on
  device will remove that serial boundary and materially improve production
  prefill.
- Target term: the query path inside the 42.248-second attention phase: 6.606 s
  issuing the projection, 5.504 s returning its 10.8 GB FP32 page image, and a
  subsequent BF16 upload after host RMS/RoPE.
- Primary metric: median 1,925-token prefill throughput over three interleaved
  repetitions per arm, plus activation transfer volume.
- Correctness gate: bit-exact operation fixture at the declared BF16 boundary;
  identical full-model token, logits, per-layer hashes, operation hashes, and
  routes against the disabled path; zero decode checkpoint reads; `make check`.
- Memory ceiling: the existing 0.95 admission, 22,135,873,536 bytes per RTX
  3090. No host allocation growth and no loss of expert-cache admission.
- Rollback: any numerical mismatch, VRAM admission failure, decode-path change,
  or a production median improvement inside observed control variance.

Before any arm, the predicted prefill/decode per-token cost ratio was below
0.25 because a page amortizes weight reads. The old 1,950-token production
baseline was 0.379. Using the accepted 26.231 tok/s prefill median and the fresh
8.627 tok/s short-context decode baseline gives 0.329. The sign improved, but
the prediction is still not met and the remaining gap is real.

## Governing cost model

Experiment 0190 instantiated

```text
tau = max_r(W_r / B_r) + sum(serial boundaries)
```

at the production operating point: `models/dsv4f`, rank-local TP2 on physical
devices 1 and 2 under `CUDA_DEVICE_ORDER=PCI_BUS_ID`, two RTX 3090s locked to
1,605 MHz / 250 W, 0.95 VRAM admission, 16K context, and an 8,192-token page
upper bound. Its 1,925-token decomposition was:

| phase | seconds | ms/token | controlling observation |
|---|---:|---:|---|
| embedding + mHC pre | 0.22 | 0.11 | negligible |
| attention | 42.25 | 21.95 | query host boundary dominates the 4.13 s sparse-attention kernel |
| MoE router | 4.35 | 2.26 | serial exact route construction |
| MoE prepare | 20.00 | 10.39 | 87.30 GB expert H2D, 19.97 s wait |
| MoE execution/join | 24.34 | 12.64 | 8.57 s max device command plus serial host collection |
| mHC post | 9.09 | 4.72 | device/host transition work |
| **total** | **100.42** | **52.17** | **MoE, 44.34 s, is `argmax_r`** |

The mechanism does not reduce the 44.34-second MoE `argmax`; it removes a
nearly tied attention serial term. That distinction matters. Routed-weight H2D
measured 4.37 GB/s aggregate against a 23.63 GB/s aggregate PCIe line rate.
Activation H2D measured 6.31 GB/s and D2H 3.47 GB/s. These low achieved rates,
combined with a full query image crossing the link twice around serial host
work, identify an overlap/ownership defect rather than proof that the links are
saturated.

The sign on other resources is bounded and favorable: expert weight volume,
MoE computation, routing, KV volume, mHC, precision, and decode are unchanged;
the GPU gains one existing FP8 tensor projection plus two small preparation
kernels per page, while host work and both link directions fall. Candidate
peak VRAM fell by about 226 MB per rank in the production arms rather than
approaching the ceiling.

## Experiment budget

The cheapest first test was an eight-token full-model parity arm and the
65,536-element operation fixture; both directly falsify arithmetic or layout
drift. A long multi-output chat was rejected because decode is excluded by the
`rows > 1` dispatch predicate. The production arm budget was about 3.2 minutes
per fresh process: roughly 102--104 seconds initialization plus 69--101 seconds
of measured prefill, a fixed/measured ratio of 1.1--1.5. Six interleaved arms
were budgeted at about 20 minutes and completed without inheriting another
experiment's prompt or output length.

## Production A/B

The order was A-B-B-A-A-B. Every arm used exactly 1,925 prompt tokens, generated
token 2107, no static expert tier, and the operating point above.

| order | arm | seconds | tok/s |
|---:|---|---:|---:|
| 1 | disabled-path control | 89.200 | 21.581 |
| 2 | device query | 89.155 | 21.592 |
| 3 | device query | 69.174 | 27.828 |
| 4 | disabled-path control | 92.185 | 20.882 |
| 5 | disabled-path control | 100.566 | 19.142 |
| 6 | device query | 73.386 | 26.231 |
|  | **control median** | **92.185** | **20.882** |
|  | **candidate median** | **73.386** | **26.231** |

The slow first candidate coincided with 32.56 seconds of routed-expert demand
wait, versus 14.04 and 16.81 seconds in the other candidates. That independent
host-tier variability explains the overlapping arm; query transfer removal is
deterministic and every paired later arm improved by more than 33%.

At the same prompt shape, aggregate activation traffic changed as follows:

| resource | control | candidate | change |
|---|---:|---:|---:|
| activation H2D | 16.711 GB | 11.289 GB | -5.422 GB |
| activation D2H | 23.512 GB | 12.668 GB | -10.844 GB |
| routed-weight H2D | 87.305 GB | 87.305 GB | unchanged |

A cheaper 619-token screen moved 15.18 to 17.10 tok/s, while the 133-token
smoke moved 5.96 to 7.84 tok/s. Those are single screens, not median headline
claims.

## Correctness

The CUDA operation fixture projects two rows through the real FP8 tensor-page
route, captures all 65,536 prepared BF16 query elements, and compares them bit
for bit with the prior host-visible projection plus sequential-FP64 RMS and the
declared RoPE/group layout. It passed exactly.

The binding full-model same-shape oracle reported identical generated token,
logits, layer hashes, operation hashes, and routes, with zero decode checkpoint
reads. A separate page-size-changing comparison was also attempted and failed
because the existing page-major expert reduction reassociates arithmetic when
page size changes; it is not used to launder or replace the same-shape gate.

Final repository gate: `make check` passed on 2026-08-25. The complete
`strata-tests` target passed in 127.64 seconds, `strata-sim-smoke` passed, and
the Gemma 4 equivalence target passed in 29.41 seconds: three of three CTest
targets, zero failures. The ignored log is
`results/0195-dsv4-device-page-query/make-check.log`.

## Outcome and remaining gap

Accepted. Batched physical-page prefill now owns query projection through
attention on the GPU. Batch-one decode takes the unchanged legacy path.

Fresh external reference medians on the same two RTX 3090s were 8.439, 41.901,
and 127.490 prefill tok/s at about 36, 500, and 1,950 tokens, and 10.346 decode
tok/s for a 128-token response after a 19-token prompt. Strata after this change
has a 26.231 tok/s median at 1,925 tokens and retains the 8.627 tok/s short-
context decode baseline. Reaching vLLM-equivalent prefill remains future work;
this experiment lands the safe ownership correction without claiming that the
larger MoE and attention costs are solved.
