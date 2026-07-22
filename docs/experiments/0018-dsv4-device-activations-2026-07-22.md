# DeepSeek device activation experiment

Status: **rejected for promotion**. Exactness passed; throughput did not improve
outside observed variance.

## Contract

- Branch: `exp/dsv4-device-activations` from `main@099a1aa`.
- Hypothesis: keeping the exact `wo_a -> BF16 -> wo_b` attention-output
  projection chain in one bounded CUDA command will materially improve median
  decode throughput by removing one host round trip and one synchronization per
  layer.
- Primary metric: median decode tok/s over three ABBAAB interleaved repetitions.
- Correctness: identical generated tokens, ordered routes, logits, operation
  hashes, and layer hashes; zero decode checkpoint reads; balanced leases;
  `make check`.
- Memory: one explicit 1 MiB activation workspace per device inside the existing
  256 MiB reserve. The synchronous CUDA path remains the default oracle.
- Rollback: mismatch, I/O or memory violation, stream-order failure, or a
  variance-confirmed regression.

## Implementation screened

The candidate adds one backend-owned, two-buffer activation workspace per
device. `begin`, device matmul/grouped-matmul, and `collect` share the existing
nonblocking stream. Generation-tagged views reject stale or out-of-order use;
capacity exhaustion, overlap, CUDA errors, and non-finite output fail the
request. Weight leases remain alive until the completion event is observed.

Only the consecutive attention output projections use this path. Normalization,
RoPE, compression/indexing, mHC, routing, cross-device MoE accumulation, and
diagnostics still require exact host-visible values. Moving those boundaries is
not justified by this experiment.

## Correctness

The pinned one-pair full-model run is under
`results/deepseek-v4-device-activations-correctness/`.

- Generated tokens, ordered routes, full logit diagnostics, operation hashes,
  and layer hashes matched.
- Decode checkpoint reads were zero for both variants.
- All CUDA weight leases balanced and no non-finite values were reported.
- Peak RSS was 145,326,268 KiB for the oracle and 145,327,356 KiB for the
  candidate.
- Peak VRAM was 13,240/20,294/20,294 MiB for the oracle and
  13,242/20,296/20,296 MiB for the candidate on devices 0/1/2.
- `make check` passed: 91 unit/operation tests and the simulator smoke test.

The harness's generic 5 tok/s acceptance field is false for this two-token
correctness smoke; that field is not the correctness decision. Every exactness,
I/O, lease, and memory gate above passed.

## Interleaved measurement

The three-repetition ABBAAB matrix is under
`results/deepseek-v4-device-activations-performance/`. Each run generated ten
decode steps before the model's stop token.

| Decode metric | Synchronous oracle | Device chain |
|---|---:|---:|
| tok/s repetitions | 1.7883, 2.4056, 2.8782 | 2.3558, 1.7381, 2.4011 |
| median tok/s | 2.4056 | 2.3558 |
| median ratio | 1.0000 | 0.9793 |
| activation H2D bytes | 126,607,360 | 112,517,120 |
| activation D2H bytes | 137,605,120 | 123,514,880 |
| synchronization calls | 6,752 | 6,322 |
| median synchronization seconds | 0.3143 | 0.3789 |
| median kernel seconds | 0.4055 | 0.4054 |
| decode workspace allocation calls | 0 | 0 |
| median attention seconds | 0.6712 | 0.6510 |
| median MoE preparation seconds | 2.5364 | 2.6259 |

The candidate removed exactly 43 synchronizations and 1,409,024 H2D plus
1,409,024 D2H bytes per decode step. Whole-run workspace allocation medians
were 54 calls/5,427,256 bytes for the oracle and 60 calls/8,179,780 bytes for
the candidate; the candidate includes its persistent per-device activation
reservation. Decode itself allocated nothing.

Median RSS remained about 138.6 GiB. Candidate peak VRAM increased by 2 MiB per
device, matching the bounded workspace/accounting granularity. All six runs had
zero decode checkpoint reads, identical tokens and routes, balanced leases, and
RSS below the 216 GiB ceiling.

## Decision

Reject this runtime path for promotion. The 2.07% median regression is smaller
than the observed run spread and therefore is not a confirmed regression, but
the candidate also provides no material performance win. Keep the synchronous
path as the default exact oracle and do not extend this ABI across the graph
without a separately simulated, larger boundary-removal hypothesis.
