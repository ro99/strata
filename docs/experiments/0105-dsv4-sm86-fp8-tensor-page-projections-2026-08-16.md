# Experiment 0105 — DSV4 SM86 FP8 tensor page projections

Status: **primary device-time and correctness gates passed; preserved for
report-and-discuss.** The same-build query-projection device term fell from
7.158365 s to 0.338589 s (21.14x). One-arm total prefill improved by 9.445 s,
but that is explicitly directional and below the measured approximately 16 s
NUMA-placement noise floor of the routed-expert arena.

## Lineage and predeclared decision

Experiment 0101 established that tensor-core GEMM is much faster than the
hand-written row kernel at the dense page shapes, but experiment 0102 stopped
before integration after proving that production projections are FP8, not
plain BF16. Experiment 0103 then reproduced the production FP8 query device
counter within 0.8%, measured 677x incumbent logical weight-read amplification,
and screened a decode-E4M3-to-BF16 WMMA tile at 6.50--23.62x faster. Its stronger
FP32 no-worse oracle gate failed. Experiment 0104 verified the actual immediate
BF16 publication boundary in source and showed the tensor path is no worse
against the FP64 oracle at the precision consumed by the model.

0105 integrates only multi-row DeepSeek attention-page projections with FP8
E4M3 block-128 weights. The hypothesis is that removing the incumbent's
per-batch-row weight rereads reduces the measured 7.159487 s query matmul device
term. The primary metric is same-build
`attention_query_matmul_kernel_seconds`; total prefill is not a decision metric
for single arms. Correctness requires the production-boundary fixture,
generated IDs `2107, 8777, 1277, 440`, zero decode checkpoint reads, and
unregressed decode. The additional device-workspace ceiling is 384 MiB/device.
Any gate result is preserve-and-report, not automatic discard.

The authorized budget was implementation and fixture work followed by one
baseline and one candidate arm at 677 tokens, page 8192, approximately four
minutes each. No 2,612-token arm, repetition or fixed/marginal fit was run.
Actual arm wall times were 2:53.54 and 2:38.28.

## Implementation and capability contract

The page path explicitly requests the tensor implementation for `wq_a`,
`wq_b` and `wkv`. Dispatch selects it only when all of these hold:

- the caller explicitly requested the DeepSeek page path;
- the device is exactly SM86, the capability screened in 0103/0104;
- there is more than one activation row and no grouped projection;
- the weight encoding is FP8 E4M3 block-128; and
- K is divisible by 128 and N by 128.

SM80, SM89, SM90, SM120 and all other capabilities retain
`native_fp8_matmul_kernel`; so do every single-row call and all non-FP8 or
grouped shapes. `dsv4_fp8_tensor_page_supported()` makes the capability choice
directly testable. The CLI's paired enable/disable flags permit a same-binary
baseline and candidate.

The source audit corrected one simplification in the standalone screen:
production activation simulation has a power-of-two scale for every row and
K128 block. The integrated quantizer writes one E4M3 byte per activation and
one E8M0 byte per block. The tensor tile applies that activation scale while
widening to BF16, then applies the checkpoint weight scale after each K128 dot.
The original FP32 input is uploaded into the eventual output allocation,
compacted into the byte buffer, and overwritten by the result. There is no
parallel FP32 encoded-value workspace and no expanded persistent weight.

The largest projection workspace is 95,069,344 bytes (90.665 MiB) per device:
2,794,656 bytes of compact activation plus scales and 92,274,688 bytes for the
padded `wq_b` output. This is below the 384 MiB ceiling. The immediate on-device
BF16 rounding kernel and all downstream consumers are unchanged.

## Correctness fixture

`tests/test_cuda_backend.cpp` exercises the actual M=677 production shapes:

| Projection | M | N | K |
|---|---:|---:|---:|
| `wq_a` | 677 | 1,024 | 4,096 |
| `wq_b` | 677 | 32,768 | 1,024 |
| `wkv` | 677 | 512 | 4,096 |

The fixture uses deterministic non-degenerate E4M3 activations and weights,
E8M0 weight scales, the production activation quantizer, and the production
BF16 publication boundary. On 4,096 deterministic outputs per complete shape,
the tensor path must have no more oracle mismatches and no worse maximum
absolute, maximum relative or RMS error than the incumbent after rounding the
FP64 oracle through the same FP32/BF16 boundary. All assertions pass. The test
also permits and bounds the rare different BF16 code selected by reassociation;
the binding contract is no-worse against truth, not path-to-path bit identity.
A separate single-row assertion proves that even an explicit tensor request
falls back to the incumbent kernel exactly.

`make check` passed before the implementation checkpoint commit and again
before this result commit. Both model arms generated exactly
`2107, 8777, 1277, 440`; both reported zero decode checkpoint-read bytes.
Decode improved from 8.748 to 9.293 tok/s, so it is unregressed.

## Same-build arm result

Both untraced arms used the same preserved binary (SHA-256
`b9baf60dfbc7f9589e8e6118b4f07e0a332337a7727af1430b2e85040875a751`),
devices 1 and 2, 677 prefill tokens, page 8192 and four decode tokens.

| Metric | Baseline native FP8 | SM86 tensor | Change |
|---|---:|---:|---:|
| **Query matmul device** | **7.158365 s** | **0.338589 s** | **21.14x faster** |
| Query bucket | 10.656438 s | 3.515625 s | -7.140813 s |
| KV matmul device | 0.322875 s | 0.051691 s | 6.25x faster |
| KV bucket | 4.246564 s | 3.585505 s | -0.661060 s |
| Score bucket | 13.975671 s | 13.960262 s | -0.015409 s |
| Attention total | 29.265366 s | 21.423619 s | -7.841747 s |
| Total prefill (directional) | 61.003811 s | 51.558812 s | -9.445000 s |
| Prefill tok/s (directional) | 11.0977 | 13.1306 | 1.183x |
| Decode tok/s | 8.7476 | 9.2931 | 1.062x |

The baseline device term is within 0.02% of 0100's 7.159487 s, and the
candidate's 0.338589 s matches 0103's approximately 0.34 s screen. This closes
the mechanism-to-production transfer cleanly. The 9.445 s total-prefill change
is in the expected direction but remains below the measured 16 s NUMA noise
floor and carries no promotion or rejection verdict.

## Resource accounting

| Resource | Baseline | Candidate | Sign |
|---|---:|---:|---:|
| RSS | 158,858,878,976 B | 158,858,760,192 B | effectively unchanged |
| GPU 1 VRAM | 22,994,354,176 B | 22,988,062,720 B | -6,291,456 B |
| GPU 2 VRAM | 22,916,759,552 B | 22,910,468,096 B | -6,291,456 B |
| Activation H2D | 5,874,353,192 B | 5,874,353,192 B | unchanged |
| Activation D2H | 8,261,985,796 B | 8,261,985,796 B | unchanged |
| Cache hits | 85,230 | 85,230 | unchanged |
| Cache misses | 11,024 | 11,066 | +42 |
| Cache evictions | 7,842 | 7,884 | +42 |
| Demand H2D | 74,083,499,520 B | 74,380,770,816 B | +297,271,296 B (0.40%) |

The mechanism reduces logical projection weight reads 61.55x and replaces
CUDA-core scalar reduction with BF16 WMMA; measured query device service falls
21.14x. It adds exact E4M3 decode, 48 KiB shared memory per CTA and higher
register pressure. Persistent FP8/E8M0 weight bytes, precision, cache capacity,
router/top-k/expert semantics and expert residency do not change. Compact
activation storage lowers measured per-GPU VRAM by 6.0 MiB rather than growing
it. The small cache-miss/eviction and demand-H2D increases are routed-expert
placement variation; the projection does not alter cache policy or weight
residency, and these counters explain why total prefill remains directional.

One scoped observation only: `native_fp4_matmul_kernel` retains the same one-
block-per-`(output_row,batch_row)` grid family as the incumbent FP8 kernel. It
was not investigated or changed in this experiment.

## Verdict

0105 passes its primary value and correctness gates. The production mechanism
lands the isolated 21x query-device improvement without extra persistent
weights, duplicate encoded activations, decode regression or a looser carried-
precision contract. The implementation, runner, binary and raw JSON/logs are
preserved on the experiment branch. Promotion measurements or work on any
other term require a new authorization.
