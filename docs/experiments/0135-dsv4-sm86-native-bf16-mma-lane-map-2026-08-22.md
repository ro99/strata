# Experiment 0135 — the shared SM86 native BF16 MMA lane/register prerequisite

Status: **C2 COMPLETE for the shared native-MMA fact.** On the identified
RTX 3090, `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` is verified
bit-exact against a host oracle, compiles to the intended
`HMMA.16816.F32.BF16`, and takes every operand from registers with zero spill
bytes, zero shared memory, and zero barriers. **No throughput was measured and
no throughput is claimed.** The FP4 portion of this probe is recorded as
preliminary FP4-specific evidence only; it is not an FP8 fact and it is not
F4-1.

## Question, contract, and budget

Hypothesis: SM86 exposes a native BF16 MMA whose lane/register-to-matrix
coordinates can be pinned exactly, which can be fed directly from registers
without a shared-memory staging tile or a materialized widened operand tile,
and which the toolchain lowers to the intended tensor instruction without
spilling.

This is milestone C2, the **shared** prerequisite of contract section 6. It is
shared only for the operand type and instruction that both tracks use. It
reduces no resource, targets no `argmax_r`, and authorizes no format-specific
conclusion.

Primary gate metrics, all of which had to hold simultaneously:

1. zero accumulator mismatches over all 128 D elements against a host oracle;
2. the intended `HMMA.16816.F32.BF16` present in SASS;
3. per-thread register count reported by both `ptxas` and the CUDA runtime;
4. zero stack frame, zero spill stores, zero spill loads;
5. zero `LDL`/`STL`/`LDS`/`STS`/`LDSM`/`LDGSTS`/`BAR` anywhere in the cubin.

Correctness gate: bit-exact equality, not a tolerance. All fixture stimuli are
small integers and exact powers of two, so every product and partial sum is
exactly representable in FP32 and the result is independent of summation
order. A tolerance would have hidden a lane-map defect; `mismatches` counts
exact inequality and `maximum_absolute` is reported alongside it.

Memory ceiling: 1,280 bytes of device allocation per MMA fixture, against the
campaign's 512 MiB ceiling. The exhaustive decoder fixture allocates a further
254,976 bytes.

Rollback: delete `apps/strata_dsv4_sm86_bf16_mma_probe.cu` and its CMake
entry. The branch carries no runtime change and no production dispatch.

Budget: each probe process is about 0.4 s wall, of which the timed device work
is a handful of microseconds — this is a correctness and code-generation gate,
not a timing run, so the setup-to-window ratio is not a defect here. The whole
experiment, including three interleaved processes, the cubin compile, the
disassembly, and the device-limit query, is under one minute. The cheaper
experiment rejected: reading the PTX ISA layout figures and asserting the map
from documentation. That was rejected because the contract forbids certifying
CUDA indexing, lane maps, inactive lanes, alignment, or memory paths from an
algorithm-only model. The more expensive experiment rejected: an Nsight
profile, which answers no C2 question because C2 makes no throughput claim.

## Device identification and the enumeration hazard

The contract's enumeration warning is real on this machine and was confirmed
rather than assumed:

| Tool | Index 0 | Index 1 | Index 2 |
|---|---|---|---|
| `nvidia-smi` | RTX 5060 Ti | RTX 3090 | RTX 3090 |
| CUDA runtime | RTX 3090 | RTX 3090 | RTX 5060 Ti |

The orders are not merely different, they are nearly reversed. The probe
therefore hard-fails on any device that does not report compute capability
8.6, and this was exercised: `--device 2` exits with
`error: BF16 MMA probe requires SM86`. Every recorded artifact carries
`device_name` and `device_capability` inline.

Measured device limits used below (`cudaGetDeviceProperties` /
`cudaDeviceGetAttribute` on the RTX 3090, not recited from documentation):
82 SMs, 65,536 registers per SM, 65,536 registers per block, 1,536 threads
(48 warps) per SM, 16 resident blocks per SM, 102,400 bytes of shared memory
per SM. Toolchain: nvcc 12.8, V12.8.93.

## What was actually run

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target strata-dsv4-sm86-bf16-mma-probe
for i in 1 2 3; do
  ./build-release/strata-dsv4-sm86-bf16-mma-probe \
    --device 0 --output results/qpn-sm86/0135-run$i.json
done
nvcc -arch=sm_86 -O3 -std=c++20 -cubin -Xptxas -v \
  -o probe.cubin apps/strata_dsv4_sm86_bf16_mma_probe.cu
cuobjdump -sass probe.cubin
```

Raw artifacts, all under the ignored `results/` tree:

- `results/qpn-sm86/0135-phase-a.json` — the preserved earlier BF16-only run.
- `results/qpn-sm86/0135-phase-b.json` — the preserved earlier run that added
  the exhaustive decoder and the FP4 fixture.
- `results/qpn-sm86/0135-run1.json`, `0135-run2.json`, `0135-run3.json` —
  three independent processes from this session.
- `results/qpn-sm86/0135-sass-audit.txt` — `ptxas` resource report, the
  tensor-instruction census, the memory-path census, and the full SASS of both
  MMA kernels.

The three new processes are byte-for-byte identical to each other, and their
correctness and decoder fields are byte-for-byte identical to the preserved
`0135-phase-b.json`. The pending evidence the contract told this session to
preserve is therefore validated rather than merely inherited.

## The shared C2 fact: the lane/register map

The instruction is
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, one warp, `M=16`,
`N=8`, `K=16`. The verified map is:

| Fragment | Registers per lane | Element | Row | Column |
|---|---|---|---|---|
| A (`.row`, 16x16) | 4 x b32, 8 bf16 | `i` in 0..7 | `lane>>2` plus 8 when `i` is 2, 3, 6, or 7 | `(lane&3)*2 + (i&1)` plus 8 when `i >= 4` |
| B (`.col`, 16x8) | 2 x b32, 4 bf16 | `i` in 0..3 | `(lane&3)*2 + (i&1)` plus 8 when `i >= 2` | `lane>>2` |
| C/D (16x8) | 4 x f32 | `i` in 0..3 | `lane>>2` plus 8 when `i >= 2` | `(lane&3)*2 + (i&1)` |

This is implemented by `pack_a`, `pack_b`, and `unpack_d` in the probe and
confirmed on hardware by two fixtures, over all 128 accumulator elements:

| Fixture | Stimulus | Maximum absolute error | Mismatches |
|---|---|---|---|
| `identity_unique` | A = I(16), B[k][c] = (k-7) + c/8, every B element distinct | 0.000000000 | 0 |
| `random_small_integer` | A in [-4,4], B in [-3,3], xorshift-seeded, deterministic | 0.000000000 | 0 |

`identity_unique` pins the composition of the three maps against a B matrix
with no repeated value, so an index collision cannot cancel.
`random_small_integer` then exercises a dense K reduction over all 16 terms.

### Residual freedom, stated honestly

A functional test of an MMA cannot distinguish three relabelings, because the
instruction is invariant under all of them:

1. a shared permutation of M between A and D (the 16 row dot products are
   independent);
2. a shared permutation of N between B and D;
3. a shared permutation of K between A and B (the K reduction is a sum, and
   with exactly representable stimuli it is order-independent).

No fixture, one-hot sweep included, can break these — they are a gauge of the
instruction, not a defect in it. A one-hot sweep was considered and rejected
for exactly this reason: it could not falsify anything the two fixtures do not
already cover.

What this means in practice: the map above is verified as a **self-consistent
triple**, which is precisely the property a kernel consumes, provided one
kernel uses all three halves of it. It matches the PTX ISA's documented
`m16n8k16` layout by inspection, which is what pins the gauge absolutely.

The gauge is not free everywhere. **The K relabeling becomes observable the
moment an external index is bound to the true K coordinate** — which is
exactly what an E8M0 group-32 FP4 scale and an E8M0 block-128 FP8 scale both
do. F4-1 and F8-1 must therefore pin K to the PTX coordinate in their prepack
and prove the scale-to-K binding independently. This experiment does not prove
it, and the FP4 fixture below deliberately does not test it.

## Code generation, registers, and the memory path

From `ptxas -v` and, independently, from `cudaFuncGetAttributes` at runtime —
the two agree exactly on register count:

| Kernel | Registers/thread (ptxas) | Registers/thread (runtime) | Stack frame | Spill stores | Spill loads | Shared bytes | Barriers |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bf16_m16n8k16_kernel` | 16 | 16 | 0 | 0 | 0 | 0 | 0 |
| `fp4_m16n8k16_kernel` | 28 | 28 | 0 | 0 | 0 | 0 | 0 |
| `exhaustive_decode_kernel` | 10 | n/a | 0 | 0 | 0 | 0 | 0 |

Whole-cubin instruction census: `HMMA.16816.F32.BF16` appears exactly twice,
once per MMA kernel, and it is the only `HMMA` form emitted. `LDL`, `STL`,
`LDS`, `STS`, `LDSM`, `LDGSTS`, `BAR`, `MEMBAR`, and `ATOM` counts are all
zero. The only memory traffic is the probe's own fixture I/O: 12 `LDG` and 9
`STG`.

The BF16 kernel's entire body is six `LDG` into R6-R11, one
`HMMA.16816.F32.BF16 R8, R8, R4, RZ`, and four `STG`. The A operand is the
four-register group at R8, the B operand the two-register group at R6, and the
accumulator is `RZ`. This is a register-fed MMA with no staging of any kind,
which is the shared fact C2 was required to establish.

### Occupancy: what the number does and does not say

The runtime reports 16 active blocks per SM and 33.3% occupancy for both MMA
kernels. **This is the 16-resident-blocks-per-SM hardware cap meeting a
one-warp-per-block probe, not register pressure.** 16 blocks x 1 warp = 16 of
48 possible warps. It must not be quoted as an occupancy result for a real
kernel; a production kernel needs at least 3 warps per block before the block
cap stops binding.

The register budget that *does* bind, measured on this device:

| Registers/thread | Warps/SM | Fraction of the 48-warp maximum |
|---:|---:|---:|
| <= 32 | 48 | 100% |
| <= 64 | 32 | 67% |
| <= 96 | 21 | 44% |
| <= 128 | 16 | 33% |
| <= 168 | 12 | 25% |

Feasibility read: a decode-plus-MMA warp costs 28 registers here, inside the
100%-occupancy band, but this is a single MMA with a single accumulator. A
real candidate carries N-tiled accumulators, multiple weight fragments in
flight, and split-K state. The table above, not the 28, is the budget F4-1 and
F8-1 must design against.

## Preliminary FP4-specific evidence — NOT a shared fact, NOT F8 evidence

The following is recorded as **FP4-track preliminary evidence only**, per the
contract's instruction to label it separately. It does not prove any
E4M3/E8M0 block-128 behavior, and no part of it may be cited in the FP8 track.

An exhaustive device-side decode of `decode_e2m1_pair_scaled`, which converts a
packed byte of two E2M1 codes plus an E8M0 exponent into a packed pair of
BF16 values by bit manipulation, was compared against a host oracle built from
an independent magnitude table and `ldexp`:

- 63,744 cases, being all 256 code bytes crossed with 249 E8M0 scale codes;
- 0 mismatches, reproduced identically in three processes.

A `fp4_e8m0_register_feed` fixture then fed that decoder's output straight into
the A operand registers of the same MMA, with no intermediate widened tile,
and matched the host oracle bit-exactly: maximum absolute error 0.000000000, 0
mismatches. In SASS, the four `LOP3` instructions producing R8, R9, R10, and
R11 are the four instructions immediately preceding the `HMMA`. That is
transferable-thesis item 3 — decoder output occupying operand registers with
no inner-loop shuffle and no materialized widened tile — demonstrated on SM86.

### Four limitations that F4-1 must close

1. **The tested E8M0 window is codes 2 through 250, not 0 through 255.** The
   decoder adds `(scale-1) << 7` to the BF16 exponent field. Codes 0 and 1
   drive the smallest E2M1 magnitudes into the BF16 subnormal range where the
   additive trick is invalid; codes 253 and above overflow the largest
   magnitudes to infinity or NaN, and 255 is the E8M0 NaN encoding. Codes 251
   and 252 are excluded conservatively rather than because they were shown to
   fail. This is **not** the exhaustive code/scale decode the FP4 gate
   requires. F4-1 must either extend the proof to the full range or carry an
   explicit admission check that rejects out-of-window scale codes — silently
   producing a wrong value for a checkpoint scale byte would be a correctness
   defect, not a performance question.
2. **The fixture binds one scale per A row, not per group of 32 along K.**
   With `K = 16` a whole row sits under one scale, so the group-32 boundary is
   never crossed and the scale-to-K binding is never exercised. Combined with
   the K gauge above, this is the single most important thing F4-1 has to
   prove, and 0135 does not prove it.
3. **One MMA is not a kernel.** There is no cold arena, no L2 scrub, no
   rotation, no split-K, no reduction, no production shape, and no timing.
   Nothing here is a bandwidth datapoint.
4. **The decode is not free, and its cost is measurable.** The FP4 kernel's
   SASS carries 47 `LOP3`, 32 `IMAD`, 15 `SHF`, and 10 `IADD3` around a single
   `HMMA`. Some of that is the fixture's address arithmetic and its B-operand
   packing, which a real kernel amortizes across N. But experiments 0130 and
   0131 found the FP4 SIMT arm ALU- and issue-bound, and this instruction mix
   is a warning that the decoder can reintroduce exactly that term. F4-1 must
   instantiate `tau = max_r(W_r/B_r) + Sigma_serial` at its own operating point
   and show the decoder's ALU cost does not become the new `argmax`. It may not
   inherit any constant from 0130, 0131, or this record.

## Cost model position

None. C2 measures feasibility and code generation. It reduces no resource,
names no target term, and changes no other resource, because it does not
change any production path. The contract requires `argmax` naming before a
mechanism is selected; no mechanism is selected here. F4-1 and F8-1 each owe
their own instantiation at their own operating points.

## Memory and residency

Peak device allocation is 1,280 bytes per MMA fixture and 254,976 bytes for
the exhaustive decoder fixture, against the 512 MiB ceiling. No persistent
weight representation exists in this probe, so the one-copy residency rule is
not yet under test. There is no widened persistent path, no duplicate probe
buffer, and no prepack workspace to declare.

## Gate verdict

| Gate | Required | Measured | Verdict |
|---|---|---|---|
| Accumulator oracle | 0 mismatches, all 128 elements | 0, two fixtures, three processes | PASS |
| Intended tensor instruction | `HMMA.16816.F32.BF16` | present, and the only `HMMA` form | PASS |
| Register count reported | yes | 16 and 28, ptxas and runtime agreeing | PASS |
| Spills | 0 stack, 0 store, 0 load | 0/0/0 on all three kernels | PASS |
| Unintended shared/local widened path | none | 0 `LDL`/`STL`/`LDS`/`STS`/`LDSM`/`LDGSTS`/`BAR` | PASS |
| Register/occupancy feasibility | measured | budget table measured on device | PASS |
| Device identity | RTX 3090, SM86 | verified inline, SM86 hard check exercised | PASS |
| `make check` | passes | 2 passed, 1 skipped, exit 0 | PASS |
| FP4 decoder evidence labeled separately | yes | labeled preliminary, four limitations recorded | PASS |

`strata-equivalence-gemma4` skips because its fixture model is absent; that is
the repository's existing behavior and unrelated to this change.

**C2 is complete for the shared native-MMA fact.** The shared instruction,
lane/register map, register-fed operand contract, and code-generation audit
are established on the target hardware. No FP4 or FP8 milestone advances on
this record alone.

## What this does not establish

- No throughput, bandwidth, or efficiency number of any kind.
- Nothing about E4M3 codes, E8M0 block-128 scales, or any FP8 operating point.
  Section 6 makes the lane map shared only because the operand type and
  instruction are shared; the FP4 decoder evidence above is explicitly not an
  FP8 proof, and F8-0 remains the required FP8 baseline.
- Nothing about production shapes, dispatch, admission, or the route census.
- Nothing that lets a candidate skip its own `argmax` profile.

## Exact next action

Two independent successors are now unblocked, and each must run its own gate:

- **F4-1**, which requires the group-32 scale-to-K binding of limitation 2, the
  full-range or admission-guarded E8M0 decode of limitation 1, an
  E2M1/E8M0 fragment prepack over the real
  `gate_up_w1 [N=2048,K=4096]` and `down_w2 [N=4096,K=2048]` shapes, and a
  fresh `tau` instantiation covering the decoder ALU term of limitation 4.
- **F8-0**, the FP8 region/shape/M/boundary census and independent W8A16
  baseline, which does not depend on F4-1 and is not blocked by D-F8-GATE.

**D-F8-GATE remains an open owner decision.** It blocks declaring F8-2 passed.
It does not block F8-0, and nothing in this record touches it.
