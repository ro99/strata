# Experiment 0144 — exact SM86 FP8 register feed, then the binding small-shape ceiling

Status: **F8-1 primitive proof PASSES; F8-1 launch/wave gate is NEGATIVE. Stop
before the full QPN8 kernel.** F8-2 remains uncleared.

## Decision contract

- **Hypothesis:** checkpoint-native E4M3/E8M0 block-128 weights can be purely
  permuted and decoded directly into the C2 BF16 A-fragment registers, and an
  exact standalone or same-input-fused launch geometry can still fit the
  owner-bound 82%-of-local-ruler budgets for `wkv` and `wq_a`.
- **Primary metrics:** exhaustive BF16 decode mismatches; bitwise matrix
  mismatches against the same-MMA BF16-fragment control; independent canonical
  matrix error; prepack inverse mismatches; registers/spills/shared/barriers;
  and read-only upper-bound time/efficiency for 2.10, 4.19, and 6.29 MiB.
- **Correctness gate:** all 254 finite E4M3 codes crossed with all 255 finite
  E8M0 scale codes; both 128-element block axes crossed; 24 broad finite BF16
  activation patterns; zero bit mismatches against the fragment control; the
  independent canonical oracle within its declared FP32-association bound;
  deliberate K-scale, N-scale, and permutation defects must all fire.
- **Memory ceiling:** 512 MiB. Phase A peaks at 500,748 bytes. Phase B peaks at
  497,032,832 bytes (474.01 MiB), including the 256 MiB L2 scrub, 128 MiB
  ruler, rotated arenas, and sink.
- **Rollback/kill:** if either exact upper bound is negative, do not build the
  full QPN8 kernel and do not hide the protected shape behind a scalar route.

The target was physical GPU 1 / CUDA device 0,
`GPU-3032cfa3-19df-028f-5ebd-43314911e0b9`, an RTX 3090 at the campaign's
350 W, stock-unlocked operating point. The second RTX 3090 remained capped.

## Phase A — the exact primitive

The SM86 `m16n8k16` A map was re-derived from C2. Each lane owns eight E4M3
bytes, paired directly into its four b32 BF16 operand registers. Every aligned
16x16 weight fragment lies wholly inside one checkpoint 128x128 scale block:
one E8M0 byte therefore binds eight consecutive K fragments and eight
consecutive N tiles. The fixture is N=256, K=256 specifically so both axes
cross their block boundary.

The decoder carries QPN8's transferable word-parallel idea without porting its
Volta opcode or FP16 layout. Placing E4M3's `S/EEEE/MMM` fields directly in a
BF16 word represents exactly `E4M3 / 2^120`, including E4M3 subnormals. A
packed BF16 multiply folds `2^(scale-7)` and therefore the checkpoint E8M0
factor directly into the operand register. A high-scale path avoids premature
factor overflow. The two E4M3 NaN encodings and E8M0 code 255 are admission
errors; all finite combinations are covered.

Three independent processes produced byte-identical phase-A records:

| Gate | Result |
|---|---:|
| Finite decoder cases | 64,770 |
| Decoder mismatches | **0** |
| Prepack inverse mismatches | **0** |
| Register feed versus predecoded BF16 fragments | **0 / 2,048 bits** |
| Canonical BF16 matrix oracle violations | **0 / 2,048** |
| Maximum error / sum of absolute products | 2.74e-7 (bound 5e-5) |
| Broken K-scale control | 2,048 bit mismatches |
| Broken N-scale control | 1,024 bit mismatches |
| Broken permutation control | 2,048 bit mismatches |
| Candidate registers / local / shared | 25 / 0 / 0 |
| Active blocks per SM | 16 |

The prepack is an invertible pure permutation: 65,536 canonical code bytes
remain 65,536 bytes and four scale bytes remain four. The widened BF16 matrix
is a probe-only independent control, counted in peak memory, and is neither a
candidate representation nor a production path.

SASS contains `HMMA.16816.F32.BF16`, eight packed `HFMA2.BF16_V2` scale
operations in the loop body, and no `LDL`, `STL`, `LDS`, `STS`, `LDSM`,
`LDGSTS`, `BAR`, or `WARPSYNC`. Runtime attributes and `cuobjdump` agree on 25
registers, zero stack/local bytes, and zero shared bytes. This establishes
correct register feed, not decoder throughput.

## Phase B — cheapest launch/wave falsifier

The upper-bound kernel performs only ILP-4 coalesced `uint4` reads. It omits
all E4M3 decode, scale work, BF16 activation traffic, MMA, reduction, and
output publication. The matrix arms rotate across a 96 MiB arena after a 256
MiB L2 scrub. A same-session 128 MiB ruler establishes the denominator. Eight
geometries cross threads `{128,256}` with grids of `{1,2,4,8}` nominal
blocks/SM. Three warmed independent processes agree at the reported precision.

| Arm | Useful bytes | DRAM floor at 845.626 GB/s | 82% budget | Best read-only time | Best GB/s | Ruler efficiency | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `wkv` | 2,097,280 | 2.480 us | 3.025 us | **6.144 us** | 341.35 | **40.37%** | FAIL |
| `wq_a` | 4,194,560 | 4.960 us | 6.049 us | **9.216 us** | 455.14 | **53.82%** | FAIL |
| fused `wkv+wq_a` | 6,291,840 | 7.440 us | 9.074 us | **12.288 us** | 512.03 | **60.55%** | FAIL |

The empty launch is 3.072 us. It is already 0.047 us larger than the entire
`wkv` budget before one weight byte is read. Fusion amortizes that serial term
once but cannot close the gap. Using `tau = max_r(W_r/B_r) + Sigma_serial`,
the ruler makes DRAM the only volume term; the observed time beyond its ideal
DRAM term is 3.664 us (`wkv`), 4.256 us (`wq_a`), and 4.848 us (fused). The
target term is therefore launch/underfilled-wave serial cost, not compressed
weight volume. A register-fed decoder reduces the incumbent's issue/shared
work but cannot reduce this already-binding term.

This is a deliberately favorable upper bound, so adding the omitted decoder,
MMA, activation, and publication work cannot rescue it. The fixed setup is
under two seconds and the measured window is milliseconds; a full kernel was
the rejected expensive experiment because this read-only result already
falsifies it.

## Verdict and exact next action

The exact E4M3/E8M0 block-128 primitive is feasible and proven. The current
per-projection QPN8-derived execution hypothesis is nevertheless **REJECTED**
by the owner-bound small-shape gate. F8-2 is not attempted. No production
dispatch, persistent representation, or runtime fallback changed.

Do not lower D-F8-GATE, manufacture a favorable larger shape, or continue with
the full kernel. The next independent campaign work is F4-3. Resuming the FP8
track requires a newly authorized launch-free execution hypothesis—such as a
persistent executor or a materially broader operation fusion—with its own
production-semantics proof and freshly instantiated cost model. That is an
architecture expansion, not continuation of the rejected F8-1 kernel.

## Reproduction and evidence

```bash
cmake --build build --target \
  strata-dsv4-sm86-fp8-register-feed-probe \
  strata-dsv4-sm86-fp8-launch-ceiling-probe -j2
build/strata-dsv4-sm86-fp8-register-feed-probe --device 0
build/strata-dsv4-sm86-fp8-launch-ceiling-probe --device 0
make check
```

Ignored raw evidence is under
`results/dsv4-sm86-fp8-f8-1-phase-a/`: three phase-A JSON records, three
phase-B JSON records, SASS, resource usage, and the launch status record.
