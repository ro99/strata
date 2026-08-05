# Experiment 0053 — fused device attention chain, rejected on both gates

Status: **runtime wiring rejected and reverted; the backend primitive and its
fixtures are kept.** Measured 0.983x against a predeclared 1.15x gate, and the
generated token sequence diverges from the reference at token 33 of 64,
deterministically, despite the chain being bit-identical to the separate-call
sequence in four independent fixture combinations.

Per the charter, failed runtime code is not merged. `--device-attention` and
its runtime branches are reverted; `CudaBackend::deepseek_decode_attention_*`
and `tests/test_cuda_backend.cpp` remain, because the primitive passed its own
exactness gate and CUDA graph capture cannot be attempted without something of
its shape.

**What landed on main: this record only.** The primitive and its 350 lines of
fixtures stay on `perf/dsv4-device-attention-chain`. They are ~800 lines of
`backend.cu` with no caller and an unexplained token divergence in the wiring
that used them (below), so they are not carried onto the validated baseline
until whoever attempts graph capture needs them. Recover with
`git checkout perf/dsv4-device-attention-chain -- kernels/cuda/backend.cu
tests/test_cuda_backend.cpp`.

## Contract

- **Hypothesis.** Attention costs 70 ms/step with 2.4 ms of device time because
  each of wq_a, wq_b, wkv, wo_a and wo_b uploads, launches, downloads and
  synchronizes separately — 215 round-trips per step with host arithmetic
  between them. Carrying the query and output chains as two stream-ordered
  device commands removes most of that.
- **Primary metric.** Median decode ms/step, three interleaved repetitions,
  chat operating point, 64 generated tokens, no detailed timing.
- **Correctness gate.** Byte-identical generated token ids, compared as full id
  lists.
- **Memory ceiling.** One reusable 180 KB device workspace and one equal pinned
  host staging buffer per device. No change to admission or VRAM fraction.
- **Rollback.** `--device-attention`, default off.
- **Kill criterion, declared before the run.** Below **1.15x** median decode
  ms/step, reject rather than tune. Screened arithmetic beforehand: the chain
  replaces 56.3 ms of a 225.6 ms step, so 1.22–1.29x if it costs 5–15 ms.

## Result: 0.983x

| metric | reference | device | delta |
|---|---:|---:|---:|
| **ms/step (median of 3)** | **209.24** | **212.96** | **+3.72** |
| attention | 70.81 | 75.20 | +4.39 |
| — attn query (fused input chain) | 24.23 | 39.04 | **+14.81** |
| — attn kv | 6.57 | 3.51 | −3.06 |
| — attn output | 22.55 | 21.77 | −0.78 |
| moe total | 113.23 | 113.61 | +0.38 |
| **synchronizations/step** | **536.10** | **407.33** | **−128.76** |

reference: 208.0, 209.2, 222.5 · device: 212.7, 213.0, 223.8

**Rejected.** The gate was 1.15x; measured 0.983x.

## The finding that matters more than the rejection

**Removing 128.8 synchronizations per step made the step slower.** That
falsifies the premise this experiment and experiment 0050's item 2 were built
on — that the 341 serial projections cost what they cost *because* they
synchronize. They do not. Whatever attention's 70 ms is, per-call
synchronization is not a material part of it, and a mechanism whose whole
argument is "fewer round-trips" should not be expected to pay.

The `+14.81 ms` on the fused input chain names the likely reason, and it is a
constraint on the whole device-residency programme rather than a bug:

**Bit-exactness forces the reductions to be serial, and one GPU thread is much
slower at a serial reduction than one CPU core.** `rms_norm_f32` and the
per-head query normalization accumulate in `double`, sequentially, in ascending
index order. A parallel tree reduction is a different association and a
different result, so the device kernels reduce each row with a single thread —
1024 dependent double adds for the query rank, 512 per head — while the host
does the same work across 28 cores with AVX2. At batch 1 there are no rows to
hide that behind.

So: moving *matmuls* to a device-resident chain is neutral-to-positive, but
moving *exact reductions* there is a loss, and the reductions cannot be
parallelized without changing the numerics the charter forbids changing
silently. Any future device-resident or graph-captured step has to keep this in
view — the win has to come from somewhere other than round-trip removal.

## The correctness failure, unresolved

The device arm's tokens diverge from the reference at **token 33 of 64**,
identically in all three repetitions, so it is deterministic rather than a race.

What this is **not**, each ruled out by measurement rather than argument:

- **Not the projections.** `tests/test_cuda_backend.cpp` compares the fused
  chain against the separate-call sequence at four combinations — small and
  production shapes, crossed with FP4 and with the production encodings
  (FP8 E4M3 block-128 for wq_a/wq_b/wkv/wo_b, Plain BF16 for the converted
  wo_a). All four are bit-identical on query rank, queries, KV row and output.
- **Not a scalar/CUDA dispatch flip.** `flash_attention_minimum_rows` is 0 and
  `should_dispatch_flash_attention_cuda` reduces to `enabled`, so the CUDA
  branch is taken at every position; there is no crossover mid-generation.
- **Not the first tokens.** A two-token run matches exactly.

Two defects were found and fixed on the way, both of the class experiment 0051
records, and both invisible to a tolerance-based test:

1. **The bfloat16 round after every projection.** `Dsv4WeightCache::matmul`
   rounds on the host; the chain kept values on the device and skipped it.
   91 of 128 queries differed.
2. **`query_rank` read back after `wq_b`'s activation quantizer overwrote it in
   place.** The host never sees that because it uploads a copy. 56 of 64
   elements differed.

The remaining divergence is in the wiring, not the primitive, and it is left as
an open defect. The next attempt should bisect it with per-layer hashes rather
than by inspection — that was tried here by reading and it did not converge.

## Cost

Two probes and one A/B. The A/B was 6 arms at ~150 s, about 18 minutes, and it
was justified by a screened arithmetic ceiling of 1.22–1.29x. The preceding
smoke run already showed the input chain at 39 ms against a 33 ms reference;
that signal was correct and could have been read as a rejection 15 minutes
earlier, though at three decode steps it was not yet separable from noise.

## Artifacts

`kernels/cuda/backend.cu` (`dsv4_*` kernels, `deepseek_decode_attention_input`
and `_output`, `launch_weight_matmul`), `tests/test_cuda_backend.cpp`. Results
under `results/deepseek-v4-device-attention-ab/` (ignored). The reverted wiring
is commit `451117b`.
