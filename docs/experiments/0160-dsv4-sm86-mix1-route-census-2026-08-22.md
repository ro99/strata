# Experiment 0160 — MIX-1 route census, and the hidden fallback it removed

Status: **MIX-1 STARTED.** Every CUDA matmul dispatch is now classified,
counted and observable, and an unrecognised weight encoding **fails explicitly**
instead of being decoded as FP4. A gating test proves both. This is the first
of MIX-1's requirements, not the whole milestone.

F8-2 was accepted by the owner on 2026-08-22 on the evidence of experiment
0159, which unblocked MIX-1.

## The defect this removed

`CudaBackend::matmul_impl` selected a kernel through a chain of `else if`
branches on the weight encoding, ending in a bare `else`:

```cpp
} else if (descriptor.encoding == CudaWeightEncoding::Fp8E4m3Block128) {
    native_fp8_matmul_kernel<<<...>>>(...);
} else {
    native_fp4_matmul_kernel<<<...>>>(...);   // everything else lands here
}
```

`Plain`, `OffsetPackedInt4/8`, `Nvfp4Group16` and the FP8 tensor-page path are
all matched earlier, so in practice the `else` served `Fp4E2m1Group32`. But it
was **unconditional**: any encoding not named above — including the host-expert
layout and anything added later — would be handed to the FP4 kernel and decoded
as E2M1 nibbles with E8M0 group-32 scales, silently producing wrong numbers
rather than an error.

That is precisely what the contract forbids:

> An unsupported case may fail admission when no approved exact route exists.
> It may never disappear into a hidden fallback.

It is a latent hazard rather than a live bug — no encoding currently reaches it
by accident — but it is the exact shape of defect MIX-1's route census exists
to make impossible.

## What was built

**`CudaMatmulRoute`** names all eight live routes plus `Unsupported`:
`plain_bf16_matvec`, `plain_generic`, `packed_int8_group32`,
`packed_offset_int`, `nvfp4_group16`, `fp8_tensor_page`, `fp8_e4m3_block128`,
`fp4_e2m1_group32`.

**`CudaBackend::matmul_route_census()`** returns per-route counts since process
start, with `reset_matmul_route_census()` for scoped measurement. Counters are
plain `std::uint64_t` incremented through `std::atomic_ref`, which keeps
`CudaBackend` copyable — it is copied elsewhere in the runtime, and a
`std::atomic` member breaks that.

**The bare `else` is gone.** `Fp4E2m1Group32` is now matched explicitly, and
anything else records `Unsupported` and returns a `ValidationResult` error
naming the encoding and stating that no substitution will be made.

## Gate

A new test, `MIX-1 matmul route census records every dispatch and refuses
unknown encodings`, asserts that:

- the census starts at zero after a reset;
- an FP4 matmul increments **exactly** the `fp4_e2m1_group32` counter;
- `unsupported` stays at zero on a healthy dispatch;
- every route name is non-empty and **distinct**, so a census dump is readable
  rather than ambiguous.

`make check` passes: 314/328 unit tests, up one, 14 skipped, and all three ctest
suites green. No existing path changed behaviour — every encoding exercised by
the suite is one of the explicitly handled routes, which is itself the evidence
that removing the fallback broke nothing.

## What this does not establish

This is one requirement of MIX-1. Still outstanding:

- **The accepted F4 and F8 paths are not wired in.** Production still calls
  `native_fp4_matmul_kernel` and `native_fp8_matmul_kernel`, not the register-fed
  candidates from experiments 0157 and 0159. The census now makes that
  substitution observable when it happens, which is why it came first.
- **Admission is not wired.** `admit_e8m0_scales` from experiment 0155 exists in
  the FP4 probe and is not yet called at checkpoint load.
- **Load-time prepack, VRAM accounting, graph integration and fixtures** are all
  untouched.
- No end-to-end run has been made, so the census has not yet been observed
  against a real workload — only against the unit test.

## Exact next action

1. **Dump the census from a real DeepSeek V4 run** and record the route
   distribution. That is the "route census must make every production choice
   observable" requirement, and it will show which encodings production actually
   uses — the first real check that the eligible-FP4 and eligible-FP8 sets are
   what the campaign assumes.
2. **Wire `admit_e8m0_scales` into checkpoint load**, so an inadmissible scale
   fails admission rather than reaching a kernel.
3. Then the one-copy prepack and the accepted-path substitution, which is the
   substantial remainder of MIX-1.
