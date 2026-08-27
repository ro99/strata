# Vendored Marlin CUDA subset

This directory is an intentionally vendored, source-level dependency. It is
not Strata product code and must not be included directly outside the adapter
at `kernels/cuda/detail/marlin_adapter.cuh`.

## Provenance

- Source distribution: vLLM 2.3.8
- Source path: `csrc/libtorch_stable/quantization/marlin/`
- Strata import commit: `42aeb0a5ac83cbdf4410d0e2cdf81ec06432f112`
- Original project: [IST-DASLab/Marlin](https://github.com/IST-DASLab/marlin)
- License: Apache-2.0; see `LICENSE`

The source distribution did not record an upstream Git commit in the original
import. The immutable Strata import commit above is therefore the reproducible
baseline for this copy; do not invent a more precise provenance claim.

## Local boundary and modifications

Strata does not link Torch, vLLM, or any framework runtime. The local
`core/scalar_type.hpp` is a compile-time type-identity shim used by the one
BF16/MXFP4 specialization. `marlin_template.h` differs from the import baseline
only for Strata's accepted FP32 result-publication path, introduced in commits
`5f9644e` and `f0edad9`. All other imported implementation headers remain
byte-identical to the import commit.

The adapter defines `STRATA_MARLIN_FP32_OUTPUT`; production code must use that
adapter rather than depending on the vendored directory's layout or symbols.
