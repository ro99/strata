# Marlin CUDA template subset

These CUDA headers are copied from vLLM 2.3.8's
`csrc/libtorch_stable/quantization/marlin/` solely to screen and implement
Strata's standalone BF16/MXFP4 specialization. They retain the upstream
Apache-2.0 notices in the implementation files. The original implementation is
derived from [IST-DASLab/Marlin](https://github.com/IST-DASLab/marlin).

Strata does not link Torch, vLLM, or a framework runtime. The local
`core/scalar_type.hpp` is a minimal compile-time type identity shim for the one
specialization and contains no framework API.
