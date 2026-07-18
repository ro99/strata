# CUDA backend status

The optional native CUDA backend is implemented in `backend.cu`. CMake builds it
when a CUDA compiler is available and otherwise builds
`src/cuda_backend_stub.cpp`. The stub is an explicit unavailable-backend path:
it reports errors for CUDA operations and never pretends that GPU work succeeded.

The native backend currently provides persistent weight arenas and workspaces,
plain BF16, QuantTrio INT4/INT8, native DeepSeek FP4/FP8 matmuls, grouped
projections, and asynchronous DeepSeek expert execution. Its operation fixtures
live in `tests/test_cuda_backend.cpp`.

CUDA remains optional and may not introduce silent CPU fallback. Full-model
target-oracle promotion is a separate correctness gate; it must not be confused
with whether CUDA kernels exist.
