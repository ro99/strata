# CUDA backend status

The CUDA backend is intentionally not scaffolded with placeholder kernels. It
will begin only after the CPU architecture oracle and real route simulations pass
their gates.

The planned backend uses persistent allocations and queues, device-resident hidden
states, exact int4-or-higher weights, and separate demand/prefetch streams. CUDA is
optional and may not introduce silent CPU fallback.
