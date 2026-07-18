# Offline tools

This directory contains small source-level helpers. Runnable applications built
by CMake live under `apps/`; reproducible probes and benchmarks live under
`scripts/`. The following dependency-light offline tools remain planned and are
not current executables:

- `strata-pack`: incremental Safetensors to immutable Strata model packs;
- `strata-trace`: normalize sequential route captures;
- `strata-affinity`: partition expert co-routing hypergraphs across devices and
  peer workers;
- `strata-replay`: validate fixed model-forward sequences.

Implemented characterization tools:

- `strata-topology-probe`: CUDA/NUMA transfer, kernel, peer-staging, and
  checkpoint-range cost matrix used by GLM-5.2 throughput stage T1. Run it
  reproducibly through `scripts/run_glm52_topology_probe.sh`.

No model conversion or trace tool may emit weight precision below four bits.
