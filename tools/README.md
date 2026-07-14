# Offline tools

Planned dependency-light C++ tools:

- `strata-pack`: incremental Safetensors to immutable Strata model packs;
- `strata-trace`: normalize sequential route captures;
- `strata-affinity`: partition expert co-routing hypergraphs across devices and
  peer workers;
- `strata-replay`: validate fixed model-forward sequences.

No model conversion or trace tool may emit weight precision below four bits.
