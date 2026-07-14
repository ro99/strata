# Model format v1

Strata uses a new immutable model format. It does not depend on Colibri containers
or framework-specific runtime metadata.

## Header

`include/strata/model_format.h` defines the fixed little-endian header:

- eight-byte `STRATA01` magic;
- format version;
- architecture ID;
- weight precision;
- tensor count;
- directory offset and length;
- manifest hash.

Readers must reject any `quant_bits < 4` before mapping or allocating payloads.
There is no legacy compatibility mode for sub-4-bit data.

## Directory records

The model packer will emit fixed-size records containing:

- stable tensor and expert IDs;
- operation role (router, attention, shared expert, routed gate/up/down, etc.);
- shape, stride, quantization, and scale layout;
- payload extent and alignment;
- content hash;
- placement atomicity group.

One routed expert's gate, up, down, and scales share an atomic placement group.
Shared experts are marked resident-spine data.

## Packs

Large payload files are organized for parallel direct reads without requiring
one file per expert. Expert groups derived from routing-affinity traces may occupy
nearby extents, but each expert remains independently addressable to avoid
read-amplifying false prefetches.

All runtime files are read-only. Route history, cache state, and machine-specific
placement do not modify the model pack.

## Conversion

The offline C++ `strata-pack` tool will ingest Safetensors incrementally, validate
architecture semantics, preserve int4-or-higher tensors, generate hashes, and
write to a temporary path before atomic rename. It must never require loading the
whole checkpoint into RAM.
