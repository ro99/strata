# Model format v1

Strata uses an immutable, self-describing model format. Runtime interpretation
does not depend on framework-specific metadata or tensor-name guessing.

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

## Q4_K64 design target

The first production four-bit layout will be `Q4_K64`:

- groups run along the input dimension and never cross an output row;
- each full group contains 64 signed two's-complement values in `[-8, 7]`;
- each group has one FP16 scale and no zero point;
- low and high nibble ordering is fixed by the format, not the host ABI;
- incomplete tail groups are zero-padded and carry their logical length through
  the tensor shape;
- the reference operation is W4A16 with FP32 accumulation.

This is a design target for the next implementation phase. The current per-row
int4 CPU matvec is an arithmetic oracle and does not yet implement this storage
layout.

Tensor precision is declared per directory record. Routers, normalization,
embeddings, output heads, attention weights, or measured outlier tensors may be
BF16/FP16. Four bits is the minimum permitted precision, not a requirement that
every tensor use Q4_K64.

## Packs

Large payload files are organized for parallel direct reads without requiring
one file per expert. Expert groups derived from routing-affinity traces may occupy
nearby extents, but each expert remains independently addressable to avoid
read-amplifying false prefetches.

All runtime files are read-only. Route history, cache state, and machine-specific
placement do not modify the model pack.

## Conversion

The offline C++ `strata-pack` tool will ingest Safetensors incrementally, validate
architecture semantics, preserve validated int4-or-higher source layouts where
possible, generate hashes, and write to a temporary path before atomic rename.
It must never require loading the whole checkpoint into RAM.

Conversion progress is extent-addressed and hash-verified. On restart, the tool
reuses only complete verified extents; it never exposes a partially converted
pack under its final name. Quantization quality is evaluated separately from
container integrity against immutable source-precision references.
