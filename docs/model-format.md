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

## Native source formats

The model format and kernel ABI must distinguish integer Q4 from floating-point
FP4. They are both four-bit representations but have different value tables,
scales, and kernels.

The first required source layouts are:

- `FP8_E4M3_B128X128`: GLM-5.2 and DeepSeek supporting matrices with declared
  128×128 block scales;
- `FP4_E2M1_K32`: DeepSeek-V4 expert values packed two per byte, using the
  E2M1 finite value table and one UE8M0 scale per 32 logical K elements;
- BF16 and FP32 tensors for normalization, routers, compression, confidence,
  and other declared sensitive roles;
- Q4_K64 output extents for GLM-5.2 routed experts.

Importers preserve source precision metadata exactly. A source tensor is never
treated as Q4 merely because its physical storage uses nibbles or `int8` bytes.

## Foreign extent manifests

A Strata model need not rewrite an official checkpoint whose native layout is
already executable. A content-addressed sidecar manifest may reference immutable
extents inside Safetensors shards when:

- the repository revision, file size, source hash, tensor header, dtype, shape,
  and extent are pinned;
- every tensor has a stable Strata role and placement group;
- overlapping, missing, unknown, or mutable extents are rejected;
- runtime opens all source files read-only;
- kernels implement the declared native format without conversion.

DeepSeek-V4-Flash-DSpark uses this path so Strata does not create a redundant
167 GB rewritten copy.

## Packs

Large payload files are organized for parallel direct reads without requiring
one file per expert. Expert groups derived from routing-affinity traces may occupy
nearby extents, but each expert remains independently addressable to avoid
read-amplifying false prefetches.

All runtime files are read-only. Route history, cache state, and machine-specific
placement do not modify the model pack.

## Conversion

The offline C++ `strata-pack` tool will ingest Safetensors incrementally, validate
architecture semantics, preserve validated four-bit-or-higher source layouts
where possible, generate hashes, and write to a temporary path before atomic
rename. It must never require loading the whole checkpoint into RAM.

Conversion progress is extent-addressed and hash-verified. On restart, the tool
reuses only complete verified extents; it never exposes a partially converted
pack under its final name. Quantization quality is evaluated separately from
container integrity against immutable source-precision references.

For GLM-5.2, the conversion planner covers the complete pinned source index
before bulk transfer. Routed expert FP8 extents are decoded and requantized in
bounded tiles; DSA, router, dense-spine, and MTP tensors remain at their declared
precision initially. A verified source shard may be released only through an
explicit operator policy after all derived extents are durable.
