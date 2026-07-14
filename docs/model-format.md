# Model format v1

Strata uses an immutable, self-describing model format. Runtime interpretation
does not depend on framework-specific metadata or tensor-name guessing.

## Header

`include/strata/model_format.h` defines the fixed little-endian header:

- eight-byte `STRATA01` magic;
- format version;
- architecture ID;
- minimum logical weight precision;
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

## QuantTrio compressed-tensors layouts

The GLM-5.2 target is already quantized. Strata preserves three native
`compressed-tensors` encodings rather than requantizing them:

- `CT_INT4_SYM_G128`: offset-packed signed four-bit routed-expert weights,
  symmetric BF16
  scale per 128 logical input elements, W4A16 with FP32 accumulation;
- `CT_INT8_SYM_G128`: signed eight-bit ordinary linear weights, symmetric BF16
  scale per 128 logical input elements, W8A16 with FP32 accumulation;
- `CT_INT8_SYM_CHANNEL`: signed eight-bit MTP weights with one symmetric BF16
  scale per output channel, W8A16 with FP32 accumulation.

Each quantized module has an I32 `weight_packed` tensor, a BF16 `weight_scale`
tensor, and an I64 two-element `weight_shape` tensor. INT4 packs eight logical
values per I32; INT8 packs four. Stored integer lanes are offset binary: decode
as `raw - 2^(bits-1)`, with the first logical value in the least-significant
bits. This is distinct from Strata's standalone two's-complement Q4 oracle.

Tensor precision is declared per directory record. Routers, normalization,
embeddings, output heads, attention weights, or measured outlier tensors may be
BF16/FP16. Four bits is the minimum permitted precision, not a requirement that
every tensor use INT4.

## Native source formats

The model format and kernel ABI must distinguish integer Q4 from floating-point
FP4. They are both four-bit representations but have different value tables,
scales, and kernels.

The required source layouts are:

- `CT_INT4_SYM_G128`, `CT_INT8_SYM_G128`, and `CT_INT8_SYM_CHANNEL` for the
  pinned QuantTrio GLM-5.2 checkpoint;
- `FP8_E4M3_B128X128`: DeepSeek supporting matrices with declared 128×128 block
  scales;
- `FP4_E2M1_K32`: DeepSeek-V4 expert values packed two per byte, using the
  E2M1 finite value table and one UE8M0 scale per 32 logical K elements;
- BF16 and FP32 tensors for normalization, routers, compression, confidence,
  and other declared sensitive roles.

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

QuantTrio/GLM-5.2-Int4-Int8Mix and DeepSeek-V4-Flash-DSpark use this path so
Strata does not create redundant 405 GB or 167 GB rewritten copies.

## Packs

Large payload files are organized for parallel direct reads without requiring
one file per expert. Expert groups derived from routing-affinity traces may occupy
nearby extents, but each expert remains independently addressable to avoid
read-amplifying false prefetches.

All runtime files are read-only. Route history, cache state, and machine-specific
placement do not modify the model pack.

## Import and optional conversion

The offline C++ `strata-pack` tool will ingest Safetensors incrementally, validate
architecture semantics, preserve validated four-bit-or-higher source layouts
where possible, generate hashes, and write to a temporary path before atomic
rename. It must never require loading the whole checkpoint into RAM.

Conversion progress is extent-addressed and hash-verified. On restart, the tool
reuses only complete verified extents; it never exposes a partially converted
pack under its final name. Quantization quality is evaluated separately from
container integrity against immutable source-precision references.

For the pinned GLM-5.2 target, the importer covers the complete source index
before bulk transfer and emits only a verified sidecar. It retains routed expert
INT4, ordinary-linear INT8, channelwise MTP INT8, and sensitive BF16/FP32 extents
in the original read-only shards. Conversion to another four-bit layout is not
part of target bring-up because it would change declared precision semantics and
require a separate quality experiment.
