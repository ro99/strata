#ifndef STRATA_MODEL_FORMAT_H
#define STRATA_MODEL_FORMAT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STRATA_MODEL_MAGIC "STRATA01"
#define STRATA_MODEL_FORMAT_VERSION 1U

enum strata_architecture_id {
    STRATA_ARCH_DENSE = 1,
    STRATA_ARCH_STANDARD_MOE = 2,
    STRATA_ARCH_DEEPSEEK = 3,
};

/* All offsets are absolute and little-endian. Payloads are immutable. */
#pragma pack(push, 1)
struct strata_model_header {
    char magic[8];
    uint32_t format_version;
    uint32_t architecture;
    uint32_t quant_bits;
    uint32_t flags;
    uint64_t tensor_count;
    uint64_t directory_offset;
    uint64_t directory_bytes;
    uint64_t manifest_hash;
};
#pragma pack(pop)

#ifdef __cplusplus
}
static_assert(sizeof(strata_model_header) == 56, "model header ABI changed");
#endif

#endif
