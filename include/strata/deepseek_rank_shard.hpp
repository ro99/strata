#pragma once

#include "strata/deepseek_manifest.hpp"
#include "strata/result.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strata {

class Dsv4CheckpointReader;

enum class Dsv4ShardOwnership : std::uint8_t {
    Replicated,
    ContiguousRows,
    StridedColumns,
};

[[nodiscard]] std::string_view to_string(Dsv4ShardOwnership ownership) noexcept;

// One source-payload extent copied into one contiguous rank-local destination
// extent. source_offset is an absolute offset in the safetensors shard, while
// relative_offset is relative to the tensor payload and is the field checked
// for block/group alignment.
struct Dsv4RankShardSlice {
    std::uint64_t source_offset{};
    std::uint64_t relative_offset{};
    std::uint64_t destination_offset{};
    std::uint64_t bytes{};
};

struct Dsv4RankShardDescriptor {
    std::string base_name;
    std::string weight_name;
    std::string scale_name;
    std::string weight_shard;
    std::string scale_shard;
    Dsv4ShardOwnership ownership{Dsv4ShardOwnership::Replicated};
    std::uint32_t rank{};
    std::uint32_t world_size{1U};
    // -1 for replicated, 0 for contiguous rows, and 1 for strided columns.
    std::int32_t shard_axis{-1};

    Dsv4TensorEncoding encoding{Dsv4TensorEncoding::Plain};
    SafetensorsDtype weight_dtype{SafetensorsDtype::Other};
    SafetensorsDtype scale_dtype{SafetensorsDtype::Other};

    // Shapes are in logical checkpoint order [out, in]. packed_shape is the
    // physical source shape; for native FP4 its second dimension is half the
    // logical width because two E2M1 values share one byte.
    std::vector<std::uint64_t> logical_shape;
    std::vector<std::uint64_t> packed_shape;
    std::vector<std::uint64_t> scale_shape;
    std::vector<std::uint64_t> local_logical_shape;
    std::vector<std::uint64_t> local_packed_shape;
    std::vector<std::uint64_t> local_scale_shape;

    std::uint64_t weight_source_offset{};
    std::uint64_t weight_source_bytes{};
    std::uint64_t scale_source_offset{};
    std::uint64_t scale_source_bytes{};
    std::uint64_t local_weight_bytes{};
    std::uint64_t local_scale_bytes{};

    // Logical elements along the sharded axis that must stay together. This
    // is 128 for FP8 blocks, 32 for FP4 groups, and 1 for plain tensors.
    std::uint64_t logical_shard_alignment{1U};
    // Encoded-byte alignment of every source slice relative to its payload.
    std::uint64_t weight_slice_alignment_bytes{1U};
    std::uint64_t scale_slice_alignment_bytes{1U};

    std::vector<Dsv4RankShardSlice> weight_slices;
    std::vector<Dsv4RankShardSlice> scale_slices;
};

struct Dsv4RankShardReadStats {
    std::uint64_t calls{};
    std::uint64_t bytes{};
};

struct Dsv4RankShardPayload {
    std::vector<std::byte> weight;
    std::vector<std::byte> scale;
    Dsv4RankShardReadStats stats;
};

// Builds and validates a rank-local descriptor from the actual manifest. A
// quantized weight and its scale are one contract: either both are described
// consistently or descriptor construction fails.
[[nodiscard]] ParseResult<Dsv4RankShardDescriptor> describe_dsv4_rank_shard(
    const Dsv4CheckpointReader& checkpoint, std::string_view base_name,
    Dsv4ShardOwnership ownership, std::uint32_t rank,
    std::uint32_t world_size);

// Public validation is intentionally independent of the reader so focused
// tests can exercise malformed rank, shape, alignment, and extent contracts.
[[nodiscard]] ValidationResult validate_dsv4_rank_shard_descriptor(
    const Dsv4RankShardDescriptor& descriptor);

// Loads only the declared rank-local weight and scale bytes. The result is
// atomic at the API boundary: any weight/scale read or contract failure clears
// both payload vectors and returns errors. No complete sharded tensor is
// allocated or decoded by this function.
[[nodiscard]] ParseResult<Dsv4RankShardPayload> load_dsv4_rank_shard(
    const Dsv4CheckpointReader& checkpoint,
    const Dsv4RankShardDescriptor& descriptor);

}  // namespace strata
