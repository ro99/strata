#pragma once

#include "strata/deepseek_kv_cache.hpp"
#include "strata/result.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace strata {

// DeepSeek V4 v2.3.8 production SM86 cache contract. These tensors are
// block-major and deliberately distinct from Strata's row-major host cache.
// The allocator block covers 256 source tokens; compressed physical pages
// contain 256 / compress_ratio rows while retaining that allocator identity.
inline constexpr std::uint32_t kDsv4PhysicalKvBlockRows = 256U;

enum class Dsv4PhysicalKvFormat : std::uint8_t {
    Fp8E4m3Group64Bf16RopeUe8m0,
    Fp8E4m3PerTensorF32Scale,
};

struct Dsv4PhysicalKvLayout {
    Dsv4PhysicalKvFormat format{
        Dsv4PhysicalKvFormat::Fp8E4m3Group64Bf16RopeUe8m0};
    std::uint32_t semantic_width{};
    std::uint32_t block_rows{};
    std::uint32_t token_data_bytes{};
    std::uint32_t token_scale_bytes{};
    std::uint64_t block_bytes{};
};

struct Dsv4PhysicalKvAdmission {
    std::uint64_t sliding_bytes{};
    std::uint64_t compressed_bytes{};
    std::uint64_t index_bytes{};
    std::uint64_t payload_bytes{};
    std::uint64_t compressor_state_bytes{};
    std::uint64_t index_state_bytes{};
    std::uint64_t total_bytes{};
    std::uint64_t allocated_blocks{};
};

[[nodiscard]] ParseResult<Dsv4PhysicalKvLayout>
dsv4_physical_kv_layout(Dsv4KvBlockKind kind,
                         std::uint32_t block_rows =
                             kDsv4PhysicalKvBlockRows);

[[nodiscard]] ValidationResult dsv4_physical_decode_kv_row(
    Dsv4KvBlockKind kind, std::span<const std::byte> block,
    std::uint32_t row, std::span<float> output);

// Quantizes one BF16-boundary row with the accepted Ampere half-up E4M3
// conversion and writes it into the block-major data and scale planes.
[[nodiscard]] ValidationResult dsv4_physical_encode_kv_row(
    Dsv4KvBlockKind kind, std::span<const float> values,
    std::uint32_t row, std::span<std::byte> block);

// Default reference Lightning Index queries are unscaled E4M3 values. This
// applies the same Ampere half-up conversion used by the installed path and
// replaces each input with its decoded FP32 value.
[[nodiscard]] ValidationResult dsv4_physical_quantize_query_e4m3_f32(
    std::span<float> values);

[[nodiscard]] ParseResult<Dsv4PhysicalKvAdmission>
dsv4_physical_kv_admission(std::uint64_t maximum_context_tokens);

}  // namespace strata
