#include "strata/models/deepseek/dsv4_attention_kv.hpp"

#include "strata/models/deepseek/deepseek_ops.hpp"
#include "strata/models/common/model_adapter.hpp"
#include "strata/platform/numerics.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

namespace strata {
namespace {

constexpr std::uint32_t kMainTokenDataBytes = 448U + 64U * 2U;
constexpr std::uint32_t kMainTokenScaleBytes = 8U;
constexpr std::uint32_t kIndexTokenDataBytes = 128U;
constexpr std::uint32_t kIndexTokenScaleBytes = sizeof(float);

std::uint8_t encode_e4m3_half_up(float value) noexcept {
    const auto sign = value < 0.0F ? 1U : 0U;
    float magnitude = std::min(std::abs(value), 448.0F);
    if (!std::isfinite(magnitude)) magnitude = 0.0F;
    if (magnitude == 0.0F) return 0U;
    float exponent = std::floor(std::log2(magnitude));
    exponent = std::clamp(exponent, -6.0F, 8.0F);
    const auto mantissa = magnitude / std::exp2(exponent);
    int exponent_field = 0;
    int mantissa_field = 0;
    if (mantissa >= 1.0F) {
        exponent_field = static_cast<int>(exponent) + 7;
        mantissa_field = static_cast<int>(std::floor(
            (mantissa - 1.0F) * 8.0F + 0.5F));
        if (mantissa_field >= 8) {
            mantissa_field = 0;
            ++exponent_field;
        }
    } else {
        mantissa_field = static_cast<int>(std::floor(
            mantissa * 8.0F + 0.5F));
        if (mantissa_field >= 8) {
            mantissa_field = 0;
            exponent_field = 1;
        }
    }
    exponent_field = std::min(exponent_field, 15);
    return static_cast<std::uint8_t>(
        (sign << 7U) | (static_cast<unsigned>(exponent_field) << 3U) |
        static_cast<unsigned>(mantissa_field));
}

float physical_scale(std::span<const float> values) noexcept {
    float maximum = 1.0e-4F;
    for (const auto value : values) {
        maximum = std::max(maximum, std::abs(bf16_round_f32(value)));
    }
    return std::exp2(std::ceil(std::log2(maximum / 448.0F)));
}

bool add(std::uint64_t& value, std::uint64_t increment) noexcept {
    if (increment > std::numeric_limits<std::uint64_t>::max() - value) {
        return false;
    }
    value += increment;
    return true;
}

bool multiply(std::uint64_t first, std::uint64_t second,
              std::uint64_t& product) noexcept {
    if (first != 0U &&
        second > std::numeric_limits<std::uint64_t>::max() / first) {
        return false;
    }
    product = first * second;
    return true;
}

}  // namespace

ParseResult<Dsv4PhysicalKvLayout> dsv4_physical_kv_layout(
    Dsv4KvBlockKind kind, std::uint32_t block_rows) {
    ParseResult<Dsv4PhysicalKvLayout> result;
    const bool supported_rows = block_rows == kDsv4PhysicalKvBlockRows ||
        block_rows == kDsv4PhysicalKvBlockRows / 4U ||
        (kind != Dsv4KvBlockKind::LearnedIndex &&
         block_rows == kDsv4PhysicalKvBlockRows / 128U);
    if (!supported_rows) {
        result.errors.emplace_back(
            "DeepSeek V4 DeepSeek-V4 exact KV page rows are unsupported");
        return result;
    }
    result.value.block_rows = block_rows;
    if (kind == Dsv4KvBlockKind::LearnedIndex) {
        result.value.format =
            Dsv4PhysicalKvFormat::Fp8E4m3PerTensorF32Scale;
        result.value.semantic_width =
            kDeepSeekV4ExecutionContract.index_head_dim;
        result.value.token_data_bytes = kIndexTokenDataBytes;
        result.value.token_scale_bytes = kIndexTokenScaleBytes;
    } else {
        result.value.format =
            Dsv4PhysicalKvFormat::Fp8E4m3Group64Bf16RopeUe8m0;
        result.value.semantic_width = kDeepSeekV4ExecutionContract.head_dim;
        result.value.token_data_bytes = kMainTokenDataBytes;
        result.value.token_scale_bytes = kMainTokenScaleBytes;
    }
    const auto row_bytes = static_cast<std::uint64_t>(
        result.value.token_data_bytes + result.value.token_scale_bytes);
    if (!multiply(block_rows, row_bytes, result.value.block_bytes)) {
        result.errors.emplace_back(
            "DeepSeek V4 DeepSeek-V4 KV page byte count overflows");
    }
    return result;
}

ValidationResult dsv4_physical_decode_kv_row(
    Dsv4KvBlockKind kind, std::span<const std::byte> block,
    std::uint32_t row, std::span<float> output) {
    ValidationResult result;
    const auto row_bytes = kind == Dsv4KvBlockKind::LearnedIndex
        ? kIndexTokenDataBytes + kIndexTokenScaleBytes
        : kMainTokenDataBytes + kMainTokenScaleBytes;
    const auto inferred_rows = block.size() % row_bytes == 0U
        ? block.size() / row_bytes : 0U;
    const auto layout = inferred_rows <= std::numeric_limits<std::uint32_t>::max()
        ? dsv4_physical_kv_layout(
              kind, static_cast<std::uint32_t>(inferred_rows))
        : ParseResult<Dsv4PhysicalKvLayout>{};
    if (!layout.ok() || block.size() != layout.value.block_bytes ||
        row >= layout.value.block_rows ||
        output.size() != layout.value.semantic_width) {
        result.errors.emplace_back(
            "DeepSeek V4 DeepSeek-V4 KV page shape or row is invalid");
        return result;
    }

    const auto data_offset = static_cast<std::size_t>(row) *
                             layout.value.token_data_bytes;
    const auto scale_offset = static_cast<std::size_t>(layout.value.block_rows) *
                                  layout.value.token_data_bytes +
                              static_cast<std::size_t>(row) *
                                  layout.value.token_scale_bytes;
    if (kind == Dsv4KvBlockKind::LearnedIndex) {
        float scale = 0.0F;
        std::memcpy(&scale, block.data() + scale_offset, sizeof(scale));
        if (!std::isfinite(scale) || scale <= 0.0F) {
            result.errors.emplace_back(
                "DeepSeek V4 DeepSeek-V4 index FP8 scale is invalid");
            return result;
        }
        for (std::size_t column = 0U; column < output.size(); ++column) {
            const auto encoded = std::to_integer<std::uint8_t>(
                block[data_offset + column]);
            const auto value = dsv4_fp8_e4m3_f32(encoded) * scale;
            if (!std::isfinite(value)) {
                result.errors.emplace_back(
                    "DeepSeek V4 DeepSeek-V4 index FP8 value is invalid");
                return result;
            }
            output[column] = bf16_round_f32(value);
        }
        return result;
    }

    constexpr std::size_t nope =
        kDeepSeekV4ExecutionContract.head_dim -
        kDeepSeekV4ExecutionContract.rope_head_dim;
    constexpr std::size_t groups = nope / 64U;
    if (std::to_integer<std::uint8_t>(block[scale_offset + groups]) != 0U) {
        result.errors.emplace_back(
            "DeepSeek V4 DeepSeek-V4 KV scale padding byte is not zero");
        return result;
    }
    for (std::size_t group = 0U; group < groups; ++group) {
        const auto scale_code = std::to_integer<std::uint8_t>(
            block[scale_offset + group]);
        const auto scale = dsv4_fp8_e8m0_scale_f32(scale_code);
        for (std::size_t column = 0U; column < 64U; ++column) {
            const auto index = group * 64U + column;
            const auto encoded = std::to_integer<std::uint8_t>(
                block[data_offset + index]);
            const auto value = dsv4_fp8_e4m3_f32(encoded) * scale;
            if (!std::isfinite(value)) {
                result.errors.emplace_back(
                    "DeepSeek V4 DeepSeek-V4 KV FP8 value is invalid");
                return result;
            }
            output[index] = bf16_round_f32(value);
        }
    }
    const auto rope_offset = data_offset + nope;
    for (std::size_t column = 0U;
         column < kDeepSeekV4ExecutionContract.rope_head_dim; ++column) {
        std::uint16_t encoded = 0U;
        std::memcpy(&encoded,
                    block.data() + rope_offset + column * sizeof(encoded),
                    sizeof(encoded));
        output[nope + column] = std::bit_cast<float>(
            static_cast<std::uint32_t>(encoded) << 16U);
    }
    return result;
}

ValidationResult dsv4_physical_encode_kv_row(
    Dsv4KvBlockKind kind, std::span<const float> values,
    std::uint32_t row, std::span<std::byte> block) {
    ValidationResult result;
    const auto row_bytes = kind == Dsv4KvBlockKind::LearnedIndex
        ? kIndexTokenDataBytes + kIndexTokenScaleBytes
        : kMainTokenDataBytes + kMainTokenScaleBytes;
    const auto inferred_rows = block.size() % row_bytes == 0U
        ? block.size() / row_bytes : 0U;
    const auto layout = inferred_rows <= std::numeric_limits<std::uint32_t>::max()
        ? dsv4_physical_kv_layout(
              kind, static_cast<std::uint32_t>(inferred_rows))
        : ParseResult<Dsv4PhysicalKvLayout>{};
    if (!layout.ok() || block.size() != layout.value.block_bytes ||
        row >= layout.value.block_rows ||
        values.size() != layout.value.semantic_width ||
        std::any_of(values.begin(), values.end(),
                    [](float value) { return !std::isfinite(value); })) {
        result.errors.emplace_back(
            "DeepSeek V4 DeepSeek-V4 KV page shape, row, or values are invalid");
        return result;
    }

    const auto data_offset = static_cast<std::size_t>(row) *
                             layout.value.token_data_bytes;
    const auto scale_offset = static_cast<std::size_t>(layout.value.block_rows) *
                                  layout.value.token_data_bytes +
                              static_cast<std::size_t>(row) *
                                  layout.value.token_scale_bytes;
    if (kind == Dsv4KvBlockKind::LearnedIndex) {
        const auto scale = physical_scale(values);
        std::memcpy(block.data() + scale_offset, &scale, sizeof(scale));
        for (std::size_t column = 0U; column < values.size(); ++column) {
            const auto rounded = bf16_round_f32(values[column]);
            const auto scaled = std::clamp(
                rounded / scale, -448.0F, 448.0F);
            block[data_offset + column] = static_cast<std::byte>(
                encode_e4m3_half_up(scaled));
        }
        return result;
    }

    constexpr std::size_t nope =
        kDeepSeekV4ExecutionContract.head_dim -
        kDeepSeekV4ExecutionContract.rope_head_dim;
    constexpr std::size_t groups = nope / 64U;
    for (std::size_t group = 0U; group < groups; ++group) {
        const auto input = values.subspan(group * 64U, 64U);
        const auto scale = physical_scale(input);
        const auto exponent = static_cast<int>(std::ilogb(scale));
        if (exponent < -127 || exponent > 127) {
            result.errors.emplace_back(
                "DeepSeek V4 DeepSeek-V4 KV FP8 scale is outside E8M0 range");
            return result;
        }
        block[scale_offset + group] =
            static_cast<std::byte>(exponent + 127);
        for (std::size_t column = 0U; column < input.size(); ++column) {
            const auto rounded = bf16_round_f32(input[column]);
            const auto scaled = std::clamp(
                rounded / scale, -448.0F, 448.0F);
            block[data_offset + group * 64U + column] =
                static_cast<std::byte>(encode_e4m3_half_up(scaled));
        }
    }
    block[scale_offset + groups] = std::byte{};
    for (std::size_t column = 0U;
         column < kDeepSeekV4ExecutionContract.rope_head_dim; ++column) {
        const auto encoded = bf16_encode(values[nope + column]);
        std::memcpy(block.data() + data_offset + nope +
                        column * sizeof(encoded),
                    &encoded, sizeof(encoded));
    }
    return result;
}

ValidationResult dsv4_physical_quantize_query_e4m3_f32(
    std::span<float> values) {
    ValidationResult result;
    if (std::any_of(values.begin(), values.end(),
                    [](float value) { return !std::isfinite(value); })) {
        result.errors.emplace_back(
            "DeepSeek V4 DeepSeek-V4 index query contains a non-finite value");
        return result;
    }
    for (auto& value : values) {
        value = dsv4_fp8_e4m3_f32(encode_e4m3_half_up(value));
    }
    return result;
}

ParseResult<Dsv4PhysicalKvAdmission> dsv4_physical_kv_admission(
    std::uint64_t maximum_context_tokens) {
    ParseResult<Dsv4PhysicalKvAdmission> result;
    if (maximum_context_tokens == 0U ||
        maximum_context_tokens >
            kDeepSeekV4ExecutionContract.maximum_context_tokens) {
        result.errors.emplace_back(
            "DeepSeek V4 DeepSeek-V4 KV context is outside the target contract");
        return result;
    }
    const auto main = dsv4_physical_kv_layout(Dsv4KvBlockKind::Sliding);
    if (!main.ok()) {
        result.errors.emplace_back(
            "DeepSeek V4 DeepSeek-V4 KV layouts are unavailable");
        return result;
    }
    const auto blocks_for = [](std::uint64_t rows,
                               std::uint32_t block_rows) {
        return (rows + block_rows - 1U) / block_rows;
    };
    const auto sliding_rows = std::min<std::uint64_t>(
        maximum_context_tokens,
        kDeepSeekV4ExecutionContract.sliding_window +
            kDsv4PhysicalKvBlockRows - 1U);
    const auto sliding_blocks = blocks_for(
        sliding_rows, kDsv4PhysicalKvBlockRows);
    std::uint64_t layer_sliding = 0U;
    if (!multiply(sliding_blocks, main.value.block_bytes, layer_sliding) ||
        !multiply(layer_sliding, kDeepSeekV4ExecutionContract.layer_count,
                  result.value.sliding_bytes) ||
        !multiply(sliding_blocks, kDeepSeekV4ExecutionContract.layer_count,
                  result.value.allocated_blocks)) {
        result.errors.emplace_back(
            "DeepSeek V4 DeepSeek-V4 sliding KV admission overflows");
        return result;
    }

    for (std::uint32_t layer = 0U;
         layer < kDeepSeekV4ExecutionContract.layer_count; ++layer) {
        const auto ratio =
            kDeepSeekV4ExecutionContract.compression_ratios[layer];
        if (ratio == 0U) continue;
        const auto compressed_rows =
            (maximum_context_tokens + ratio - 1U) / ratio;
        const auto kind = ratio == 4U ? Dsv4KvBlockKind::Csa
                                     : Dsv4KvBlockKind::Hca;
        const auto physical_rows = kDsv4PhysicalKvBlockRows / ratio;
        const auto compressed = dsv4_physical_kv_layout(
            kind, physical_rows);
        if (!compressed.ok()) {
            result.errors.emplace_back(
                "DeepSeek V4 DeepSeek-V4 compressed KV layout is unavailable");
            return result;
        }
        const auto blocks = blocks_for(compressed_rows, physical_rows);
        std::uint64_t bytes = 0U;
        if (!multiply(blocks, compressed.value.block_bytes, bytes) ||
            !add(result.value.compressed_bytes, bytes) ||
            !add(result.value.allocated_blocks, blocks)) {
            result.errors.emplace_back(
                "DeepSeek V4 DeepSeek-V4 compressed KV admission overflows");
            return result;
        }
        const auto overlap = ratio == 4U ? 2U : 1U;
        std::uint64_t compressor_state = overlap;
        if (!multiply(compressor_state, ratio, compressor_state) ||
            !multiply(compressor_state, overlap, compressor_state) ||
            !multiply(compressor_state,
                      kDeepSeekV4ExecutionContract.head_dim,
                      compressor_state) ||
            !multiply(compressor_state, sizeof(float) * 2U,
                      compressor_state) ||
            !add(result.value.compressor_state_bytes, compressor_state)) {
            result.errors.emplace_back(
                "DeepSeek V4 DeepSeek-V4 compressor-state admission overflows");
            return result;
        }
        if (ratio == 4U) {
            const auto index = dsv4_physical_kv_layout(
                Dsv4KvBlockKind::LearnedIndex, physical_rows);
            if (!index.ok()) {
                result.errors.emplace_back(
                    "DeepSeek V4 DeepSeek-V4 index KV layout is unavailable");
                return result;
            }
            if (!multiply(blocks, index.value.block_bytes, bytes) ||
                !add(result.value.index_bytes, bytes) ||
                !add(result.value.allocated_blocks, blocks)) {
                result.errors.emplace_back(
                    "DeepSeek V4 DeepSeek-V4 index KV admission overflows");
                return result;
            }
            constexpr std::uint64_t index_state =
                2U * 4U * 2U *
                kDeepSeekV4ExecutionContract.index_head_dim *
                sizeof(float) * 2U;
            if (!add(result.value.index_state_bytes, index_state)) {
                result.errors.emplace_back(
                    "DeepSeek V4 DeepSeek-V4 index-state admission overflows");
                return result;
            }
        }
    }
    if (!add(result.value.payload_bytes, result.value.sliding_bytes) ||
        !add(result.value.payload_bytes, result.value.compressed_bytes) ||
        !add(result.value.payload_bytes, result.value.index_bytes)) {
        result.errors.emplace_back(
            "DeepSeek V4 DeepSeek-V4 KV payload admission overflows");
        return result;
    }
    result.value.total_bytes = result.value.payload_bytes;
    if (!add(result.value.total_bytes,
             result.value.compressor_state_bytes) ||
        !add(result.value.total_bytes, result.value.index_state_bytes)) {
        result.errors.emplace_back(
            "DeepSeek V4 DeepSeek-V4 total KV admission overflows");
    }
    return result;
}

}  // namespace strata
