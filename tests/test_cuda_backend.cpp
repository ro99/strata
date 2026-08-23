#include "test.hpp"

#include "strata/attention_reference.hpp"
#include "strata/cuda_backend.hpp"
#include "strata/deepseek_kv_cache.hpp"
#include "strata/deepseek_host_expert.hpp"
#include "strata/deepseek_ops.hpp"
#include "strata/dsv4_attention_kv.hpp"
#include "strata/numerics.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_set>
#include <vector>

namespace {

void store_u32(std::byte* output, std::uint32_t value) {
    std::memcpy(output, &value, sizeof(value));
}

std::array<std::byte, 2> bf16(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto high = static_cast<std::uint16_t>(bits >> 16U);
    std::array<std::byte, 2> output{};
    std::memcpy(output.data(), &high, sizeof(high));
    return output;
}

float round_bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7F80'0000U) == 0x7F80'0000U) return value;
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return std::bit_cast<float>(bits & 0xFFFF'0000U);
}

float decode_e4m3(std::uint8_t encoded) {
    const bool negative = (encoded & 0x80U) != 0U;
    const auto exponent = static_cast<unsigned int>((encoded >> 3U) & 0x0FU);
    const auto mantissa = static_cast<unsigned int>(encoded & 0x07U);
    float value = 0.0F;
    if (exponent == 0U) {
        value = std::ldexp(static_cast<float>(mantissa) / 8.0F, -6);
    } else if (exponent == 0x0FU && mantissa == 0x07U) {
        return std::numeric_limits<float>::quiet_NaN();
    } else {
        value = std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                           static_cast<int>(exponent) - 7);
    }
    return negative ? -value : value;
}

float decode_e8m0(std::uint8_t encoded) {
    return encoded == 0xFFU
               ? std::numeric_limits<float>::quiet_NaN()
               : std::ldexp(1.0F, static_cast<int>(encoded) - 127);
}

strata::CudaWeight upload_fp4(
    strata::CudaBackend& backend, int device, std::uint64_t rows,
    std::uint64_t columns, std::uint8_t seed) {
    strata::CudaWeightDescriptor descriptor;
    descriptor.encoding = strata::CudaWeightEncoding::Fp4E2m1Group32;
    descriptor.dtype = strata::SafetensorsDtype::I8;
    descriptor.rows = rows;
    descriptor.columns = columns;
    descriptor.packed_columns = (columns + 1U) / 2U;
    descriptor.scale_columns = (columns + 31U) / 32U;
    descriptor.group_size = 32U;
    std::vector<std::byte> weights(
        static_cast<std::size_t>(rows * descriptor.packed_columns));
    for (std::uint64_t row = 0U; row < rows; ++row) {
        for (std::uint64_t packed = 0U; packed < descriptor.packed_columns;
             ++packed) {
            const auto low = static_cast<std::uint8_t>(
                (seed + row * 3U + packed * 5U) & 0x0FU);
            const auto high = static_cast<std::uint8_t>(
                (seed + row * 7U + packed * 11U + 1U) & 0x0FU);
            weights[static_cast<std::size_t>(
                row * descriptor.packed_columns + packed)] =
                static_cast<std::byte>(low | static_cast<std::uint8_t>(high << 4U));
        }
    }
    std::vector<std::byte> scales(
        static_cast<std::size_t>(rows * descriptor.scale_columns));
    for (std::size_t index = 0U; index < scales.size(); ++index) {
        scales[index] = static_cast<std::byte>(
            0x78U + static_cast<std::uint8_t>((index + seed) % 3U));
    }
    strata::CudaWeight result;
    REQUIRE(backend.upload(device, descriptor, weights, scales, result).ok());
    return result;
}

strata::CudaWeight upload_fp8(
    strata::CudaBackend& backend, int device, std::uint64_t rows,
    std::uint64_t columns, std::uint8_t seed) {
    strata::CudaWeightDescriptor descriptor;
    descriptor.encoding = strata::CudaWeightEncoding::Fp8E4m3Block128;
    descriptor.dtype = strata::SafetensorsDtype::F8E4M3;
    descriptor.rows = rows;
    descriptor.columns = columns;
    descriptor.packed_columns = columns;
    descriptor.scale_columns = (columns + 127U) / 128U;
    descriptor.group_size = 128U;
    constexpr std::array<std::uint8_t, 8> encodings{
        0x00U, 0x30U, 0xB0U, 0x38U, 0xB8U, 0x40U, 0xC0U, 0x28U};
    std::vector<std::byte> weights(static_cast<std::size_t>(rows * columns));
    for (std::size_t index = 0U; index < weights.size(); ++index) {
        weights[index] = static_cast<std::byte>(
            encodings[(index + seed) % encodings.size()]);
    }
    const auto scale_rows = (rows + 127U) / 128U;
    std::vector<std::byte> scales(
        static_cast<std::size_t>(scale_rows * descriptor.scale_columns),
        std::byte{0x78U});
    strata::CudaWeight result;
    REQUIRE(backend.upload(device, descriptor, weights, scales, result).ok());
    return result;
}

strata::CudaWeight upload_int4(
    strata::CudaBackend& backend, int device, std::uint64_t rows,
    std::uint64_t columns, std::uint8_t seed) {
    strata::CudaWeightDescriptor descriptor;
    descriptor.encoding = strata::CudaWeightEncoding::OffsetPackedInt4;
    descriptor.dtype = strata::SafetensorsDtype::I32;
    descriptor.rows = rows;
    descriptor.columns = columns;
    descriptor.packed_columns = (columns + 7U) / 8U;
    descriptor.scale_columns = (columns + 127U) / 128U;
    descriptor.group_size = 128U;
    std::vector<std::byte> packed(
        static_cast<std::size_t>(rows * descriptor.packed_columns * 4U));
    for (std::uint64_t row = 0U; row < rows; ++row) {
        for (std::uint64_t word = 0U; word < descriptor.packed_columns; ++word) {
            std::uint32_t value = 0U;
            for (std::uint32_t lane = 0U; lane < 8U; ++lane) {
                value |= static_cast<std::uint32_t>(
                    (seed + row * 3U + word * 5U + lane * 7U) & 0x0FU)
                    << (lane * 4U);
            }
            store_u32(packed.data() + static_cast<std::ptrdiff_t>(
                (row * descriptor.packed_columns + word) * 4U), value);
        }
    }
    const auto scale = bf16(0.015625F);
    std::vector<std::byte> scales(
        static_cast<std::size_t>(rows * descriptor.scale_columns * 2U));
    for (std::size_t index = 0U; index < scales.size(); index += 2U) {
        std::copy(scale.begin(), scale.end(),
                  scales.begin() + static_cast<std::ptrdiff_t>(index));
    }
    strata::CudaWeight result;
    REQUIRE(backend.upload(device, descriptor, packed, scales, result).ok());
    return result;
}

std::vector<float> reference_int4_expert(
    strata::CudaBackend& backend, const strata::CudaWeight& gate,
    const strata::CudaWeight& up, const strata::CudaWeight& down,
    std::span<const float> hidden, std::uint32_t rows,
    std::uint64_t intermediate) {
    std::vector<float> gate_output(
        static_cast<std::size_t>(rows) * intermediate);
    std::vector<float> up_output(gate_output.size());
    REQUIRE(backend.matmul(gate, hidden, rows, gate_output).ok());
    REQUIRE(backend.matmul(up, hidden, rows, up_output).ok());
    for (std::size_t index = 0U; index < gate_output.size(); ++index) {
        gate_output[index] = strata::silu_f32(gate_output[index]) *
                             up_output[index];
    }
    std::vector<float> output(hidden.size());
    REQUIRE(backend.matmul(down, gate_output, rows, output).ok());
    return output;
}

strata::CudaWeight upload_fp4_nan_scale(
    strata::CudaBackend& backend, int device, std::uint64_t rows,
    std::uint64_t columns) {
    strata::CudaWeightDescriptor descriptor;
    descriptor.encoding = strata::CudaWeightEncoding::Fp4E2m1Group32;
    descriptor.dtype = strata::SafetensorsDtype::I8;
    descriptor.rows = rows;
    descriptor.columns = columns;
    descriptor.packed_columns = (columns + 1U) / 2U;
    descriptor.scale_columns = (columns + 31U) / 32U;
    descriptor.group_size = 32U;
    std::vector<std::byte> weights(
        static_cast<std::size_t>(rows * descriptor.packed_columns),
        std::byte{0x11U});
    std::vector<std::byte> scales(
        static_cast<std::size_t>(rows * descriptor.scale_columns),
        std::byte{0xFFU});
    strata::CudaWeight result;
    REQUIRE(backend.upload(device, descriptor, weights, scales, result).ok());
    return result;
}

std::vector<float> reference_expert(
    strata::CudaBackend& backend, const strata::CudaWeight& w1,
    const strata::CudaWeight& w3, const strata::CudaWeight& w2,
    std::span<const float> hidden, std::uint64_t intermediate,
    float coefficient, bool routed) {
    std::vector<float> gate(static_cast<std::size_t>(intermediate));
    std::vector<float> up(static_cast<std::size_t>(intermediate));
    std::vector<float> activated(static_cast<std::size_t>(intermediate));
    REQUIRE(backend.matmul(w1, hidden, 1U, gate).ok());
    REQUIRE(backend.matmul(w3, hidden, 1U, up).ok());
    for (auto& value : gate) value = round_bf16(value);
    for (auto& value : up) value = round_bf16(value);
    REQUIRE(strata::dsv4_swiglu_f32(activated, gate, up, 10.0F).ok());
    for (auto& value : activated) {
        if (routed) {
            value *= coefficient;
        }
        value = round_bf16(value);
    }
    std::vector<float> output(hidden.size());
    REQUIRE(backend.matmul(w2, activated, 1U, output).ok());
    for (auto& value : output) value = round_bf16(value);
    return output;
}

struct Dsv4HostMoeCallbackFixture {
    std::vector<float> rank_partials;
    std::uint64_t calls{};
    bool accepted{true};
};

bool fill_dsv4_host_moe_partials(
    void* opaque, std::span<float> output) noexcept {
    auto& fixture = *static_cast<Dsv4HostMoeCallbackFixture*>(opaque);
    ++fixture.calls;
    if (!fixture.accepted || output.size() != fixture.rank_partials.size()) {
        return false;
    }
    std::copy(fixture.rank_partials.begin(), fixture.rank_partials.end(),
              output.begin());
    return true;
}

}  // namespace

TEST_CASE("MIX-1 FP8 fragment prepack accepts block-128 weights and refuses others") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;
    strata::CudaBackend backend;
    const int device = devices.front();
    const std::vector<int> selected{device};
    REQUIRE(backend.initialize(selected, true).ok());

    // Shapes matching the DeepSeek V4 shared expert: rows%16 and columns%32.
    const auto fp8 = upload_fp8(backend, device, 32U, 128U, 0x11U);
    REQUIRE(backend.dsv4_fp8_prepack_fragment(device, fp8).ok());
    // The permutation replaces the canonical layout in place, so the weight
    // stays valid and its declared byte count is unchanged.
    REQUIRE(fp8.valid());

    // An FP4 weight is not block-128 FP8 and must be refused rather than
    // silently permuted with the wrong element size.
    const auto fp4 = upload_fp4(backend, device, 32U, 128U, 0x22U);
    const auto refused = backend.dsv4_fp8_prepack_fragment(device, fp4);
    REQUIRE(!refused.ok());
    REQUIRE(refused.errors.front().find("Fp8E4m3Block128") != std::string::npos);

    // A row count that is not a multiple of the 16-row MMA tile must be
    // refused: the fragment map is undefined for a partial tile.
    const auto ragged = upload_fp8(backend, device, 24U, 128U, 0x33U);
    const auto rejected = backend.dsv4_fp8_prepack_fragment(device, ragged);
    REQUIRE(!rejected.ok());

    // An invalid weight must be reported, not dereferenced.
    strata::CudaWeight empty;
    REQUIRE(!backend.dsv4_fp8_prepack_fragment(device, empty).ok());
}

TEST_CASE("MIX-1 matmul route census records every dispatch and refuses unknown encodings") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;
    strata::CudaBackend backend;
    const int device = devices.front();
    const std::vector<int> selected{device};
    REQUIRE(backend.initialize(selected, true).ok());

    strata::reset_cuda_matmul_route_census();
    const auto before = strata::cuda_matmul_route_census();
    for (const auto count : before.counts) REQUIRE(count == 0U);

    // An FP4 matmul must be recorded on exactly one FP4 route, and which one
    // it is must follow the register-fed switch rather than being ambiguous.
    constexpr std::uint64_t rows = 64U, columns = 128U;
    std::vector<float> hidden(static_cast<std::size_t>(columns), 0.25F);
    std::vector<float> out(static_cast<std::size_t>(rows));
    const auto fp4 = static_cast<std::size_t>(
        strata::CudaMatmulRoute::Fp4E2m1Group32);
    const auto regfed = static_cast<std::size_t>(
        strata::CudaMatmulRoute::Fp4RegisterFed);
    const auto unsupported = static_cast<std::size_t>(
        strata::CudaMatmulRoute::Unsupported);

    strata::set_register_fed_matmul(false);
    const auto scalar_weight = upload_fp4(backend, device, rows, columns, 0x5AU);
    REQUIRE(backend.matmul(scalar_weight, hidden, 1U, out).ok());
    auto after = strata::cuda_matmul_route_census();
    REQUIRE(after.counts[fp4] == 1U);
    REQUIRE(after.counts[regfed] == 0U);

    strata::set_register_fed_matmul(true);
    const auto regfed_weight = upload_fp4(backend, device, rows, columns, 0x5AU);
    REQUIRE(backend.matmul(regfed_weight, hidden, 1U, out).ok());
    after = strata::cuda_matmul_route_census();
    REQUIRE(after.counts[fp4] == 1U);
    REQUIRE(after.counts[regfed] == 1U);
    REQUIRE(after.counts[unsupported] == 0U);

    // Every route name must be distinct and non-empty, so a census dump is
    // readable rather than ambiguous.
    for (std::size_t i = 0U;
         i < static_cast<std::size_t>(strata::CudaMatmulRoute::Count); ++i) {
        const auto* name = strata::cuda_matmul_route_name(
            static_cast<strata::CudaMatmulRoute>(i));
        REQUIRE(name != nullptr);
        REQUIRE(name[0] != '\0');
        for (std::size_t j = 0U; j < i; ++j) {
            REQUIRE(std::string(name) !=
                    std::string(strata::cuda_matmul_route_name(
                        static_cast<strata::CudaMatmulRoute>(j))));
        }
    }
}

TEST_CASE("native CUDA FlashAttention validates device support before dispatch") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;

    strata::CudaBackend backend;
    REQUIRE(!backend.validate_flash_attention_device(devices.front()).ok());
    const std::array<int, 1> selected_device{devices.front()};
    REQUIRE(backend.initialize(selected_device).ok());
    const auto supported =
        backend.validate_flash_attention_device(devices.front());
    if (!supported.ok()) {
        REQUIRE(!supported.errors.empty());
        REQUIRE(supported.errors.front().find("supports only SM86 and SM120") !=
                std::string::npos);
    }
}

TEST_CASE("native CUDA Lightning Indexer matches the scalar top-512 oracle") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;

    strata::CudaBackend backend;
    const std::array<int, 1> selected_device{devices.front()};
    REQUIRE(backend.initialize(selected_device, true).ok());
    const auto supported =
        backend.validate_lightning_index_device(devices.front());
    if (!supported.ok()) return;

    constexpr std::uint32_t heads = 64U;
    constexpr std::uint32_t head_dim = 128U;
    constexpr std::uint32_t positions = 513U;
    constexpr std::uint32_t top_k = 512U;
    std::vector<float> raw_queries(
        static_cast<std::size_t>(heads) * head_dim, 0.0F);
    for (std::uint32_t head = 0U; head < heads; ++head) {
        raw_queries[static_cast<std::size_t>(head) * head_dim +
                    (head * 17U) % head_dim] = head % 2U == 0U ? 1.0F : -1.0F;
    }
    auto simulated_queries = raw_queries;
    for (std::uint32_t head = 0U; head < heads; ++head) {
        auto query = std::span<float>(simulated_queries).subspan(
            static_cast<std::size_t>(head) * head_dim, head_dim);
        REQUIRE(strata::dsv4_hadamard_rotate_f32(query).ok());
        REQUIRE(strata::dsv4_fp4_e2m1_simulate_f32(query).ok());
    }
    std::vector<float> weights(heads);
    for (std::uint32_t head = 0U; head < heads; ++head) {
        weights[head] = static_cast<float>(static_cast<int>(head % 7U) - 3) /
                        8.0F;
    }
    constexpr std::array<float, 15> fp4_values{
        -6.0F, -4.0F, -3.0F, -2.0F, -1.5F, -1.0F, -0.5F,
         0.0F,  0.5F,  1.0F,  1.5F,  2.0F,  3.0F,  4.0F, 6.0F};
    std::vector<float> keys(static_cast<std::size_t>(positions) * head_dim);
    const auto row_bytes = strata::dsv4_kv_row_bytes(
        strata::Dsv4KvBlockKind::LearnedIndex,
        strata::Dsv4KvFormat::Fp4E2m1Group32);
    std::vector<std::byte> encoded(static_cast<std::size_t>(positions) *
                                   row_bytes);
    for (std::uint32_t row = 0U; row < positions; ++row) {
        auto key = std::span<float>(keys).subspan(
            static_cast<std::size_t>(row) * head_dim, head_dim);
        for (std::uint32_t column = 0U; column < head_dim; ++column) {
            key[column] = fp4_values[(row * 13U + column * 7U) %
                                     fp4_values.size()];
        }
        REQUIRE(strata::dsv4_encode_kv_row(
            strata::Dsv4KvBlockKind::LearnedIndex,
            strata::Dsv4KvFormat::Fp4E2m1Group32, key,
            std::span<std::byte>(encoded).subspan(
                static_cast<std::size_t>(row) * row_bytes,
                static_cast<std::size_t>(row_bytes))).ok());
    }
    std::vector<float> scores(positions);
    REQUIRE(strata::dsv4_index_scores_f32(
        scores, simulated_queries, keys, weights, heads, head_dim).ok());
    const auto expected = strata::dsv4_index_topk_f32(scores, top_k);
    REQUIRE(expected.ok());

    const std::array segments{strata::CudaLightningIndexSegment{
        nullptr, encoded, 0U, positions}};
    strata::CudaLightningIndexRequest request;
    request.queries = raw_queries;
    request.weights = weights;
    request.segments = segments;
    request.heads = heads;
    request.head_dim = head_dim;
    request.top_k = top_k;
    std::vector<std::uint32_t> actual(top_k);
    REQUIRE(backend.lightning_index(
        devices.front(), request, actual).ok());
    REQUIRE(actual == expected.positions);

    strata::CudaBuffer resident_keys;
    REQUIRE(backend.upload_buffer(
        devices.front(), encoded, resident_keys).ok());
    const std::array resident_segments{strata::CudaLightningIndexSegment{
        &resident_keys, {}, 0U, positions}};
    request.segments = resident_segments;
    std::fill(actual.begin(), actual.end(), 0U);
    REQUIRE(backend.lightning_index(
        devices.front(), request, actual).ok());
    REQUIRE(actual == expected.positions);

    constexpr std::uint32_t maximum_candidates = 1'048'576U / 4U;
    std::vector<std::byte> maximum_history(
        static_cast<std::size_t>(maximum_candidates) * row_bytes);
    const std::array maximum_segments{strata::CudaLightningIndexSegment{
        nullptr, maximum_history, 0U, maximum_candidates}};
    request.segments = maximum_segments;
    std::fill(actual.begin(), actual.end(), std::numeric_limits<std::uint32_t>::max());
    REQUIRE(backend.lightning_index(
        devices.front(), request, actual).ok());
    for (std::uint32_t index = 0U; index < top_k; ++index) {
        REQUIRE(actual[index] == index);
    }

    request.segments = {};
    std::span<std::uint32_t> empty;
    REQUIRE(backend.lightning_index(devices.front(), request, empty).ok());
    const auto stats = backend.stats();
    REQUIRE(stats.lightning_index_calls == 3U);
    REQUIRE(stats.lightning_index_kernel_launches == 12U);
    REQUIRE(stats.lightning_index_candidates ==
            positions * 2U + maximum_candidates);
    REQUIRE(stats.lightning_index_selected == top_k * 3U);
    REQUIRE(stats.lightning_index_d2h_bytes ==
            3U * (top_k * sizeof(std::uint32_t) + sizeof(unsigned int)));
    REQUIRE(stats.lightning_index_useful_selection_bytes ==
            3U * top_k * row_bytes);
}

TEST_CASE("physical Lightning Indexer matches the scalar oracle at the 1M candidate count") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;

    strata::CudaBackend backend;
    const std::array<int, 1> selected_device{devices.front()};
    REQUIRE(backend.initialize(selected_device, true).ok());
    if (!backend.validate_lightning_index_device(devices.front()).ok()) return;

    constexpr std::uint32_t heads = 64U;
    constexpr std::uint32_t head_dim = 128U;
    constexpr std::uint32_t top_k = 512U;
    constexpr std::uint32_t block_rows = strata::kDsv4PhysicalKvBlockRows;

    const auto layout = strata::dsv4_physical_kv_layout(
        strata::Dsv4KvBlockKind::LearnedIndex, block_rows);
    REQUIRE(layout.ok());
    REQUIRE(layout.value.semantic_width == head_dim);

    // Queries cross the same E4M3 round trip the runtime applies before it
    // hands them to the backend, so the oracle and the kernel see identical
    // inputs and the comparison is exactness, not tolerance.
    std::vector<float> queries(static_cast<std::size_t>(heads) * head_dim);
    for (std::size_t index = 0U; index < queries.size(); ++index) {
        queries[index] = static_cast<float>((index * 37U) % 23U) / 8.0F - 1.5F;
    }
    REQUIRE(strata::dsv4_physical_quantize_query_e4m3_f32(queries).ok());
    std::vector<float> weights(heads);
    for (std::uint32_t head = 0U; head < heads; ++head) {
        weights[head] = static_cast<float>(static_cast<int>(head % 7U) - 3) /
                        8.0F;
    }

    // Two page shapes are exercised: a partially filled tail page proves the
    // kernel never scores uncommitted rows, and the padded candidate count
    // proves selection holds at the width a 1M context actually produces.
    const auto run = [&](std::uint32_t pages, std::uint32_t tail_rows) {
        const auto candidates = (pages - 1U) * block_rows + tail_rows;
        std::vector<std::byte> storage(
            static_cast<std::size_t>(pages) * layout.value.block_bytes);
        std::vector<float> keys(
            static_cast<std::size_t>(candidates) * head_dim);
        for (std::uint32_t row = 0U; row < candidates; ++row) {
            auto key = std::span<float>(keys).subspan(
                static_cast<std::size_t>(row) * head_dim, head_dim);
            for (std::uint32_t column = 0U; column < head_dim; ++column) {
                key[column] = static_cast<float>(
                    static_cast<int>((row * 31U + column * 11U) % 17U) - 8) /
                    4.0F;
            }
            auto page = std::span<std::byte>(storage).subspan(
                static_cast<std::size_t>(row / block_rows) *
                    layout.value.block_bytes,
                static_cast<std::size_t>(layout.value.block_bytes));
            REQUIRE(strata::dsv4_physical_encode_kv_row(
                strata::Dsv4KvBlockKind::LearnedIndex, key,
                row % block_rows, page).ok());
        }
        // The oracle reads the encoded rows back rather than the pre-encoding
        // floats, so quantization error cannot hide inside the comparison.
        std::vector<float> decoded(keys.size());
        for (std::uint32_t row = 0U; row < candidates; ++row) {
            auto page = std::span<const std::byte>(storage).subspan(
                static_cast<std::size_t>(row / block_rows) *
                    layout.value.block_bytes,
                static_cast<std::size_t>(layout.value.block_bytes));
            REQUIRE(strata::dsv4_physical_decode_kv_row(
                strata::Dsv4KvBlockKind::LearnedIndex, page,
                row % block_rows,
                std::span<float>(decoded).subspan(
                    static_cast<std::size_t>(row) * head_dim,
                    head_dim)).ok());
        }
        std::vector<float> scores(candidates);
        REQUIRE(strata::dsv4_index_scores_f32(
            scores, queries, decoded, weights, heads, head_dim).ok());
        const auto expected = strata::dsv4_index_topk_f32(scores, top_k);
        REQUIRE(expected.ok());

        strata::CudaBuffer resident;
        REQUIRE(backend.upload_buffer(devices.front(), storage, resident).ok());
        std::vector<strata::CudaDsv4PhysicalIndexPage> page_descriptors;
        for (std::uint32_t page = 0U; page < pages; ++page) {
            page_descriptors.push_back(strata::CudaDsv4PhysicalIndexPage{
                &resident,
                static_cast<std::uint64_t>(page) * layout.value.block_bytes,
                block_rows,
                page + 1U == pages ? tail_rows : block_rows});
        }
        strata::CudaDsv4PhysicalIndexRequest request;
        request.queries = queries;
        request.weights = weights;
        request.pages = page_descriptors;
        request.heads = heads;
        request.head_dim = head_dim;
        request.top_k = top_k;
        std::vector<std::uint32_t> actual(top_k);
        REQUIRE(backend.dsv4_physical_lightning_index(
            devices.front(), request, actual).ok());
        REQUIRE(actual == expected.positions);
    };

    run(5U, 37U);
    // 262,144 candidates is exactly context/4 at the declared 1,048,576-token
    // maximum: the width one layer must select over on every decode step.
    run(1'024U, block_rows);
}

TEST_CASE("physical Lightning Indexer rejects malformed pages and shapes") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;

    strata::CudaBackend backend;
    const std::array<int, 1> selected_device{devices.front()};
    REQUIRE(backend.initialize(selected_device, false).ok());
    if (!backend.validate_lightning_index_device(devices.front()).ok()) return;

    constexpr std::uint32_t heads = 64U;
    constexpr std::uint32_t head_dim = 128U;
    constexpr std::uint32_t block_rows = strata::kDsv4PhysicalKvBlockRows;
    const auto layout = strata::dsv4_physical_kv_layout(
        strata::Dsv4KvBlockKind::LearnedIndex, block_rows);
    REQUIRE(layout.ok());
    std::vector<std::byte> storage(
        static_cast<std::size_t>(layout.value.block_bytes));
    std::vector<float> key(head_dim, 0.5F);
    for (std::uint32_t row = 0U; row < block_rows; ++row) {
        REQUIRE(strata::dsv4_physical_encode_kv_row(
            strata::Dsv4KvBlockKind::LearnedIndex, key, row, storage).ok());
    }
    strata::CudaBuffer resident;
    REQUIRE(backend.upload_buffer(devices.front(), storage, resident).ok());

    std::vector<float> queries(static_cast<std::size_t>(heads) * head_dim,
                               0.25F);
    std::vector<float> weights(heads, 0.5F);
    const std::array pages{strata::CudaDsv4PhysicalIndexPage{
        &resident, 0U, block_rows, block_rows}};
    strata::CudaDsv4PhysicalIndexRequest request;
    request.queries = queries;
    request.weights = weights;
    request.pages = pages;
    request.heads = heads;
    request.head_dim = head_dim;
    request.top_k = 64U;
    std::vector<std::uint32_t> output(64U);
    REQUIRE(backend.dsv4_physical_lightning_index(
        devices.front(), request, output).ok());

    // A page claiming more committed rows than its block holds would read past
    // the payload; it must be refused rather than clamped.
    const std::array overrun{strata::CudaDsv4PhysicalIndexPage{
        &resident, 0U, block_rows, block_rows + 1U}};
    auto invalid = request;
    invalid.pages = overrun;
    REQUIRE(!backend.dsv4_physical_lightning_index(
        devices.front(), invalid, output).ok());

    // A byte offset that leaves less than one page inside the buffer likewise
    // has to fail closed.
    const std::array shifted{strata::CudaDsv4PhysicalIndexPage{
        &resident, layout.value.block_bytes / 2U, block_rows, block_rows}};
    invalid = request;
    invalid.pages = shifted;
    REQUIRE(!backend.dsv4_physical_lightning_index(
        devices.front(), invalid, output).ok());

    // top_k above the candidate count has no defined selection.
    invalid = request;
    invalid.top_k = block_rows + 1U;
    std::vector<std::uint32_t> wide(block_rows + 1U);
    REQUIRE(!backend.dsv4_physical_lightning_index(
        devices.front(), invalid, wide).ok());
}

TEST_CASE("native CUDA backend executes generic tiled online FlashAttention when available") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;

    strata::CudaBackend backend;
    const std::array<int, 1> selected_device{devices.front()};
    REQUIRE(backend.initialize(selected_device, true).ok());
    constexpr std::uint32_t heads = 4U;
    constexpr std::uint32_t dimension = 64U;
    constexpr std::uint32_t source_rows = 41U;
    std::vector<float> queries(heads * dimension);
    std::vector<float> keys(source_rows * dimension);
    for (std::size_t index = 0U; index < queries.size(); ++index) {
        queries[index] = static_cast<float>(static_cast<int>(index % 17U) - 8) / 16.0F;
    }
    for (std::size_t index = 0U; index < keys.size(); ++index) {
        keys[index] = static_cast<float>(static_cast<int>(index % 23U) - 11) / 32.0F;
    }
    const std::array<std::uint32_t, 7> gathered{40U, 1U, 17U, 3U, 29U, 8U, 0U};
    const std::array<float, heads> sinks{-0.5F, 0.0F, 0.25F, 1.0F};
    const std::array<strata::FlashAttentionSegment, 1> segments{{
        {keys, {}, gathered}}};
    strata::FlashAttentionRequest request;
    request.queries = queries;
    request.segments = segments;
    request.head_sinks = sinks;
    request.query_rows = 1U;
    request.query_heads = heads;
    request.key_value_heads = 1U;
    request.query_key_dim = dimension;
    request.value_dim = dimension;
    request.scale = 1.0F / 8.0F;
    std::vector<float> expected(heads * dimension);
    std::vector<float> actual(heads * dimension, -19.0F);
    REQUIRE(strata::flash_attention_reference_f32(request, expected).ok());
    const auto status = backend.flash_attention(devices.front(), request, actual);
    if (!status.ok() && !status.errors.empty() &&
        status.errors.front().find("supports only SM86 and SM120") != std::string::npos) {
        return;
    }
    REQUIRE(status.ok());
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        REQUIRE_NEAR(actual[index], expected[index], 2.0e-5F);
    }
    const auto stats = backend.stats();
    REQUIRE(stats.flash_attention_calls == 1U);
    REQUIRE(stats.flash_attention_kernel_launches == 1U);
    REQUIRE(stats.flash_attention_h2d_bytes != 0U);
    REQUIRE(stats.flash_attention_d2h_bytes == actual.size() * sizeof(float) +
                                                    sizeof(unsigned int));
    REQUIRE(stats.flash_attention_useful_staging_bytes ==
            gathered.size() * dimension * sizeof(float));
    REQUIRE(stats.flash_attention_wasted_staging_bytes == 0U);
}

TEST_CASE("native CUDA FlashAttention matches the DeepSeek shared-KV shape on supported GPUs") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;

    constexpr std::uint32_t heads = 64U;
    constexpr std::uint32_t dimension = 512U;
    constexpr std::uint32_t window_rows = 128U;
    constexpr std::uint32_t compressed_source_rows = 521U;
    constexpr std::uint32_t compressed_selected_rows = 512U;
    std::vector<float> queries(static_cast<std::size_t>(heads) * dimension);
    std::vector<float> window(static_cast<std::size_t>(window_rows) * dimension);
    std::vector<float> compressed(
        static_cast<std::size_t>(compressed_source_rows) * dimension);
    for (std::size_t index = 0U; index < queries.size(); ++index) {
        queries[index] = static_cast<float>(static_cast<int>(index % 29U) - 14) / 64.0F;
    }
    for (std::size_t index = 0U; index < window.size(); ++index) {
        window[index] = static_cast<float>(static_cast<int>(index % 31U) - 15) / 64.0F;
    }
    for (std::size_t index = 0U; index < compressed.size(); ++index) {
        compressed[index] = static_cast<float>(static_cast<int>(index % 37U) - 18) / 64.0F;
    }
    std::vector<std::uint32_t> selected(compressed_selected_rows);
    for (std::uint32_t index = 0U; index < compressed_selected_rows; ++index) {
        selected[index] = (index * 73U + 11U) % compressed_source_rows;
    }
    std::vector<float> sinks(heads);
    for (std::uint32_t head = 0U; head < heads; ++head) {
        sinks[head] = static_cast<float>(static_cast<int>(head % 9U) - 4) * 0.125F;
    }
    const std::array<strata::FlashAttentionSegment, 2> segments{{
        {window, {}, {}}, {compressed, {}, selected}}};
    strata::FlashAttentionRequest request;
    request.queries = queries;
    request.segments = segments;
    request.head_sinks = sinks;
    request.query_rows = 1U;
    request.query_heads = heads;
    request.key_value_heads = 1U;
    request.query_key_dim = dimension;
    request.value_dim = dimension;
    request.scale = 1.0F / std::sqrt(static_cast<float>(dimension));
    request.numerics =
        strata::FlashAttentionNumerics::f64_dot_f32_score_f32_accum;
    std::vector<float> expected(static_cast<std::size_t>(heads) * dimension);
    REQUIRE(strata::flash_attention_reference_f32(request, expected).ok());

    std::size_t supported = 0U;
    for (const int device : devices) {
        strata::CudaBackend backend;
        const std::array<int, 1> selected_device{device};
        REQUIRE(backend.initialize(selected_device, true).ok());
        std::vector<float> actual(expected.size(), -7.0F);
        const auto status = backend.flash_attention(device, request, actual);
        if (!status.ok() && !status.errors.empty() &&
            status.errors.front().find("supports only SM86 and SM120") != std::string::npos) {
            continue;
        }
        REQUIRE(status.ok());
        ++supported;
        for (std::size_t index = 0U; index < actual.size(); ++index) {
            REQUIRE_NEAR(actual[index], expected[index], 5.0e-4F);
        }
    }
    REQUIRE(supported != 0U);
}

TEST_CASE("native CUDA FlashAttention grows decode scratch geometrically") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;

    constexpr std::uint32_t heads = 4U;
    constexpr std::uint32_t dimension = 64U;
    constexpr std::uint32_t maximum_rows = 17U;
    std::vector<float> queries(static_cast<std::size_t>(heads) * dimension);
    std::vector<float> keys(static_cast<std::size_t>(maximum_rows) * dimension);
    std::vector<float> sinks(heads);
    for (std::size_t index = 0U; index < queries.size(); ++index) {
        queries[index] = static_cast<float>(static_cast<int>(index % 19U) - 9) /
                         32.0F;
    }
    for (std::size_t index = 0U; index < keys.size(); ++index) {
        keys[index] = static_cast<float>(static_cast<int>(index % 23U) - 11) /
                      32.0F;
    }
    for (std::size_t index = 0U; index < sinks.size(); ++index) {
        sinks[index] = static_cast<float>(index) * 0.0625F;
    }

    strata::CudaBackend backend;
    const std::array<int, 1> selected_device{devices.front()};
    REQUIRE(backend.initialize(selected_device, true).ok());
    std::vector<float> expected(static_cast<std::size_t>(heads) * dimension);
    std::vector<float> actual(expected.size());
    for (std::uint32_t rows = 1U; rows <= maximum_rows; ++rows) {
        const std::array<strata::FlashAttentionSegment, 1> segments{{
            {std::span<const float>(keys).first(
                 static_cast<std::size_t>(rows) * dimension),
             {}, {}}}};
        strata::FlashAttentionRequest request;
        request.queries = queries;
        request.segments = segments;
        request.head_sinks = sinks;
        request.query_rows = 1U;
        request.query_heads = heads;
        request.key_value_heads = 1U;
        request.query_key_dim = dimension;
        request.value_dim = dimension;
        request.scale = 1.0F / std::sqrt(static_cast<float>(dimension));
        request.numerics =
            strata::FlashAttentionNumerics::f64_dot_f32_score_f32_accum;
        REQUIRE(strata::flash_attention_reference_f32(request, expected).ok());
        REQUIRE(backend.flash_attention(
            devices.front(), request, actual).ok());
        for (std::size_t index = 0U; index < actual.size(); ++index) {
            REQUIRE_NEAR(actual[index], expected[index], 5.0e-4F);
        }
    }

    // Three combined initial buffers plus seven power-of-two upload/score
    // growth boundaries. Exact-sized allocation would grow twice per row.
    REQUIRE(backend.stats().workspace_allocation_calls == 10U);
}

TEST_CASE("native CUDA FlashAttention preserves an all-F32 adapter contract") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;

    strata::CudaBackend backend;
    const std::array<int, 1> selected_device{devices.front()};
    REQUIRE(backend.initialize(selected_device, true).ok());
    constexpr std::uint32_t query_rows = 2U;
    constexpr std::uint32_t heads = 2U;
    constexpr std::uint32_t query_dim = 8U;
    constexpr std::uint32_t value_dim = 4U;
    constexpr std::uint32_t key_rows = 5U;
    std::vector<float> queries(query_rows * heads * query_dim);
    std::vector<float> keys(key_rows * heads * query_dim);
    std::vector<float> values(key_rows * heads * value_dim);
    for (std::size_t index = 0U; index < queries.size(); ++index) {
        queries[index] = static_cast<float>(static_cast<int>(index % 13U) - 6) /
                         16.0F;
    }
    for (std::size_t index = 0U; index < keys.size(); ++index) {
        keys[index] = static_cast<float>(static_cast<int>(index % 17U) - 8) /
                      16.0F;
    }
    for (std::size_t index = 0U; index < values.size(); ++index) {
        values[index] = static_cast<float>(static_cast<int>(index % 11U) - 5) /
                        8.0F;
    }
    const std::array<std::uint32_t, query_rows> causal_limits{3U, 5U};
    const std::array<strata::FlashAttentionSegment, 1> segments{{
        {keys, values, {}}}};
    strata::FlashAttentionRequest request;
    request.queries = queries;
    request.segments = segments;
    request.causal_key_counts = causal_limits;
    request.query_rows = query_rows;
    request.query_heads = heads;
    request.key_value_heads = heads;
    request.query_key_dim = query_dim;
    request.value_dim = value_dim;
    request.scale = 1.0F / std::sqrt(static_cast<float>(query_dim));
    request.numerics =
        strata::FlashAttentionNumerics::f32_dot_f32_softmax_f32_accum;
    std::vector<float> expected(query_rows * heads * value_dim);
    std::vector<float> actual(expected.size());
    REQUIRE(strata::flash_attention_reference_f32(request, expected).ok());
    const auto status = backend.flash_attention(
        devices.front(), request, actual);
    if (!status.ok() && !status.errors.empty() &&
        status.errors.front().find("supports only SM86 and SM120") !=
            std::string::npos) {
        return;
    }
    REQUIRE(status.ok());
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        REQUIRE_NEAR(actual[index], expected[index], 1.0e-6F);
    }
}

TEST_CASE("native CUDA GLM absorbed attention matches expanded target-shape KV") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;

    constexpr std::uint32_t query_rows = 2U;
    constexpr std::uint32_t key_rows = 5U;
    constexpr std::uint32_t heads = 64U;
    constexpr std::uint32_t nope = 192U;
    constexpr std::uint32_t rope_dim = 64U;
    constexpr std::uint32_t value_dim = 256U;
    constexpr std::uint32_t latent_dim = 512U;
    constexpr std::uint32_t projected_dim = heads * (nope + value_dim);

    strata::CudaBackend backend;
    const std::array<int, 1> selected_device{devices.front()};
    REQUIRE(backend.initialize(selected_device, true).ok());
    auto projection = upload_int4(
        backend, devices.front(), projected_dim, latent_dim, 9U);
    std::vector<float> queries(query_rows * heads * (nope + rope_dim));
    std::vector<float> latent(key_rows * latent_dim);
    std::vector<float> rope(key_rows * rope_dim);
    for (std::size_t index = 0U; index < queries.size(); ++index) {
        queries[index] = static_cast<float>(static_cast<int>(index % 29U) - 14) /
                         128.0F;
    }
    for (std::size_t index = 0U; index < latent.size(); ++index) {
        latent[index] = static_cast<float>(static_cast<int>(index % 23U) - 11) /
                        128.0F;
    }
    for (std::size_t index = 0U; index < rope.size(); ++index) {
        rope[index] = static_cast<float>(static_cast<int>(index % 19U) - 9) /
                      128.0F;
    }

    std::vector<float> projected(key_rows * projected_dim);
    REQUIRE(backend.matmul(projection, latent, key_rows, projected).ok());
    std::vector<float> keys(key_rows * heads * (nope + rope_dim));
    std::vector<float> values(key_rows * heads * value_dim);
    for (std::uint32_t row = 0U; row < key_rows; ++row) {
        for (std::uint32_t head = 0U; head < heads; ++head) {
            const auto* source = projected.data() +
                (static_cast<std::size_t>(row) * heads + head) *
                    (nope + value_dim);
            auto* key = keys.data() +
                (static_cast<std::size_t>(row) * heads + head) *
                    (nope + rope_dim);
            auto* value = values.data() +
                (static_cast<std::size_t>(row) * heads + head) * value_dim;
            std::copy_n(source, nope, key);
            std::copy_n(rope.data() + static_cast<std::size_t>(row) * rope_dim,
                        rope_dim, key + nope);
            std::copy_n(source + nope, value_dim, value);
        }
    }
    const std::array<std::uint32_t, query_rows> causal_limits{4U, 5U};
    const std::array<strata::FlashAttentionSegment, 1> segments{{
        {keys, values, {}}}};
    strata::FlashAttentionRequest reference_request;
    reference_request.queries = queries;
    reference_request.segments = segments;
    reference_request.causal_key_counts = causal_limits;
    reference_request.query_rows = query_rows;
    reference_request.query_heads = heads;
    reference_request.key_value_heads = heads;
    reference_request.query_key_dim = nope + rope_dim;
    reference_request.value_dim = value_dim;
    reference_request.scale = 1.0F / std::sqrt(static_cast<float>(nope + rope_dim));
    reference_request.numerics =
        strata::FlashAttentionNumerics::f32_dot_f32_softmax_f32_accum;
    std::vector<float> expected(query_rows * heads * value_dim);
    REQUIRE(strata::flash_attention_reference_f32(
        reference_request, expected).ok());

    strata::CudaGlmAbsorbedAttentionRequest request;
    request.queries = queries;
    request.latent = latent;
    request.rope = rope;
    request.causal_key_counts = causal_limits;
    request.scale = reference_request.scale;
    std::vector<float> actual(expected.size());
    const auto before = backend.stats();
    REQUIRE(backend.glm_absorbed_attention(
        projection, request, actual).ok());
    const auto after = backend.stats();
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        const float tolerance = 2.0e-3F *
            std::max(1.0F, std::abs(expected[index]));
        REQUIRE(std::abs(actual[index] - expected[index]) <= tolerance);
    }
    REQUIRE(after.flash_attention_calls == before.flash_attention_calls + 1U);
    REQUIRE(after.flash_attention_kernel_launches ==
            before.flash_attention_kernel_launches + 1U);
    REQUIRE(after.flash_attention_h2d_bytes ==
            before.flash_attention_h2d_bytes + queries.size() * sizeof(float) +
                latent.size() * sizeof(float) + rope.size() * sizeof(float) +
                causal_limits.size() * sizeof(std::uint32_t));
    REQUIRE(after.flash_attention_d2h_bytes ==
            before.flash_attention_d2h_bytes + actual.size() * sizeof(float) +
                sizeof(unsigned int));
}

TEST_CASE("native CUDA exact FlashAttention batches disjoint query visibility") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;

    constexpr std::uint32_t query_rows = 3U;
    constexpr std::uint32_t heads = 4U;
    constexpr std::uint32_t dimension = 64U;
    constexpr std::uint32_t key_rows = 9U;
    std::vector<float> queries(query_rows * heads * dimension);
    std::vector<float> keys(key_rows * dimension);
    std::vector<std::uint8_t> mask(query_rows * key_rows);
    for (std::size_t index = 0U; index < queries.size(); ++index) {
        queries[index] = static_cast<float>(static_cast<int>(index % 19U) - 9) /
                         32.0F;
    }
    for (std::size_t index = 0U; index < keys.size(); ++index) {
        keys[index] = static_cast<float>(static_cast<int>(index % 23U) - 11) /
                      32.0F;
    }
    for (std::uint32_t query = 0U; query < query_rows; ++query) {
        for (std::uint32_t key = 0U; key < key_rows; ++key) {
            mask[query * key_rows + key] =
                static_cast<std::uint8_t>((key + query) % 3U != 0U);
        }
    }
    const std::array<strata::FlashAttentionSegment, 1> segments{{
        {keys, {}, {}}}};
    strata::FlashAttentionRequest request;
    request.queries = queries;
    request.segments = segments;
    request.query_key_mask = mask;
    request.query_rows = query_rows;
    request.query_heads = heads;
    request.key_value_heads = 1U;
    request.query_key_dim = dimension;
    request.value_dim = dimension;
    request.scale = 1.0F / 8.0F;
    request.numerics =
        strata::FlashAttentionNumerics::f32_dot_f32_softmax_f32_accum;
    std::vector<float> expected(queries.size());
    std::vector<float> actual(queries.size());
    REQUIRE(strata::flash_attention_reference_f32(request, expected).ok());

    strata::CudaBackend backend;
    const std::array<int, 1> selected_device{devices.front()};
    REQUIRE(backend.initialize(selected_device, true).ok());
    const auto status = backend.flash_attention(
        devices.front(), request, actual);
    if (!status.ok() && !status.errors.empty() &&
        status.errors.front().find("supports only SM86 and SM120") !=
            std::string::npos) {
        return;
    }
    REQUIRE(status.ok());
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        REQUIRE_NEAR(actual[index], expected[index], 1.0e-6F);
    }
}

TEST_CASE("native CUDA backend reuses a strict bounded weight arena when available") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;

    const int device = devices.front();
    strata::CudaBackend backend;
    const std::array<int, 1> selected{device};
    REQUIRE(backend.initialize(selected).ok());
    REQUIRE(backend.reserve_weight_arena(device, 768U).ok());
    REQUIRE(!backend.reserve_weight_arena(device, 768U).ok());

    strata::CudaWeightDescriptor descriptor;
    descriptor.encoding = strata::CudaWeightEncoding::Plain;
    descriptor.dtype = strata::SafetensorsDtype::Bf16;
    descriptor.rows = 4U;
    descriptor.columns = 8U;
    std::array<std::byte, 64> payload{};

    strata::CudaWeight first;
    strata::CudaWeight second;
    strata::CudaWeight third;
    strata::CudaWeight coalesced;
    REQUIRE(backend.upload(device, descriptor, payload, {}, first).ok());
    REQUIRE(backend.upload(device, descriptor, payload, {}, second).ok());
    REQUIRE(backend.upload(device, descriptor, payload, {}, third).ok());
    REQUIRE(!backend.upload(device, descriptor, payload, {}, coalesced).ok());
    first = {};
    second = {};
    descriptor.rows = 32U;
    std::array<std::byte, 512> large_payload{};
    REQUIRE(backend.upload(device, descriptor, large_payload, {}, coalesced).ok());

    REQUIRE(first.device_bytes() == 0U);
    REQUIRE(second.device_bytes() == 0U);
    REQUIRE(third.device_bytes() == 256U);
    REQUIRE(coalesced.device_bytes() == 512U);
    const auto stats = backend.stats();
    REQUIRE(stats.weight_allocation_calls == 1U);
    REQUIRE(stats.weight_allocation_bytes == 768U);
    REQUIRE(stats.weight_upload_bytes == 704U);
}

TEST_CASE("native CUDA backend executes offset-packed groupwise matmul when available") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;

    strata::CudaBackend backend;
    const std::array<int, 1> selected{devices.front()};
    REQUIRE(backend.initialize(selected, true).ok());

    // Row 0 = [-8, -7, -6, -5], row 1 = [4, 3, 2, 1].
    std::array<std::byte, 8> packed{};
    store_u32(packed.data(), 0x3210U);
    store_u32(packed.data() + 4U, 0x9ABCU);
    std::array<std::byte, 4> scales{};
    const auto one = bf16(1.0F);
    const auto half = bf16(0.5F);
    std::copy(one.begin(), one.end(), scales.begin());
    std::copy(half.begin(), half.end(), scales.begin() + 2);

    strata::CudaWeightDescriptor descriptor;
    descriptor.encoding = strata::CudaWeightEncoding::OffsetPackedInt4;
    descriptor.dtype = strata::SafetensorsDtype::I32;
    descriptor.rows = 2U;
    descriptor.columns = 4U;
    descriptor.packed_columns = 1U;
    descriptor.scale_columns = 1U;
    descriptor.group_size = 4U;
    strata::CudaWeight weight;
    REQUIRE(backend.upload(devices.front(), descriptor, packed, scales, weight).ok());

    const std::array<float, 4> input{1.0F, 2.0F, 3.0F, 4.0F};
    std::array<float, 2> output{};
    strata::CudaMatmulProfile profile;
    REQUIRE(backend.matmul(weight, input, 1U, output, false, &profile).ok());
    REQUIRE(output[0] == -60.0F);
    REQUIRE(output[1] == 10.0F);
    REQUIRE(profile.synchronization_nanoseconds > 0U);
    REQUIRE(profile.kernel_nanoseconds > 0U);
    const auto first_stats = backend.stats();
    REQUIRE(first_stats.matmul_calls == 1U);
    REQUIRE(first_stats.weight_upload_bytes == 12U);
    REQUIRE(first_stats.activation_h2d_bytes == 16U);
    REQUIRE(first_stats.activation_d2h_bytes == 8U);
    REQUIRE(first_stats.weight_allocation_calls == 2U);
    REQUIRE(first_stats.workspace_allocation_calls == 2U);
    REQUIRE(first_stats.synchronization_calls == 2U);
    REQUIRE(first_stats.devices.size() == 1U);
    REQUIRE(first_stats.devices[0].device == devices.front());
    REQUIRE(first_stats.devices[0].kernel_nanoseconds > 0U);
    const auto& first_device = first_stats.devices[0];
    REQUIRE(first_device.synchronization_calls ==
            first_device.weight_synchronization.calls +
                first_device.attention_synchronization.calls +
                first_device.projection_synchronization.calls +
                first_device.mhc_synchronization.calls +
                first_device.moe_synchronization.calls +
                first_device.other_synchronization.calls);
    REQUIRE(first_device.synchronization_nanoseconds ==
            first_device.weight_synchronization.nanoseconds +
                first_device.attention_synchronization.nanoseconds +
                first_device.projection_synchronization.nanoseconds +
                first_device.mhc_synchronization.nanoseconds +
                first_device.moe_synchronization.nanoseconds +
                first_device.other_synchronization.nanoseconds);
    REQUIRE(first_device.weight_synchronization.calls == 1U);
    REQUIRE(first_device.projection_synchronization.calls == 1U);

    std::array<std::byte, 8> plain{};
    const std::array<float, 4> plain_values{1.0F, 2.0F, 3.0F, 4.0F};
    for (std::size_t index = 0; index < plain_values.size(); ++index) {
        const auto encoded = bf16(plain_values[index]);
        std::copy(encoded.begin(), encoded.end(), plain.begin() +
                                                    static_cast<std::ptrdiff_t>(index * 2U));
    }
    descriptor = {};
    descriptor.encoding = strata::CudaWeightEncoding::Plain;
    descriptor.dtype = strata::SafetensorsDtype::Bf16;
    descriptor.rows = 2U;
    descriptor.columns = 2U;
    strata::CudaWeight plain_weight;
    REQUIRE(backend.upload(devices.front(), descriptor, plain, {}, plain_weight).ok());
    const std::array<float, 2> plain_input{5.0F, 6.0F};
    std::array<float, 2> plain_output{};
    REQUIRE(backend.matmul(plain_weight, plain_input, 1U, plain_output).ok());
    REQUIRE(plain_output[0] == 17.0F);
    REQUIRE(plain_output[1] == 39.0F);
    std::array<float, 2> softcapped_output{};
    REQUIRE(backend.matmul_softcap(
        plain_weight, plain_input, 30.0F, softcapped_output).ok());
    for (std::size_t index = 0U; index < softcapped_output.size(); ++index) {
        auto expected = strata::bf16_round_f32(plain_output[index] / 30.0F);
        expected = strata::bf16_round_f32(std::tanh(expected));
        expected = strata::bf16_round_f32(expected * 30.0F);
        REQUIRE(softcapped_output[index] == expected);
    }

    std::array<std::byte, 4> packed_int8{};
    store_u32(packed_int8.data(), 0x7C83'7E81U);
    descriptor = {};
    descriptor.encoding = strata::CudaWeightEncoding::OffsetPackedInt8;
    descriptor.dtype = strata::SafetensorsDtype::I32;
    descriptor.rows = 1U;
    descriptor.columns = 4U;
    descriptor.packed_columns = 1U;
    descriptor.scale_columns = 1U;
    descriptor.group_size = 0U;
    strata::CudaWeight channel_weight;
    const auto two = bf16(2.0F);
    REQUIRE(backend.upload(devices.front(), descriptor, packed_int8, two,
                           channel_weight).ok());
    const std::array<float, 4> channel_input{1.0F, 1.0F, 1.0F, 1.0F};
    std::array<float, 1> channel_output{};
    REQUIRE(backend.matmul(channel_weight, channel_input, 1U, channel_output).ok());
    REQUIRE(channel_output[0] == -4.0F);

    descriptor = {};
    descriptor.encoding = strata::CudaWeightEncoding::OffsetPackedInt8;
    descriptor.dtype = strata::SafetensorsDtype::I32;
    descriptor.rows = 16U;
    descriptor.columns = 32U;
    descriptor.packed_columns = 8U;
    descriptor.scale_columns = 1U;
    descriptor.group_size = 32U;
    std::vector<std::byte> tensorcore_weights(16U * 32U, std::byte{0x81U});
    std::vector<std::byte> tensorcore_scales(16U * 2U);
    for (std::size_t offset = 0U; offset < tensorcore_scales.size(); offset += 2U) {
        std::copy(one.begin(), one.end(), tensorcore_scales.begin() +
                  static_cast<std::ptrdiff_t>(offset));
    }
    strata::CudaWeight tensorcore_weight;
    REQUIRE(backend.upload(devices.front(), descriptor, tensorcore_weights,
                           tensorcore_scales, tensorcore_weight).ok());
    std::array<float, 32> tensorcore_input{};
    tensorcore_input.fill(1.0F);
    std::array<float, 16> tensorcore_output{};
    REQUIRE(backend.matmul(tensorcore_weight, tensorcore_input, 1U,
                           tensorcore_output).ok());
    REQUIRE(std::all_of(tensorcore_output.begin(), tensorcore_output.end(),
                        [](float value) { return value == 32.0F; }));
}

TEST_CASE("native CUDA backend keeps a Gemma 4 decode layer resident") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;
    const int device = devices.front();
    strata::CudaBackend backend;
    const std::array<int, 1> selected{device};
    REQUIRE(backend.initialize(selected).ok());

    const auto zero_weight = [&](std::uint64_t rows, std::uint64_t columns) {
        strata::CudaWeightDescriptor descriptor;
        descriptor.encoding = strata::CudaWeightEncoding::OffsetPackedInt8;
        descriptor.dtype = strata::SafetensorsDtype::I32;
        descriptor.rows = rows;
        descriptor.columns = columns;
        descriptor.packed_columns = columns / 4U;
        descriptor.scale_columns = columns / 32U;
        descriptor.group_size = 32U;
        std::vector<std::byte> packed(
            static_cast<std::size_t>(rows * descriptor.packed_columns * 4U),
            std::byte{0x80U});
        std::vector<std::byte> scales(
            static_cast<std::size_t>(rows * descriptor.scale_columns * 2U));
        const auto one = bf16(1.0F);
        for (std::size_t offset = 0U; offset < scales.size(); offset += 2U) {
            std::copy(one.begin(), one.end(), scales.begin() +
                      static_cast<std::ptrdiff_t>(offset));
        }
        strata::CudaWeight output;
        REQUIRE(backend.upload(device, descriptor, packed, scales, output).ok());
        return output;
    };
    const auto upload_f32 = [&](std::span<const float> values) {
        strata::CudaBuffer output;
        REQUIRE(backend.upload_buffer(device, std::as_bytes(values), output).ok());
        return output;
    };

    constexpr std::uint32_t hidden_columns = 32U;
    constexpr std::uint32_t head_dim = 8U;
    constexpr std::uint32_t query_columns = 32U;
    constexpr std::uint32_t kv_columns = 8U;
    constexpr std::uint32_t intermediate = 32U;
    auto query = zero_weight(query_columns, hidden_columns);
    auto key = zero_weight(kv_columns, hidden_columns);
    auto value_projection = zero_weight(kv_columns, hidden_columns);
    auto projection = zero_weight(hidden_columns, query_columns);
    auto gate = zero_weight(intermediate, hidden_columns);
    auto up = zero_weight(intermediate, hidden_columns);
    auto down = zero_weight(hidden_columns, intermediate);
    std::array<float, hidden_columns> norms{};
    norms.fill(1.0F);
    std::array<float, head_dim> head_norms{};
    head_norms.fill(1.0F);
    auto input_norm = upload_f32(norms);
    auto post_attention_norm = upload_f32(norms);
    auto pre_feedforward_norm = upload_f32(norms);
    auto post_feedforward_norm = upload_f32(norms);
    auto query_norm = upload_f32(head_norms);
    auto key_norm = upload_f32(head_norms);
    strata::CudaBuffer cache;
    constexpr std::uint32_t cache_rows = 4U;
    REQUIRE(backend.allocate_buffer(
        device, 2ULL * cache_rows * kv_columns * sizeof(std::uint16_t),
        cache).ok());
    REQUIRE(backend.upload_gemma4_kv(
        cache, {}, {}, 0U, cache_rows, kv_columns).ok());

    std::array<std::uint16_t, kv_columns> next_keys{};
    std::array<std::uint16_t, kv_columns> next_values{};
    const std::array<strata::CudaGemma4DecodeLayer, 1> layers{{{
        &query, &key, &value_projection, &projection, &gate, &up, &down,
        &input_norm, &post_attention_norm, &pre_feedforward_norm,
        &post_feedforward_norm, &query_norm, &key_norm, &cache,
        next_keys, next_values, cache_rows, 0U, 0U, 1.0F,
    }}};
    std::array<float, hidden_columns> input{};
    for (std::size_t index = 0U; index < input.size(); ++index) {
        input[index] = static_cast<float>(static_cast<int>(index) - 16) / 16.0F;
    }
    std::array<float, hidden_columns> output{};
    REQUIRE(backend.gemma4_decode_layers(
        device, layers, input, 0U, output).ok());
    for (std::size_t index = 0U; index < output.size(); ++index) {
        REQUIRE(output[index] == strata::bf16_round_f32(input[index]));
    }
    REQUIRE(std::all_of(next_keys.begin(), next_keys.end(),
                        [](auto item) { return item == 0U; }));
    REQUIRE(std::all_of(next_values.begin(), next_values.end(),
                        [](auto item) { return item == 0U; }));
}

TEST_CASE("native CUDA backend executes DeepSeek FP4 FP8 and grouped projections") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;
    strata::CudaBackend backend;
    const std::array<int, 1> selected{devices.front()};
    REQUIRE(backend.initialize(selected).ok());

    constexpr std::array<std::uint8_t, 16> fp4_source{
        0xacU, 0x54U, 0xa2U, 0x44U, 0xccU, 0x54U, 0x6cU, 0x55U,
        0x2aU, 0x2cU, 0xe0U, 0xecU, 0x2dU, 0xfdU, 0x85U, 0x42U};
    std::array<std::byte, 16> fp4{};
    for (std::size_t index = 0U; index < fp4.size(); ++index) {
        fp4[index] = static_cast<std::byte>(fp4_source[index]);
    }
    constexpr std::array<std::byte, 1> fp4_scale{std::byte{0x78U}};
    strata::CudaWeightDescriptor descriptor;
    descriptor.encoding = strata::CudaWeightEncoding::Fp4E2m1Group32;
    descriptor.dtype = strata::SafetensorsDtype::I8;
    descriptor.rows = 1U;
    descriptor.columns = 32U;
    descriptor.packed_columns = 16U;
    descriptor.scale_columns = 1U;
    descriptor.group_size = 32U;
    strata::CudaWeight fp4_weight;
    REQUIRE(backend.upload(devices.front(), descriptor, fp4, fp4_scale, fp4_weight).ok());
    std::array<float, 32> fp4_input{};
    std::fill_n(fp4_input.begin(), 8U, 1.0F);
    std::array<float, 1> fp4_output{};
    REQUIRE(backend.matmul(fp4_weight, fp4_input, 1U, fp4_output).ok());
    REQUIRE_NEAR(fp4_output[0], 0.046875F, 1.0e-6F);

    constexpr std::array<float, 16> fp4_expected{
        0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
        0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F};
    std::array<std::byte, 16U * 16U> exhaustive_fp4{};
    for (std::size_t row = 0U; row < fp4_expected.size(); ++row) {
        const auto nibble = static_cast<std::uint8_t>(row);
        const auto packed = static_cast<std::byte>(nibble | (nibble << 4U));
        std::fill_n(exhaustive_fp4.begin() +
                        static_cast<std::ptrdiff_t>(row * 16U),
                    16U, packed);
    }
    std::array<std::byte, 16> exhaustive_scales{};
    exhaustive_scales.fill(std::byte{0x7FU});
    descriptor.rows = fp4_expected.size();
    strata::CudaWeight exhaustive_fp4_weight;
    REQUIRE(backend.upload(devices.front(), descriptor, exhaustive_fp4,
                           exhaustive_scales, exhaustive_fp4_weight).ok());
    std::array<float, 32> exhaustive_input{};
    exhaustive_input.front() = 1.0F;
    std::array<float, 16> exhaustive_output{};
    REQUIRE(backend.matmul(exhaustive_fp4_weight, exhaustive_input, 1U,
                           exhaustive_output).ok());
    for (std::size_t code = 0U; code < fp4_expected.size(); ++code) {
        REQUIRE(std::bit_cast<std::uint32_t>(exhaustive_output[code]) ==
                std::bit_cast<std::uint32_t>(fp4_expected[code]));
    }

    constexpr std::array<std::uint8_t, 8> fp8_source{
        0xe0U, 0xf0U, 0x6dU, 0x6cU, 0x68U, 0x41U, 0x63U, 0xefU};
    std::array<std::byte, 128> fp8{};
    for (std::size_t index = 0U; index < fp8_source.size(); ++index) {
        fp8[index] = static_cast<std::byte>(fp8_source[index]);
    }
    constexpr std::array<std::byte, 1> fp8_scale{std::byte{0x73U}};
    descriptor = {};
    descriptor.encoding = strata::CudaWeightEncoding::Fp8E4m3Block128;
    descriptor.dtype = strata::SafetensorsDtype::F8E4M3;
    descriptor.rows = 1U;
    descriptor.columns = 128U;
    descriptor.packed_columns = 128U;
    descriptor.scale_columns = 1U;
    descriptor.group_size = 128U;
    strata::CudaWeight fp8_weight;
    REQUIRE(backend.upload(devices.front(), descriptor, fp8, fp8_scale, fp8_weight).ok());
    std::array<float, 128> fp8_input{};
    std::fill_n(fp8_input.begin(), fp8_source.size(), 1.0F);
    std::array<float, 1> fp8_output{};
    REQUIRE(backend.matmul(fp8_weight, fp8_input, 1U, fp8_output).ok());
    REQUIRE_NEAR(fp8_output[0], 0.00738525390625F, 1.0e-7F);
    std::array<float, 1> fp8_single_row_requested{};
    REQUIRE(backend.matmul(fp8_weight, fp8_input, 1U,
                           fp8_single_row_requested, false, nullptr,
                           true).ok());
    REQUIRE(std::bit_cast<std::uint32_t>(fp8_single_row_requested[0]) ==
            std::bit_cast<std::uint32_t>(fp8_output[0]));

    std::array<std::byte, 8> grouped_values{};
    const std::array<float, 4> values{1.0F, 2.0F, 3.0F, 4.0F};
    for (std::size_t index = 0U; index < values.size(); ++index) {
        const auto encoded = bf16(values[index]);
        std::copy(encoded.begin(), encoded.end(),
                  grouped_values.begin() + static_cast<std::ptrdiff_t>(index * 2U));
    }
    descriptor = {};
    descriptor.encoding = strata::CudaWeightEncoding::Plain;
    descriptor.dtype = strata::SafetensorsDtype::Bf16;
    descriptor.rows = 2U;
    descriptor.columns = 2U;
    strata::CudaWeight grouped_weight;
    REQUIRE(backend.upload(devices.front(), descriptor, grouped_values, {},
                           grouped_weight).ok());
    constexpr std::array<float, 4> grouped_input{5.0F, 6.0F, 7.0F, 8.0F};
    std::array<float, 2> grouped_output{};
    REQUIRE(backend.matmul_grouped(grouped_weight, grouped_input, 2U, 1U,
                                   grouped_output).ok());
    REQUIRE(grouped_output[0] == 17.0F);
    REQUIRE(grouped_output[1] == 53.0F);
    constexpr std::array<float, 8> grouped_rows_input{
        5.0F, 6.0F, 7.0F, 8.0F,
        1.0F, 1.0F, 2.0F, 3.0F};
    std::array<float, 4> grouped_rows_output{};
    REQUIRE(backend.matmul_grouped_rows(
        grouped_weight, grouped_rows_input, 2U, 2U, 1U,
        grouped_rows_output).ok());
    REQUIRE(grouped_rows_output[0] == 17.0F);
    REQUIRE(grouped_rows_output[1] == 53.0F);
    REQUIRE(grouped_rows_output[2] == 3.0F);
    REQUIRE(grouped_rows_output[3] == 18.0F);
}

TEST_CASE("SM86 DeepSeek FP8 page projections match at the BF16 boundary") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;
    strata::CudaBackend backend;
    const std::array<int, 1> selected{devices.front()};
    REQUIRE(backend.initialize(selected).ok());
    if (!backend.dsv4_fp8_tensor_page_supported(devices.front())) return;

    constexpr std::uint32_t rows = 677U;
    struct Shape {
        std::uint32_t outputs;
        std::uint32_t inputs;
        std::uint8_t seed;
    };
    constexpr std::array<Shape, 3> shapes{{
        {1024U, 4096U, 11U},
        {32768U, 1024U, 29U},
        {512U, 4096U, 47U},
    }};
    for (const auto& shape : shapes) {
        strata::CudaWeightDescriptor descriptor;
        descriptor.encoding = strata::CudaWeightEncoding::Fp8E4m3Block128;
        descriptor.dtype = strata::SafetensorsDtype::F8E4M3;
        descriptor.rows = shape.outputs;
        descriptor.columns = shape.inputs;
        descriptor.packed_columns = shape.inputs;
        descriptor.scale_columns = shape.inputs / 128U;
        descriptor.group_size = 128U;
        std::uint32_t random =
            0xA341'316CU ^ shape.outputs ^ (shape.inputs << 8U) ^ shape.seed;
        const auto next_random = [&random] {
            random ^= random << 13U;
            random ^= random >> 17U;
            random ^= random << 5U;
            return random;
        };
        const auto random_fp8 = [&next_random] {
            const auto sign = (next_random() & 1U) << 7U;
            const auto exponent = (4U + next_random() % 8U) << 3U;
            const auto mantissa = next_random() & 7U;
            return static_cast<std::uint8_t>(sign | exponent | mantissa);
        };
        std::vector<std::byte> weight_bytes(
            static_cast<std::size_t>(shape.outputs) * shape.inputs);
        for (auto& value : weight_bytes) {
            value = static_cast<std::byte>(random_fp8());
        }
        std::vector<std::byte> weight_scales(
            static_cast<std::size_t>(shape.outputs / 128U) *
            descriptor.scale_columns);
        for (auto& value : weight_scales) {
            value = static_cast<std::byte>(123U + next_random() % 9U);
        }
        strata::CudaWeight weight;
        REQUIRE(backend.upload(devices.front(), descriptor, weight_bytes,
                               weight_scales, weight).ok());
        std::vector<float> input(
            static_cast<std::size_t>(rows) * shape.inputs);
        for (auto& value : input) {
            value = decode_e4m3(random_fp8());
        }

        const auto output_elements =
            static_cast<std::size_t>(rows) * shape.outputs;
        std::vector<float> incumbent(
            output_elements, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> tensor(
            output_elements, std::numeric_limits<float>::quiet_NaN());
        REQUIRE(backend.matmul(weight, input, rows, incumbent, true, nullptr,
                               false).ok());
        REQUIRE(backend.matmul(weight, input, rows, tensor, true, nullptr,
                               true).ok());
        REQUIRE(std::all_of(incumbent.begin(), incumbent.end(),
                            [](float value) { return std::isfinite(value); }));
        REQUIRE(std::all_of(tensor.begin(), tensor.end(),
                            [](float value) { return std::isfinite(value); }));
        std::uint64_t path_mismatches = 0U;
        for (std::size_t index = 0U; index < output_elements; ++index) {
            if (std::bit_cast<std::uint32_t>(tensor[index]) !=
                std::bit_cast<std::uint32_t>(incumbent[index])) {
                ++path_mismatches;
            }
        }

        constexpr std::size_t oracle_samples = 4096U;
        std::vector<std::uint64_t> indices;
        indices.reserve(oracle_samples);
        std::unordered_set<std::uint64_t> seen;
        seen.reserve(oracle_samples * 2U);
        std::uint64_t oracle_random =
            0x9E37'79B9'7F4A'7C15ULL ^ shape.outputs ^
            (static_cast<std::uint64_t>(shape.inputs) << 32U);
        while (indices.size() < oracle_samples) {
            oracle_random ^= oracle_random << 13U;
            oracle_random ^= oracle_random >> 7U;
            oracle_random ^= oracle_random << 17U;
            const auto index = oracle_random % output_elements;
            if (seen.insert(index).second) indices.push_back(index);
        }
        struct BoundaryError {
            double maximum_absolute{};
            double maximum_relative{};
            long double squared_sum{};
            std::uint64_t mismatches{};
        } incumbent_error, tensor_error;
        const auto scale_columns = shape.inputs / 128U;
        for (const auto index : indices) {
            const auto row = static_cast<std::uint32_t>(index / shape.outputs);
            const auto column = static_cast<std::uint32_t>(index % shape.outputs);
            double oracle = 0.0;
            for (std::uint32_t k = 0U; k < shape.inputs; ++k) {
                const auto encoded_weight = std::to_integer<std::uint8_t>(
                    weight_bytes[static_cast<std::size_t>(column) *
                                     shape.inputs + k]);
                const auto encoded_scale = std::to_integer<std::uint8_t>(
                    weight_scales[static_cast<std::size_t>(column / 128U) *
                                      scale_columns + k / 128U]);
                oracle += static_cast<double>(
                              input[static_cast<std::size_t>(row) *
                                    shape.inputs + k]) *
                          static_cast<double>(decode_e4m3(encoded_weight)) *
                          static_cast<double>(decode_e8m0(encoded_scale));
            }
            const auto expected = round_bf16(static_cast<float>(oracle));
            const auto accumulate_error = [expected](float actual,
                                                      BoundaryError& error) {
                const double absolute = std::fabs(
                    static_cast<double>(actual) - expected);
                const double relative = absolute /
                    std::max(std::fabs(static_cast<double>(expected)), 1.0e-9);
                error.maximum_absolute = std::max(error.maximum_absolute,
                                                  absolute);
                error.maximum_relative = std::max(error.maximum_relative,
                                                  relative);
                error.squared_sum += absolute * absolute;
                if (std::bit_cast<std::uint32_t>(actual) !=
                    std::bit_cast<std::uint32_t>(expected)) {
                    ++error.mismatches;
                }
            };
            accumulate_error(incumbent[index], incumbent_error);
            accumulate_error(tensor[index], tensor_error);
        }
        const auto incumbent_rms = std::sqrt(static_cast<double>(
            incumbent_error.squared_sum / oracle_samples));
        const auto tensor_rms = std::sqrt(static_cast<double>(
            tensor_error.squared_sum / oracle_samples));
        REQUIRE(tensor_error.mismatches <= incumbent_error.mismatches);
        REQUIRE(tensor_error.maximum_absolute <=
                incumbent_error.maximum_absolute);
        REQUIRE(tensor_error.maximum_relative <=
                incumbent_error.maximum_relative);
        REQUIRE(tensor_rms <= incumbent_rms);
        // Reassociation is visible in FP32 and can occasionally select a
        // different BF16 code; the binding contract is independently no worse
        // against the FP64 oracle at the carried BF16 boundary.
        REQUIRE(path_mismatches < output_elements / 1'000U);
    }
}

TEST_CASE("native CUDA backend batches reusable target-shape GLM INT4 MoE commands") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;
    constexpr std::uint32_t rows = 2U;
    constexpr std::uint64_t hidden_columns = 6144U;
    constexpr std::uint64_t intermediate_columns = 2048U;
    const int device = devices.front();
    strata::CudaBackend backend;
    const std::array<int, 1> selected{device};
    REQUIRE(backend.initialize(selected, true).ok());

    auto gate0 = upload_int4(
        backend, device, intermediate_columns, hidden_columns, 1U);
    auto up0 = upload_int4(
        backend, device, intermediate_columns, hidden_columns, 3U);
    auto down0 = upload_int4(
        backend, device, hidden_columns, intermediate_columns, 5U);
    auto gate1 = upload_int4(
        backend, device, intermediate_columns, hidden_columns, 7U);
    auto up1 = upload_int4(
        backend, device, intermediate_columns, hidden_columns, 9U);
    auto down1 = upload_int4(
        backend, device, hidden_columns, intermediate_columns, 11U);
    auto shared_gate = upload_int4(
        backend, device, intermediate_columns, hidden_columns, 13U);
    auto shared_up = upload_int4(
        backend, device, intermediate_columns, hidden_columns, 2U);
    auto shared_down = upload_int4(
        backend, device, hidden_columns, intermediate_columns, 4U);
    std::array<float, rows * hidden_columns> hidden{};
    for (std::size_t index = 0U; index < hidden.size(); ++index) {
        hidden[index] = static_cast<float>(static_cast<int>(index % 13U) - 6) /
                        16.0F;
    }
    const auto expected0 = reference_int4_expert(
        backend, gate0, up0, down0, hidden, rows, intermediate_columns);
    const auto expected1 = reference_int4_expert(
        backend, gate1, up1, down1, hidden, rows, intermediate_columns);
    const auto expected_shared = reference_int4_expert(
        backend, shared_gate, shared_up, shared_down, hidden, rows,
        intermediate_columns);
    const std::array<strata::CudaMoeExpert, 2> routed{{
        {&gate0, &up0, &down0, 1.0F},
        {&gate1, &up1, &down1, 1.0F},
    }};
    const strata::CudaMoeExpert shared{
        &shared_gate, &shared_up, &shared_down, 1.0F};
    std::array<float, 2U * rows * hidden_columns> routed_output{};
    std::array<float, rows * hidden_columns> shared_output{};
    REQUIRE(backend.enqueue_moe(device, hidden, rows, routed, &shared).ok());
    REQUIRE(backend.collect_moe(device, routed_output, shared_output).ok());
    for (std::size_t index = 0U; index < expected0.size(); ++index) {
        REQUIRE_NEAR(routed_output[index], expected0[index], 1.0e-4F);
        REQUIRE_NEAR(routed_output[expected0.size() + index],
                     expected1[index], 1.0e-4F);
        REQUIRE_NEAR(shared_output[index], expected_shared[index], 1.0e-4F);
    }
    const auto first = backend.stats();
    REQUIRE(backend.enqueue_moe(device, hidden, rows, routed, &shared).ok());
    REQUIRE(backend.collect_moe(device, routed_output, shared_output).ok());
    const auto second = backend.stats();
    REQUIRE(second.workspace_allocation_calls == first.workspace_allocation_calls);
    REQUIRE(second.deepseek_moe_calls - first.deepseek_moe_calls == 1U);
    REQUIRE(second.deepseek_moe_kernel_launches -
                first.deepseek_moe_kernel_launches == 2U);
}

TEST_CASE("native CUDA backend enqueues exact grouped DeepSeek MoE when available") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;
    constexpr std::uint64_t hidden_columns = 32U;
    constexpr std::uint64_t intermediate_columns = 32U;
    const int device = devices.front();
    strata::CudaBackend backend;
    const std::array<int, 1> selected{device};
    REQUIRE(backend.initialize(selected, true).ok());

    auto routed0_w1 = upload_fp4(
        backend, device, intermediate_columns, hidden_columns, 1U);
    auto routed0_w3 = upload_fp4(
        backend, device, intermediate_columns, hidden_columns, 6U);
    auto routed0_w2 = upload_fp4(
        backend, device, hidden_columns, intermediate_columns, 11U);
    auto routed1_w1 = upload_fp4(
        backend, device, intermediate_columns, hidden_columns, 3U);
    auto routed1_w3 = upload_fp4(
        backend, device, intermediate_columns, hidden_columns, 9U);
    auto routed1_w2 = upload_fp4(
        backend, device, hidden_columns, intermediate_columns, 14U);
    auto shared_w1 = upload_fp8(
        backend, device, intermediate_columns, hidden_columns, 1U);
    auto shared_w3 = upload_fp8(
        backend, device, intermediate_columns, hidden_columns, 4U);
    auto shared_w2 = upload_fp8(
        backend, device, hidden_columns, intermediate_columns, 7U);

    std::array<float, hidden_columns> hidden{};
    for (std::size_t index = 0U; index < hidden.size(); ++index) {
        hidden[index] = static_cast<float>(static_cast<int>(index % 9U) - 4) *
                        0.125F;
    }
    constexpr float coefficient0 = 0.75F;
    constexpr float coefficient1 = 0.3125F;
    const auto expected0 = reference_expert(
        backend, routed0_w1, routed0_w3, routed0_w2, hidden,
        intermediate_columns, coefficient0, true);
    const auto expected1 = reference_expert(
        backend, routed1_w1, routed1_w3, routed1_w2, hidden,
        intermediate_columns, coefficient1, true);
    const auto expected_shared = reference_expert(
        backend, shared_w1, shared_w3, shared_w2, hidden,
        intermediate_columns, 1.0F, false);

    const std::array<strata::CudaDeepSeekMoeExpert, 2> routed{{
        {&routed0_w1, &routed0_w3, &routed0_w2, coefficient0},
        {&routed1_w1, &routed1_w3, &routed1_w2, coefficient1},
    }};
    const strata::CudaDeepSeekMoeExpert shared{
        &shared_w1, &shared_w3, &shared_w2, 1.0F};
    const auto before = backend.stats();
    REQUIRE(backend.enqueue_deepseek_moe(
        device, hidden, routed, &shared, 10.0F).ok());
    REQUIRE(!backend.enqueue_deepseek_moe(
        device, hidden, routed, &shared, 10.0F).ok());
    REQUIRE(!backend.synchronize(device).ok());

    std::array<float, 2U * hidden_columns> routed_output{};
    std::array<float, hidden_columns> shared_output{};
    REQUIRE(backend.collect_deepseek_moe(
        device, routed_output, shared_output).ok());
    float maximum_difference = 0.0F;
    for (std::size_t column = 0U; column < hidden_columns; ++column) {
        const float difference0 = std::abs(routed_output[column] - expected0[column]);
        const float difference1 = std::abs(
            routed_output[hidden_columns + column] - expected1[column]);
        const float shared_difference =
            std::abs(shared_output[column] - expected_shared[column]);
        maximum_difference = std::max(
            maximum_difference,
            std::max(difference0, std::max(difference1, shared_difference)));
        REQUIRE(std::bit_cast<std::uint32_t>(routed_output[column]) ==
                std::bit_cast<std::uint32_t>(expected0[column]));
        REQUIRE(std::bit_cast<std::uint32_t>(
                    routed_output[hidden_columns + column]) ==
                std::bit_cast<std::uint32_t>(expected1[column]));
        REQUIRE(std::bit_cast<std::uint32_t>(shared_output[column]) ==
                std::bit_cast<std::uint32_t>(expected_shared[column]));
        REQUIRE((std::bit_cast<std::uint32_t>(routed_output[column]) & 0xFFFFU) == 0U);
        REQUIRE((std::bit_cast<std::uint32_t>(
                     routed_output[hidden_columns + column]) & 0xFFFFU) == 0U);
        REQUIRE((std::bit_cast<std::uint32_t>(shared_output[column]) & 0xFFFFU) == 0U);
    }
    REQUIRE(maximum_difference == 0.0F);

    const auto after = backend.stats();
    REQUIRE(after.deepseek_moe_calls - before.deepseek_moe_calls == 1U);
    REQUIRE(after.deepseek_moe_kernel_launches -
                before.deepseek_moe_kernel_launches == 7U);
    REQUIRE(after.deepseek_moe_h2d_transfers -
                before.deepseek_moe_h2d_transfers == 1U);
    REQUIRE(after.deepseek_moe_d2h_transfers -
                before.deepseek_moe_d2h_transfers == 2U);
    REQUIRE(after.deepseek_moe_h2d_bytes - before.deepseek_moe_h2d_bytes ==
            hidden_columns * sizeof(float));
    REQUIRE(after.deepseek_moe_d2h_bytes - before.deepseek_moe_d2h_bytes ==
            3U * hidden_columns * sizeof(float) + sizeof(unsigned int));
    REQUIRE(after.matmul_calls - before.matmul_calls == 9U);
    REQUIRE(after.activation_h2d_bytes - before.activation_h2d_bytes ==
            hidden_columns * sizeof(float));
    REQUIRE(after.activation_d2h_bytes - before.activation_d2h_bytes ==
            3U * hidden_columns * sizeof(float));
    REQUIRE(after.workspace_allocation_calls -
                before.workspace_allocation_calls == 6U);
    REQUIRE(after.synchronization_calls - before.synchronization_calls == 1U);
    REQUIRE(after.deepseek_moe_h2d_nanoseconds >
            before.deepseek_moe_h2d_nanoseconds);
    REQUIRE(after.deepseek_moe_kernel_nanoseconds >
            before.deepseek_moe_kernel_nanoseconds);
    REQUIRE(after.deepseek_moe_d2h_nanoseconds >
            before.deepseek_moe_d2h_nanoseconds);
    REQUIRE(after.deepseek_moe_nanoseconds - before.deepseek_moe_nanoseconds ==
            (after.deepseek_moe_h2d_nanoseconds -
             before.deepseek_moe_h2d_nanoseconds) +
            (after.deepseek_moe_kernel_nanoseconds -
             before.deepseek_moe_kernel_nanoseconds) +
            (after.deepseek_moe_d2h_nanoseconds -
             before.deepseek_moe_d2h_nanoseconds));

    // The host callback boundary supplies two rank-local partials through one
    // stream callback. The GPU must preserve the retained association exactly:
    // bf16(bf16(rank0 + rank1) + bf16(shared)).
    Dsv4HostMoeCallbackFixture host_fixture;
    host_fixture.rank_partials.insert(
        host_fixture.rank_partials.end(), expected0.begin(), expected0.end());
    host_fixture.rank_partials.insert(
        host_fixture.rank_partials.end(), expected1.begin(), expected1.end());
    std::array<float, hidden_columns> host_join_output{};
    const auto before_host_join = backend.stats();
    REQUIRE(backend.enqueue_dsv4_host_moe(
        device, hidden, shared, 10.0F, fill_dsv4_host_moe_partials,
        &host_fixture).ok());
    REQUIRE(!backend.synchronize(device).ok());
    REQUIRE(backend.collect_deepseek_moe(
        device, {}, host_join_output).ok());
    REQUIRE(host_fixture.calls == 1U);
    for (std::size_t column = 0U; column < hidden_columns; ++column) {
        const auto expected = round_bf16(
            round_bf16(expected0[column] + expected1[column]) +
            round_bf16(expected_shared[column]));
        REQUIRE(std::bit_cast<std::uint32_t>(host_join_output[column]) ==
                std::bit_cast<std::uint32_t>(expected));
    }
    const auto after_host_join = backend.stats();
    REQUIRE(after_host_join.deepseek_moe_calls -
                before_host_join.deepseek_moe_calls == 1U);
    REQUIRE(after_host_join.deepseek_moe_kernel_launches -
                before_host_join.deepseek_moe_kernel_launches == 5U);
    REQUIRE(after_host_join.deepseek_moe_h2d_transfers -
                before_host_join.deepseek_moe_h2d_transfers == 2U);
    REQUIRE(after_host_join.deepseek_moe_h2d_bytes -
                before_host_join.deepseek_moe_h2d_bytes ==
            3U * hidden_columns * sizeof(float));
    REQUIRE(after_host_join.deepseek_moe_d2h_transfers -
                before_host_join.deepseek_moe_d2h_transfers == 2U);
    REQUIRE(after_host_join.deepseek_moe_d2h_bytes -
                before_host_join.deepseek_moe_d2h_bytes ==
            hidden_columns * sizeof(float) + sizeof(unsigned int));
    REQUIRE(after_host_join.matmul_calls - before_host_join.matmul_calls == 3U);

    host_fixture.accepted = false;
    host_join_output.fill(123.0F);
    REQUIRE(backend.enqueue_dsv4_host_moe(
        device, hidden, shared, 10.0F, fill_dsv4_host_moe_partials,
        &host_fixture).ok());
    REQUIRE(!backend.collect_deepseek_moe(
        device, {}, host_join_output).ok());
    REQUIRE(host_fixture.calls == 2U);
    REQUIRE(std::all_of(host_join_output.begin(), host_join_output.end(),
                        [](float value) { return value == 123.0F; }));
    host_fixture.accepted = true;
    REQUIRE(backend.enqueue_dsv4_host_moe(
        device, hidden, shared, 10.0F, fill_dsv4_host_moe_partials,
        &host_fixture).ok());
    REQUIRE(backend.collect_deepseek_moe(
        device, {}, host_join_output).ok());

    // Non-finite W1/W3 output is an explicit failure and never reaches the
    // caller through the backend-owned staging buffer.
    auto invalid_w1 = upload_fp4_nan_scale(
        backend, device, intermediate_columns, hidden_columns);
    const std::array<strata::CudaDeepSeekMoeExpert, 1> invalid_routed{{
        {&invalid_w1, &routed0_w3, &routed0_w2, coefficient0},
    }};
    std::array<float, hidden_columns> invalid_output{};
    invalid_output.fill(123.0F);
    REQUIRE(backend.enqueue_deepseek_moe(
        device, hidden, invalid_routed, nullptr, 10.0F).ok());
    REQUIRE(!backend.collect_deepseek_moe(
        device, invalid_output, {}).ok());
    REQUIRE(std::all_of(invalid_output.begin(), invalid_output.end(),
                        [](float value) { return value == 123.0F; }));

    // A failed collect must drain the command before returning so weight
    // leases and the persistent workspace can be reused safely.
    REQUIRE(backend.enqueue_deepseek_moe(
        device, hidden, routed, &shared, 10.0F).ok());
    REQUIRE(!backend.collect_deepseek_moe(device, {}, {}).ok());
    REQUIRE(backend.enqueue_deepseek_moe(
        device, hidden, routed, &shared, 10.0F).ok());
    REQUIRE(backend.collect_deepseek_moe(
        device, routed_output, shared_output).ok());
}

TEST_CASE("native CUDA DeepSeek paged attention reads persistent physical pages") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;
    const auto device = devices.front();
    strata::CudaBackend backend;
    const std::array<int, 1> selected_devices{device};
    REQUIRE(backend.initialize(selected_devices, true).ok());

    constexpr std::uint32_t page_rows = 256U;
    std::vector<std::byte> physical_page(
        static_cast<std::size_t>(page_rows) * 584U);
    // Every candidate decodes to an all-one 512-wide value. This exercises
    // the block-major FP8/BF16 materialization and the complete nonzero
    // finish path while retaining an exact analytical result: identical
    // zero scores give a weighted average of one in every output lane.
    constexpr std::size_t data_row_bytes = 576U;
    constexpr std::size_t nope_columns = 448U;
    constexpr std::size_t rope_columns = 64U;
    const auto one_bf16 = bf16(1.0F);
    for (std::uint32_t row = 0U; row < page_rows; ++row) {
        const auto data_offset = static_cast<std::size_t>(row) * data_row_bytes;
        std::fill_n(physical_page.begin() +
                        static_cast<std::ptrdiff_t>(data_offset),
                    nope_columns, std::byte{0x38U});
        for (std::size_t column = 0U; column < rope_columns; ++column) {
            std::copy(one_bf16.begin(), one_bf16.end(),
                      physical_page.begin() + static_cast<std::ptrdiff_t>(
                          data_offset + nope_columns + column * 2U));
        }
        const auto scale_offset =
            static_cast<std::size_t>(page_rows) * data_row_bytes +
            static_cast<std::size_t>(row) * 8U;
        std::fill_n(physical_page.begin() +
                        static_cast<std::ptrdiff_t>(scale_offset),
                    7U, std::byte{127U});
    }
    strata::CudaBuffer page_buffer;
    REQUIRE(backend.upload_buffer(device, physical_page, page_buffer).ok());
    const std::array<strata::CudaDsv4PhysicalPage, 1> pages{{
        {&page_buffer, page_rows},
    }};
    std::vector<strata::CudaDsv4AttentionCandidate> candidates(128U);
    for (std::uint32_t row = 0U; row < candidates.size(); ++row) {
        candidates[row] = {0U, row, true};
    }
    std::vector<float> queries(64U * 512U);
    std::array<float, 64> sinks{};
    sinks.fill(-1.0e30F);
    std::vector<float> output(queries.size());
    const auto before = backend.stats();
    for (std::size_t head_begin = 0U; head_begin < 64U; head_begin += 32U) {
        strata::CudaDsv4PagedAttentionRequest request;
        request.queries = std::span<const float>(queries).subspan(
            head_begin * 512U, 32U * 512U);
        request.head_sinks = std::span<const float>(sinks).subspan(
            head_begin, 32U);
        request.pages = pages;
        request.candidates = candidates;
        request.scale = 1.0F / std::sqrt(512.0F);
        const auto executed = backend.dsv4_paged_attention(
            device, request,
            std::span<float>(output).subspan(
                head_begin * 512U, 32U * 512U));
        if (!executed.ok() && std::any_of(
            executed.errors.begin(), executed.errors.end(),
            [](const std::string& error) {
                return error.find("SM86") != std::string::npos;
            })) {
            return;
        }
        REQUIRE(executed.ok());
    }
    REQUIRE(std::all_of(output.begin(), output.end(), [](float value) {
        return std::bit_cast<std::uint32_t>(value) ==
               std::bit_cast<std::uint32_t>(1.0F);
    }));
    const auto after = backend.stats();
    REQUIRE(after.dsv4_paged_attention_calls -
                before.dsv4_paged_attention_calls == 2U);
    REQUIRE(after.dsv4_paged_attention_kernel_launches -
                before.dsv4_paged_attention_kernel_launches == 14U);
    REQUIRE(after.dsv4_paged_attention_page_bytes -
                before.dsv4_paged_attention_page_bytes ==
            2U * physical_page.size());
    REQUIRE(after.dsv4_paged_attention_h2d_bytes >
            before.dsv4_paged_attention_h2d_bytes);
    REQUIRE(after.dsv4_paged_attention_d2h_bytes -
                before.dsv4_paged_attention_d2h_bytes ==
            output.size() * sizeof(std::uint16_t) +
                2U * sizeof(unsigned int));
}

TEST_CASE("native CUDA DeepSeek paged attention batches rows bit exactly") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;
    const auto device = devices.front();
    strata::CudaBackend backend;
    const std::array<int, 1> selected_devices{device};
    REQUIRE(backend.initialize(selected_devices, true).ok());

    constexpr std::uint32_t physical_rows = 256U;
    constexpr std::size_t data_row_bytes = 576U;
    constexpr std::size_t nope_columns = 448U;
    constexpr std::size_t rope_columns = 64U;
    std::vector<std::byte> physical_page(
        static_cast<std::size_t>(physical_rows) * 584U);
    for (std::uint32_t row = 0U; row < physical_rows; ++row) {
        const auto data_offset = static_cast<std::size_t>(row) * data_row_bytes;
        const auto e4m3 = static_cast<std::uint8_t>(
            0x30U + ((row * 3U) & 0x0FU));
        std::fill_n(physical_page.begin() +
                        static_cast<std::ptrdiff_t>(data_offset),
                    nope_columns, static_cast<std::byte>(e4m3));
        const auto rope = bf16(round_bf16(
            static_cast<float>((row % 17U) + 1U) / 32.0F));
        for (std::size_t column = 0U; column < rope_columns; ++column) {
            std::copy(rope.begin(), rope.end(),
                      physical_page.begin() + static_cast<std::ptrdiff_t>(
                          data_offset + nope_columns + column * 2U));
        }
        const auto scale_offset =
            static_cast<std::size_t>(physical_rows) * data_row_bytes +
            static_cast<std::size_t>(row) * 8U;
        std::fill_n(physical_page.begin() +
                        static_cast<std::ptrdiff_t>(scale_offset),
                    7U, std::byte{127U});
    }
    strata::CudaBuffer page_buffer;
    REQUIRE(backend.upload_buffer(device, physical_page, page_buffer).ok());
    const std::array<strata::CudaDsv4PhysicalPage, 1> pages{{
        {&page_buffer, physical_rows},
    }};
    std::array<float, 32U> sinks{};
    for (std::size_t head = 0U; head < sinks.size(); ++head) {
        sinks[head] = -8.0F + static_cast<float>(head) / 32.0F;
    }

    constexpr std::size_t row_query_elements = 32U * 512U;
    const std::array<std::uint32_t, 2U> page_rows{4U, 64U};
    const std::array<std::uint32_t, 3U> prompt_rows{8U, 52U, 144U};
    for (const auto page_size : page_rows) {
        for (const auto prompt_size : prompt_rows) {
            const auto candidate_width =
                ((prompt_size + 127U) / 128U) * 128U;
            std::vector<float> queries(
                static_cast<std::size_t>(prompt_size) * row_query_elements);
            for (std::uint32_t row = 0U; row < prompt_size; ++row) {
                for (std::size_t index = 0U; index < row_query_elements;
                     ++index) {
                    const auto signed_code = static_cast<int>(
                        (row * 11U + static_cast<std::uint32_t>(index)) % 15U) -
                        7;
                    queries[static_cast<std::size_t>(row) * row_query_elements +
                            index] = round_bf16(
                        static_cast<float>(signed_code) / 64.0F);
                }
            }
            std::vector<strata::CudaDsv4AttentionCandidate> candidates(
                static_cast<std::size_t>(prompt_size) * candidate_width);
            for (std::uint32_t row = 0U; row < prompt_size; ++row) {
                auto row_candidates = std::span(candidates).subspan(
                    static_cast<std::size_t>(row) * candidate_width,
                    candidate_width);
                for (std::uint32_t key = 0U; key <= row; ++key) {
                    row_candidates[key] = {0U, key, true};
                }
            }

            std::vector<float> reference(queries.size());
            for (std::uint32_t row = 0U; row < prompt_size; ++row) {
                strata::CudaDsv4PagedAttentionRequest request;
                request.queries = std::span<const float>(queries).subspan(
                    static_cast<std::size_t>(row) * row_query_elements,
                    row_query_elements);
                request.head_sinks = sinks;
                request.pages = pages;
                request.candidates =
                    std::span<const strata::CudaDsv4AttentionCandidate>(
                        candidates).subspan(
                            static_cast<std::size_t>(row) * candidate_width,
                            candidate_width);
                request.candidate_width = candidate_width;
                request.scale = 1.0F / std::sqrt(512.0F);
                auto executed = backend.dsv4_paged_attention(
                    device, request, std::span<float>(reference).subspan(
                        static_cast<std::size_t>(row) * row_query_elements,
                        row_query_elements));
                if (!executed.ok() && std::any_of(
                    executed.errors.begin(), executed.errors.end(),
                    [](const std::string& error) {
                        return error.find("SM86") != std::string::npos;
                    })) {
                    return;
                }
                REQUIRE(executed.ok());
            }

            std::vector<float> batched(queries.size());
            const auto forced_workspace =
                backend.dsv4_paged_attention_to_mhc_page_workspace_bytes(
                    pages, page_size, candidate_width);
            REQUIRE(forced_workspace.ok());
            const auto admitted =
                backend.dsv4_paged_attention_to_mhc_page_maximum_rows(
                    pages, prompt_size, candidate_width,
                    forced_workspace.value);
            REQUIRE(admitted.ok());
            REQUIRE(admitted.value == std::min(page_size, prompt_size));
            if (prompt_size > page_size) {
                const auto unsplit_workspace =
                    backend.dsv4_paged_attention_to_mhc_page_workspace_bytes(
                        pages, prompt_size, candidate_width);
                REQUIRE(unsplit_workspace.ok());
                REQUIRE(unsplit_workspace.value > forced_workspace.value);
            }
            for (std::uint32_t begin = 0U; begin < prompt_size;
                 begin += admitted.value) {
                const auto rows = std::min(admitted.value, prompt_size - begin);
                strata::CudaDsv4PagedAttentionRequest request;
                request.queries = std::span<const float>(queries).subspan(
                    static_cast<std::size_t>(begin) * row_query_elements,
                    static_cast<std::size_t>(rows) * row_query_elements);
                request.head_sinks = sinks;
                request.pages = pages;
                request.candidates =
                    std::span<const strata::CudaDsv4AttentionCandidate>(
                        candidates).subspan(
                            static_cast<std::size_t>(begin) * candidate_width,
                            static_cast<std::size_t>(rows) * candidate_width);
                request.rows = rows;
                request.candidate_width = candidate_width;
                request.maximum_workspace_bytes = forced_workspace.value;
                request.scale = 1.0F / std::sqrt(512.0F);
                REQUIRE(backend.dsv4_paged_attention(
                    device, request, std::span<float>(batched).subspan(
                        static_cast<std::size_t>(begin) * row_query_elements,
                        static_cast<std::size_t>(rows) * row_query_elements)).ok());
            }
            REQUIRE(reference.size() == batched.size());
            for (std::size_t index = 0U; index < reference.size(); ++index) {
                REQUIRE(std::bit_cast<std::uint32_t>(reference[index]) ==
                        std::bit_cast<std::uint32_t>(batched[index]));
            }
        }
    }

    // The production page calls the already-prepared index selection once
    // compressed history exceeds top-k. The Lightning Indexer fixture above
    // proves the 513 -> 512 selection itself; this continuation proves that a
    // different selected set per row retains exactly the same descriptor order
    // and arithmetic when those sparse rows are batched.
    constexpr std::uint32_t sparse_history = 513U;
    constexpr std::uint32_t sparse_selected = 512U;
    constexpr std::uint32_t sparse_rows = 8U;
    constexpr std::uint32_t sparse_candidate_width = 640U;
    const std::array<strata::CudaDsv4PhysicalPage, 3U> sparse_pages{{
        {&page_buffer, physical_rows},
        {&page_buffer, physical_rows},
        {&page_buffer, physical_rows},
    }};
    std::vector<float> sparse_queries(
        static_cast<std::size_t>(sparse_rows) * row_query_elements);
    for (std::uint32_t row = 0U; row < sparse_rows; ++row) {
        for (std::size_t index = 0U; index < row_query_elements; ++index) {
            const auto code = static_cast<int>(
                (row * 19U + static_cast<std::uint32_t>(index) * 7U) % 17U) -
                8;
            sparse_queries[static_cast<std::size_t>(row) * row_query_elements +
                           index] = round_bf16(static_cast<float>(code) / 64.0F);
        }
    }
    std::vector<strata::CudaDsv4AttentionCandidate> sparse_candidates(
        static_cast<std::size_t>(sparse_rows) * sparse_candidate_width);
    for (std::uint32_t row = 0U; row < sparse_rows; ++row) {
        const auto omitted = row % sparse_history;
        auto row_candidates = std::span(sparse_candidates).subspan(
            static_cast<std::size_t>(row) * sparse_candidate_width,
            sparse_candidate_width);
        std::uint32_t item = 0U;
        for (std::uint32_t position = 0U; position < sparse_history;
             ++position) {
            if (position == omitted) continue;
            row_candidates[item++] = {
                position / physical_rows, position % physical_rows, true};
        }
        REQUIRE(item == sparse_selected);
    }
    std::vector<float> sparse_reference(sparse_queries.size());
    for (std::uint32_t row = 0U; row < sparse_rows; ++row) {
        strata::CudaDsv4PagedAttentionRequest request;
        request.queries = std::span<const float>(sparse_queries).subspan(
            static_cast<std::size_t>(row) * row_query_elements,
            row_query_elements);
        request.head_sinks = sinks;
        request.pages = sparse_pages;
        request.candidates =
            std::span<const strata::CudaDsv4AttentionCandidate>(
                sparse_candidates).subspan(
                    static_cast<std::size_t>(row) * sparse_candidate_width,
                    sparse_candidate_width);
        request.candidate_width = sparse_candidate_width;
        request.scale = 1.0F / std::sqrt(512.0F);
        REQUIRE(backend.dsv4_paged_attention(
            device, request,
            std::span<float>(sparse_reference).subspan(
                static_cast<std::size_t>(row) * row_query_elements,
                row_query_elements)).ok());
    }
    for (const auto page_size : page_rows) {
        std::vector<float> sparse_batched(sparse_queries.size());
        for (std::uint32_t begin = 0U; begin < sparse_rows;
             begin += page_size) {
            const auto rows = std::min(page_size, sparse_rows - begin);
            strata::CudaDsv4PagedAttentionRequest request;
            request.queries = std::span<const float>(sparse_queries).subspan(
                static_cast<std::size_t>(begin) * row_query_elements,
                static_cast<std::size_t>(rows) * row_query_elements);
            request.head_sinks = sinks;
            request.pages = sparse_pages;
            request.candidates =
                std::span<const strata::CudaDsv4AttentionCandidate>(
                    sparse_candidates).subspan(
                        static_cast<std::size_t>(begin) *
                            sparse_candidate_width,
                        static_cast<std::size_t>(rows) *
                            sparse_candidate_width);
            request.rows = rows;
            request.candidate_width = sparse_candidate_width;
            request.maximum_workspace_bytes = 384ULL << 20U;
            request.scale = 1.0F / std::sqrt(512.0F);
            REQUIRE(backend.dsv4_paged_attention(
                device, request,
                std::span<float>(sparse_batched).subspan(
                    static_cast<std::size_t>(begin) * row_query_elements,
                    static_cast<std::size_t>(rows) * row_query_elements)).ok());
        }
        REQUIRE(sparse_reference.size() == sparse_batched.size());
        for (std::size_t index = 0U; index < sparse_reference.size(); ++index) {
            REQUIRE(std::bit_cast<std::uint32_t>(sparse_reference[index]) ==
                    std::bit_cast<std::uint32_t>(sparse_batched[index]));
        }
    }
}

TEST_CASE("native CUDA DeepSeek transformed expert shards match the canonical ones") {
    // Prefill uploads the routed experts in the host expert's decode layout,
    // because that is the only copy of them host memory holds. The kernels
    // therefore address two layouts, and the whole point is that they read the
    // same byte for the same (row, column): same accumulation order, same
    // bits, not a tolerance.
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;
    const auto device = devices.front();
    strata::CudaBackend backend;
    const std::array<int, 1> selected_devices{device};
    REQUIRE(backend.initialize(selected_devices, true).ok());

    constexpr std::uint64_t hidden = 128U;
    constexpr std::uint64_t intermediate = 192U;
    constexpr std::uint64_t shards = strata::kCudaDsv4TiledShards;
    constexpr std::uint32_t rows = 5U;
    constexpr float swiglu_limit = 10.0F;

    // Same bytes for both layouts, generated exactly as upload_fp4 does so the
    // canonical upload and the transform describe one expert.
    struct Payload {
        std::vector<std::byte> packed;
        std::vector<std::byte> scales;
    };
    const auto make_payload = [](std::uint64_t weight_rows,
                                 std::uint64_t columns, std::uint8_t seed) {
        Payload payload;
        const auto packed_columns = (columns + 1U) / 2U;
        const auto scale_columns = (columns + 31U) / 32U;
        payload.packed.resize(
            static_cast<std::size_t>(weight_rows * packed_columns));
        for (std::uint64_t row = 0U; row < weight_rows; ++row) {
            for (std::uint64_t packed = 0U; packed < packed_columns; ++packed) {
                const auto low = static_cast<std::uint8_t>(
                    (seed + row * 3U + packed * 5U) & 0x0FU);
                const auto high = static_cast<std::uint8_t>(
                    (seed + row * 7U + packed * 11U + 1U) & 0x0FU);
                payload.packed[static_cast<std::size_t>(
                    row * packed_columns + packed)] = static_cast<std::byte>(
                    low | static_cast<std::uint8_t>(high << 4U));
            }
        }
        payload.scales.resize(
            static_cast<std::size_t>(weight_rows * scale_columns));
        for (std::size_t index = 0U; index < payload.scales.size(); ++index) {
            payload.scales[index] = static_cast<std::byte>(
                0x78U + static_cast<std::uint8_t>((index + seed) % 3U));
        }
        return payload;
    };
    const auto w1 = make_payload(intermediate, hidden, 5U);
    const auto w3 = make_payload(intermediate, hidden, 12U);
    const auto w2 = make_payload(hidden, intermediate, 21U);

    strata::Dsv4HostExpertWeights canonical;
    canonical.w1_packed = w1.packed;
    canonical.w1_scales = w1.scales;
    canonical.w3_packed = w3.packed;
    canonical.w3_scales = w3.scales;
    canonical.w2_packed = w2.packed;
    canonical.w2_scales = w2.scales;

    const auto upload_canonical = [&](std::uint64_t weight_rows,
                                      std::uint64_t columns,
                                      const Payload& source) {
        strata::CudaWeightDescriptor descriptor;
        descriptor.encoding = strata::CudaWeightEncoding::Fp4E2m1Group32;
        descriptor.dtype = strata::SafetensorsDtype::I8;
        descriptor.rows = weight_rows;
        descriptor.columns = columns;
        descriptor.packed_columns = (columns + 1U) / 2U;
        descriptor.scale_columns = (columns + 31U) / 32U;
        descriptor.group_size = 32U;
        strata::CudaWeight weight;
        REQUIRE(backend.upload(device, descriptor, source.packed, source.scales,
                               weight).ok());
        return weight;
    };
    const auto canonical_w1 = upload_canonical(intermediate, hidden, w1);
    const auto canonical_w3 = upload_canonical(intermediate, hidden, w3);
    const auto canonical_w2 = upload_canonical(hidden, intermediate, w2);

    const auto shard_bytes = strata::dsv4_tiled_expert_shard_bytes(
        hidden, intermediate, shards);
    REQUIRE(shard_bytes != 0U);
    std::array<strata::CudaWeight, shards> shard_weights;
    for (std::uint64_t shard = 0U; shard < shards; ++shard) {
        std::vector<std::byte> storage(static_cast<std::size_t>(shard_bytes));
        REQUIRE(strata::dsv4_transform_tiled_expert_shard(
                    storage, canonical, hidden, intermediate, shard, shards)
                    .ok());
        strata::CudaWeightDescriptor descriptor;
        descriptor.encoding = strata::CudaWeightEncoding::Fp4E2m1Tiled32;
        descriptor.dtype = strata::SafetensorsDtype::I8;
        descriptor.rows = hidden;
        descriptor.columns = intermediate / shards;
        descriptor.group_size = 32U;
        REQUIRE(backend.upload(device, descriptor, storage, {},
                               shard_weights[shard]).ok());
    }

    std::vector<float> hidden_state(rows * static_cast<std::size_t>(hidden));
    std::uint32_t seed = 0x51a7c3U;
    for (auto& value : hidden_state) {
        seed = seed * 1'664'525U + 1'013'904'223U;
        value = static_cast<float>(
                    static_cast<std::int32_t>((seed >> 10U) % 2'048U) - 1'024) /
                4'096.0F;
    }
    const std::array<std::uint32_t, rows> group_rows{0U, 3U, 1U, 4U, 2U};
    const std::array<float, rows> coefficients{
        0.5F, 1.25F, 0.125F, 0.875F, 1.5F};

    const auto run = [&](bool tiled) {
        strata::CudaDeepSeekMoeRowGroup group;
        if (tiled) {
            for (std::size_t shard = 0U; shard < shards; ++shard) {
                group.tiled_shards[shard] = &shard_weights[shard];
            }
        } else {
            group.w1 = &canonical_w1;
            group.w3 = &canonical_w3;
            group.w2 = &canonical_w2;
        }
        group.rows = group_rows;
        group.coefficients = coefficients;
        const std::array<strata::CudaDeepSeekMoeRowGroup, 1> page{group};
        REQUIRE(backend.enqueue_deepseek_moe_rows(
                    device, hidden_state, rows, page, nullptr, {},
                    swiglu_limit).ok());
        std::vector<float> routed(rows * static_cast<std::size_t>(hidden));
        REQUIRE(backend.collect_deepseek_moe_rows(device, routed, {}).ok());
        return routed;
    };

    const auto expected = run(false);
    const auto actual = run(true);
    const auto repeated = run(true);
    REQUIRE(expected.size() == actual.size());
    REQUIRE(std::any_of(expected.begin(), expected.end(),
                        [](float value) { return value != 0.0F; }));

    // The transformed kernel is a reassociation, so it is held to the declared
    // numerical contract against the canonical one rather than to bit
    // equality -- but it must be deterministic in itself.
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        REQUIRE(std::bit_cast<std::uint32_t>(actual[index]) ==
                std::bit_cast<std::uint32_t>(repeated[index]));
    }
    float magnitude = 0.0F;
    for (const float value : expected) magnitude = std::max(magnitude, std::fabs(value));
    REQUIRE(magnitude > 0.0F);
    // One BF16 mantissa step of the page's largest canonical output. Both
    // paths round their result to BF16, so anything below this is invisible.
    const float tolerance = magnitude / 256.0F;
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        REQUIRE(std::fabs(actual[index] - expected[index]) <= tolerance);
    }
}

TEST_CASE("native CUDA DeepSeek device mHC slots interleave rows exactly") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;
    const auto device = devices.front();
    strata::CudaBackend backend;
    const std::array<int, 1> selected_devices{device};
    REQUIRE(backend.initialize(selected_devices, true).ok());
    if (!backend.validate_dsv4_mhc_device(device).ok()) return;

    constexpr std::size_t hidden = 4096U;
    constexpr std::size_t multiplier = 4U;
    constexpr std::size_t mixes = 24U;
    constexpr std::size_t rows = 3U;
    constexpr std::size_t transitions = 2U;

    // A degenerate all-ones state cannot distinguish an exact swap from a
    // stale one, so every input is a distinct deterministic value.
    std::uint32_t seed = 0x5eed1234U;
    const auto next_value = [&seed]() {
        seed = seed * 1'664'525U + 1'013'904'223U;
        return static_cast<float>(static_cast<std::int32_t>(seed >> 8U) %
                                  2'048) / 4'096.0F;
    };
    std::vector<float> projection(mixes * multiplier * hidden);
    for (auto& value : projection) value = next_value();
    std::array<float, 3> scale{0.5F, 1.25F, 0.75F};
    std::array<float, mixes> base{};
    for (auto& value : base) value = next_value();
    std::vector<float> norm(hidden);
    for (auto& value : norm) value = 1.0F + next_value();
    strata::CudaDsv4MhcWeights weights;
    REQUIRE(backend.upload_dsv4_mhc_weights(
        device, projection, scale, base, norm, weights).ok());

    std::vector<std::vector<float>> initial(rows);
    std::vector<std::array<std::vector<float>, transitions + 1U>> branches(rows);
    for (std::size_t row = 0U; row < rows; ++row) {
        initial[row].resize(multiplier * hidden);
        for (auto& value : initial[row]) value = next_value();
        for (auto& branch : branches[row]) {
            branch.resize(hidden);
            for (auto& value : branch) value = next_value();
        }
    }

    const auto bits = [](const std::vector<float>& values) {
        std::vector<std::uint32_t> encoded(values.size());
        for (std::size_t index = 0U; index < values.size(); ++index) {
            encoded[index] = std::bit_cast<std::uint32_t>(values[index]);
        }
        return encoded;
    };

    // Reference: every row runs to completion before the next one starts,
    // which is the accepted token-major order.
    std::vector<std::vector<std::uint32_t>> reference_inputs;
    std::vector<std::vector<std::uint32_t>> reference_residuals;
    for (std::size_t row = 0U; row < rows; ++row) {
        std::vector<float> layer_input(hidden);
        std::vector<float> residual(multiplier * hidden);
        REQUIRE(backend.dsv4_mhc_begin(
            device, weights, initial[row], std::span<float>{},
            layer_input).ok());
        reference_inputs.push_back(bits(layer_input));
        for (std::size_t step = 0U; step < transitions; ++step) {
            REQUIRE(backend.dsv4_mhc_transition(
                device, weights, branches[row][step], std::span<float>{},
                layer_input).ok());
            reference_inputs.push_back(bits(layer_input));
        }
        REQUIRE(backend.dsv4_mhc_finish(
            device, branches[row][transitions], residual).ok());
        reference_residuals.push_back(bits(residual));
    }

    // Candidate: one slot per row, advanced step by step across all rows.
    REQUIRE(backend.dsv4_mhc_reserve_slots(device, rows).ok());
    std::vector<std::vector<std::uint32_t>> candidate_inputs(
        rows * (transitions + 1U));
    std::vector<std::vector<std::uint32_t>> candidate_residuals(rows);
    std::vector<std::vector<float>> layer_inputs(
        rows, std::vector<float>(hidden));
    std::vector<std::vector<float>> residuals(
        rows, std::vector<float>(multiplier * hidden));
    for (std::size_t row = 0U; row < rows; ++row) {
        REQUIRE(backend.dsv4_mhc_select_slot(
            device, static_cast<std::uint32_t>(row)).ok());
        REQUIRE(backend.dsv4_mhc_begin(
            device, weights, initial[row], std::span<float>{},
            layer_inputs[row]).ok());
        candidate_inputs[row * (transitions + 1U)] = bits(layer_inputs[row]);
    }
    for (std::size_t step = 0U; step < transitions; ++step) {
        for (std::size_t row = 0U; row < rows; ++row) {
            REQUIRE(backend.dsv4_mhc_select_slot(
                device, static_cast<std::uint32_t>(row)).ok());
            REQUIRE(backend.dsv4_mhc_transition(
                device, weights, branches[row][step], std::span<float>{},
                layer_inputs[row]).ok());
            candidate_inputs[row * (transitions + 1U) + step + 1U] =
                bits(layer_inputs[row]);
        }
    }
    for (std::size_t row = 0U; row < rows; ++row) {
        REQUIRE(backend.dsv4_mhc_select_slot(
            device, static_cast<std::uint32_t>(row)).ok());
        REQUIRE(backend.dsv4_mhc_finish(
            device, branches[row][transitions], residuals[row]).ok());
        candidate_residuals[row] = bits(residuals[row]);
    }
    REQUIRE(backend.dsv4_mhc_select_slot(device, 0U).ok());

    REQUIRE(candidate_inputs == reference_inputs);
    REQUIRE(candidate_residuals == reference_residuals);

    // Out-of-range and out-of-order selections must be refused rather than
    // silently binding another row's state.
    REQUIRE(!backend.dsv4_mhc_select_slot(device, 8192U).ok());
    REQUIRE(!backend.dsv4_mhc_reserve_slots(device, 0U).ok());
    REQUIRE(!backend.dsv4_mhc_reserve_slots(device, 8193U).ok());
}

TEST_CASE("native CUDA DeepSeek device mHC keeps the residual across transitions") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;
    const auto device = devices.front();
    strata::CudaBackend backend;
    const std::array<int, 1> selected_devices{device};
    REQUIRE(backend.initialize(selected_devices, true).ok());
    if (!backend.validate_dsv4_mhc_device(device).ok()) return;

    constexpr std::size_t hidden = 4096U;
    constexpr std::size_t multiplier = 4U;
    constexpr std::size_t mixes = 24U;
    std::vector<float> projection(mixes * multiplier * hidden);
    std::array<float, 3> scale{};
    std::array<float, mixes> base{};
    std::vector<float> norm(hidden, 1.0F);
    strata::CudaDsv4MhcWeights weights;
    const auto before = backend.stats();
    REQUIRE(backend.upload_dsv4_mhc_weights(
        device, projection, scale, base, norm, weights).ok());
    REQUIRE(weights.valid());

    std::vector<float> residual(multiplier * hidden, 1.0F);
    std::vector<float> weighted(hidden);
    std::vector<float> layer_input(hidden);
    REQUIRE(backend.dsv4_mhc_begin(
        device, weights, residual, weighted, layer_input).ok());
    REQUIRE(std::all_of(weighted.begin(), weighted.end(), [](float value) {
        return value == 2.0F;
    }));
    REQUIRE(std::all_of(layer_input.begin(), layer_input.end(), [](float value) {
        return value == 1.0F;
    }));

    std::vector<float> branch(hidden, 1.0F);
    std::vector<float> post_residual(multiplier * hidden);
    REQUIRE(backend.dsv4_mhc_transition(
        device, weights, branch, weighted, layer_input,
        post_residual).ok());
    REQUIRE(std::all_of(post_residual.begin(), post_residual.end(),
                        [](float value) { return value == 2.0F; }));
    REQUIRE(std::all_of(weighted.begin(), weighted.end(), [](float value) {
        return value == 4.0F;
    }));
    REQUIRE(std::all_of(layer_input.begin(), layer_input.end(), [](float value) {
        return value == 1.0F;
    }));

    REQUIRE(backend.dsv4_mhc_finish(
        device, branch, residual).ok());
    REQUIRE(std::all_of(residual.begin(), residual.end(), [](float value) {
        return value == 3.0F;
    }));
    const auto after = backend.stats();
    REQUIRE(after.dsv4_mhc_calls - before.dsv4_mhc_calls == 3U);
    REQUIRE(after.dsv4_mhc_standalone_calls -
                before.dsv4_mhc_standalone_calls == 1U);
    REQUIRE(after.dsv4_mhc_transition_calls -
                before.dsv4_mhc_transition_calls == 1U);
    REQUIRE(after.dsv4_mhc_final_calls -
                before.dsv4_mhc_final_calls == 1U);
    REQUIRE(after.dsv4_mhc_kernel_launches -
                before.dsv4_mhc_kernel_launches == 8U);
    REQUIRE(after.dsv4_mhc_resident_weight_bytes -
                before.dsv4_mhc_resident_weight_bytes ==
            weights.device_bytes());
    REQUIRE(after.dsv4_mhc_h2d_bytes - before.dsv4_mhc_h2d_bytes ==
            49'152U);
    REQUIRE(after.dsv4_mhc_d2h_bytes - before.dsv4_mhc_d2h_bytes ==
            98'304U);
    REQUIRE(after.dsv4_mhc_kernel_nanoseconds -
                before.dsv4_mhc_kernel_nanoseconds < 60'000'000'000U);
    REQUIRE(after.dsv4_mhc_device_nanoseconds -
                before.dsv4_mhc_device_nanoseconds > 0U);
    REQUIRE(after.dsv4_mhc_device_nanoseconds -
                before.dsv4_mhc_device_nanoseconds < 60'000'000'000U);
    REQUIRE(after.dsv4_mhc_host_nanoseconds -
                before.dsv4_mhc_host_nanoseconds > 0U);
    REQUIRE(after.mhc_synchronization.calls -
                before.mhc_synchronization.calls == 3U);

    const auto production_before = backend.stats();
    REQUIRE(backend.dsv4_mhc_begin(
        device, weights, residual, std::span<float>{}, layer_input).ok());
    REQUIRE(std::all_of(layer_input.begin(), layer_input.end(), [](float value) {
        return value == 1.0F;
    }));
    REQUIRE(backend.dsv4_mhc_transition(
        device, weights, branch, std::span<float>{}, layer_input).ok());
    REQUIRE(std::all_of(layer_input.begin(), layer_input.end(), [](float value) {
        return value == 1.0F;
    }));
    REQUIRE(backend.dsv4_mhc_finish(device, branch, residual).ok());
    const auto production_after = backend.stats();
    REQUIRE(production_after.dsv4_mhc_h2d_bytes -
                production_before.dsv4_mhc_h2d_bytes ==
            49'152U);
    REQUIRE(production_after.dsv4_mhc_d2h_bytes -
                production_before.dsv4_mhc_d2h_bytes ==
            49'152U);

    // The live device path leaves the exact shared/routed BF16 result in
    // the persistent mHC branch buffer. Its consumer must reject missing or
    // stale producers and match the retained host bridge bit for bit.
    constexpr std::uint64_t shared_intermediate = 128U;
    auto shared_w1 = upload_fp8(
        backend, device, shared_intermediate, hidden, 1U);
    auto shared_w3 = upload_fp8(
        backend, device, shared_intermediate, hidden, 4U);
    auto shared_w2 = upload_fp8(
        backend, device, hidden, shared_intermediate, 7U);
    const strata::CudaDeepSeekMoeExpert shared{
        &shared_w1, &shared_w3, &shared_w2, 1.0F};
    Dsv4HostMoeCallbackFixture fixture;
    fixture.rank_partials.resize(2U * hidden);
    for (std::size_t index = 0U; index < fixture.rank_partials.size(); ++index) {
        fixture.rank_partials[index] =
            static_cast<float>(static_cast<int>(index % 13U) - 6) * 0.03125F;
    }

    const std::vector<float> initial(multiplier * hidden, 1.0F);
    std::vector<float> device_residual = initial;
    std::vector<float> device_input(hidden);
    std::vector<float> device_post(multiplier * hidden);
    std::vector<float> device_join(hidden);
    REQUIRE(backend.dsv4_mhc_begin(
        device, weights, device_residual, {}, device_input).ok());
    REQUIRE(!backend.dsv4_mhc_transition_device(
        device, weights, {}, device_input, device_post).ok());
    REQUIRE(backend.enqueue_dsv4_host_moe_from_mhc(
        device, shared, 10.0F, fill_dsv4_host_moe_partials,
        &fixture).ok());
    REQUIRE(backend.collect_deepseek_moe(device, {}, device_join).ok());
    REQUIRE(!backend.dsv4_mhc_transition(
        device, weights, branch, {}, device_input, device_post).ok());
    REQUIRE(backend.dsv4_mhc_transition_device(
        device, weights, {}, device_input, device_post).ok());
    REQUIRE(backend.dsv4_mhc_finish(
        device, branch, device_residual).ok());

    std::vector<float> host_residual = initial;
    std::vector<float> host_input(hidden);
    std::vector<float> host_post(multiplier * hidden);
    std::vector<float> host_join(hidden);
    REQUIRE(backend.dsv4_mhc_begin(
        device, weights, host_residual, {}, host_input).ok());
    REQUIRE(backend.enqueue_dsv4_host_moe(
        device, host_input, shared, 10.0F,
        fill_dsv4_host_moe_partials, &fixture).ok());
    REQUIRE(backend.collect_deepseek_moe(device, {}, host_join).ok());
    REQUIRE(backend.dsv4_mhc_transition(
        device, weights, host_join, {}, host_input, host_post).ok());
    REQUIRE(backend.dsv4_mhc_finish(
        device, branch, host_residual).ok());
    for (std::size_t index = 0U; index < hidden; ++index) {
        REQUIRE(std::bit_cast<std::uint32_t>(device_join[index]) ==
                std::bit_cast<std::uint32_t>(host_join[index]));
    }
    for (std::size_t index = 0U; index < device_post.size(); ++index) {
        REQUIRE(std::bit_cast<std::uint32_t>(device_post[index]) ==
                std::bit_cast<std::uint32_t>(host_post[index]));
    }

    std::vector<float> traffic_residual = initial;
    std::vector<float> traffic_input(hidden);
    REQUIRE(backend.dsv4_mhc_begin(
        device, weights, traffic_residual, {}, traffic_input).ok());
    const auto bridge_before = backend.stats();
    REQUIRE(backend.enqueue_dsv4_host_moe_from_mhc(
        device, shared, 10.0F, fill_dsv4_host_moe_partials,
        &fixture).ok());
    REQUIRE(backend.collect_deepseek_moe(device, {}, {}).ok());
    const auto bridge_after = backend.stats();
    REQUIRE(bridge_after.deepseek_moe_kernel_launches -
                bridge_before.deepseek_moe_kernel_launches == 5U);
    REQUIRE(bridge_after.deepseek_moe_h2d_transfers -
                bridge_before.deepseek_moe_h2d_transfers == 1U);
    REQUIRE(bridge_after.deepseek_moe_h2d_bytes -
                bridge_before.deepseek_moe_h2d_bytes ==
            2U * hidden * sizeof(float));
    REQUIRE(bridge_after.deepseek_moe_d2h_transfers -
                bridge_before.deepseek_moe_d2h_transfers == 1U);
    REQUIRE(bridge_after.deepseek_moe_d2h_bytes -
                bridge_before.deepseek_moe_d2h_bytes == sizeof(unsigned int));
    const auto finish_before = backend.stats();
    REQUIRE(backend.dsv4_mhc_finish_device(
        device, traffic_residual).ok());
    const auto finish_after = backend.stats();
    REQUIRE(finish_after.dsv4_mhc_h2d_bytes -
                finish_before.dsv4_mhc_h2d_bytes == 0U);
}

TEST_CASE("rank-local CUDA bridges fail closed before a borrowed lifetime exists") {
    strata::CudaBackend backend;
    strata::CudaDsv4MhcDeviceView mhc_view{};
    strata::CudaDsv4HostMoeDeviceView moe_view{};
    strata::CudaDsv4MhcWeights weights;
    strata::CudaWeight router;

    REQUIRE(!backend.dsv4_mhc_device_view(0, mhc_view).ok());
    REQUIRE(!backend.dsv4_mhc_branch_to_fp32(0, nullptr).ok());
    REQUIRE(!backend.dsv4_mhc_commit_reduced_branch(0, nullptr).ok());
    REQUIRE(!backend.dsv4_mhc_abort_branch(0).ok());
    REQUIRE(!backend.dsv4_mhc_transition_router_device(0, weights, router).ok());
    REQUIRE(!backend.dsv4_mhc_transition_next_device(0, weights).ok());
    REQUIRE(!backend.enqueue_dsv4_host_moe_from_device_input_device_view(
        0, strata::CudaDeepSeekMoeExpert{}, 10.0F, nullptr, nullptr,
        moe_view).ok());
    REQUIRE(mhc_view.stream == nullptr);
    REQUIRE(moe_view.stream == nullptr);
}

TEST_CASE("MIX-2 register-fed matmul matches the scalar route it replaces") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;
    strata::CudaBackend backend;
    const int device = devices.front();
    const std::vector<int> selected{device};
    REQUIRE(backend.initialize(selected, true).ok());

    // Activation values are exact E4M3 numbers, so matmul_impl's activation
    // quantization is the identity and both routes see the same operands. That
    // makes the comparison a statement about the kernels rather than about the
    // rounding in front of them.
    constexpr std::array<float, 6> palette{1.0F, -1.0F, 0.5F, 2.0F, -0.5F, -2.0F};

    const auto compare = [&](strata::CudaWeightEncoding encoding,
                             std::uint64_t rows, std::uint64_t columns,
                             std::uint32_t batch, std::uint8_t seed) {
        std::vector<float> activation(
            static_cast<std::size_t>(columns) * batch);
        for (std::size_t index = 0U; index < activation.size(); ++index) {
            activation[index] = palette[(index + seed) % palette.size()];
        }
        const auto upload = [&](std::uint8_t s) {
            return encoding == strata::CudaWeightEncoding::Fp4E2m1Group32
                       ? upload_fp4(backend, device, rows, columns, s)
                       : upload_fp8(backend, device, rows, columns, s);
        };
        // Two uploads of the same payload: the register-fed route permutes its
        // weight in place, so the control needs its own canonical copy.
        const auto control = upload(seed);
        const auto candidate = upload(seed);

        std::vector<float> expected(static_cast<std::size_t>(rows) * batch);
        strata::set_register_fed_matmul(false);
        REQUIRE(backend.matmul(control, activation, batch, expected).ok());

        std::vector<float> measured(expected.size());
        strata::set_register_fed_matmul(true);
        REQUIRE(backend.matmul(candidate, activation, batch, measured).ok());
        REQUIRE(strata::CudaBackend::fragment_prepacked(candidate));
        REQUIRE(!strata::CudaBackend::fragment_prepacked(control));

        // The two paths multiply the same real numbers and differ only in FP32
        // accumulation order, so the residual is reordering, not precision. The
        // bound is relative to the row's own magnitude because these fixtures
        // have rows whose sums cancel to near zero.
        double worst = 0.0;
        for (std::size_t index = 0U; index < expected.size(); ++index) {
            const double scale = std::max(1.0, std::abs(
                static_cast<double>(expected[index])));
            worst = std::max(worst, std::abs(static_cast<double>(measured[index]) -
                                             static_cast<double>(expected[index])) /
                                        scale);
        }
        if (!(worst < 1e-4)) {
            std::fprintf(stderr,
                         "register-fed mismatch: encoding %d rows %llu "
                         "columns %llu batch %u worst relative residual %g\n",
                         static_cast<int>(encoding),
                         static_cast<unsigned long long>(rows),
                         static_cast<unsigned long long>(columns), batch, worst);
        }
        REQUIRE(worst < 1e-4);
    };

    // FP4 needs rows%16 and columns%64; FP8 needs both extents on 128.
    compare(strata::CudaWeightEncoding::Fp4E2m1Group32, 64U, 128U, 1U, 0x11U);
    compare(strata::CudaWeightEncoding::Fp4E2m1Group32, 64U, 128U, 8U, 0x23U);
    compare(strata::CudaWeightEncoding::Fp4E2m1Group32, 128U, 256U, 16U, 0x35U);
    compare(strata::CudaWeightEncoding::Fp8E4m3Block128, 128U, 128U, 1U, 0x11U);
    compare(strata::CudaWeightEncoding::Fp8E4m3Block128, 128U, 256U, 5U, 0x27U);
    compare(strata::CudaWeightEncoding::Fp8E4m3Block128, 256U, 256U, 16U, 0x41U);
    strata::set_register_fed_matmul(true);
}

TEST_CASE("MIX-2 register-fed dispatch leaves inadmissible shapes on the scalar route") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;
    strata::CudaBackend backend;
    const int device = devices.front();
    const std::vector<int> selected{device};
    REQUIRE(backend.initialize(selected, true).ok());
    strata::set_register_fed_matmul(true);

    // 48 columns is neither a whole FP4 load block (64) nor an FP8 block (128),
    // so neither weight may be permuted and both must still produce a result.
    const auto fp4 = upload_fp4(backend, device, 32U, 48U, 0x19U);
    std::vector<float> activation(48U, 0.5F);
    std::vector<float> output(32U);
    REQUIRE(backend.matmul(fp4, activation, 1U, output).ok());
    REQUIRE(!strata::CudaBackend::fragment_prepacked(fp4));

    const auto fp8 = upload_fp8(backend, device, 32U, 64U, 0x19U);
    std::vector<float> narrow(64U, 0.5F);
    std::vector<float> fp8_output(32U);
    REQUIRE(backend.matmul(fp8, narrow, 1U, fp8_output).ok());
    REQUIRE(!strata::CudaBackend::fragment_prepacked(fp8));

    // The explicit entry point must say so rather than permute into a layout
    // the kernel cannot read.
    REQUIRE(!backend.prepack_fragment(device, fp4).ok());
    REQUIRE(!backend.prepack_fragment(device, fp8).ok());
}
