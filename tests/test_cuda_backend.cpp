#include "test.hpp"

#include "strata/cuda_backend.hpp"
#include "strata/deepseek_ops.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

}  // namespace

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
    REQUIRE(backend.matmul(weight, input, 1U, output).ok());
    REQUIRE(output[0] == -60.0F);
    REQUIRE(output[1] == 10.0F);
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
    const auto reference0 = reference_expert(
        backend, routed0_w1, routed0_w3, routed0_w2, hidden,
        intermediate_columns, coefficient0, true);
    const auto reference1 = reference_expert(
        backend, routed1_w1, routed1_w3, routed1_w2, hidden,
        intermediate_columns, coefficient1, true);
    const auto reference_shared = reference_expert(
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
        const float difference0 = std::abs(routed_output[column] - reference0[column]);
        const float difference1 = std::abs(
            routed_output[hidden_columns + column] - reference1[column]);
        const float shared_difference =
            std::abs(shared_output[column] - reference_shared[column]);
        maximum_difference = std::max(
            maximum_difference,
            std::max(difference0, std::max(difference1, shared_difference)));
        REQUIRE(std::bit_cast<std::uint32_t>(routed_output[column]) ==
                std::bit_cast<std::uint32_t>(reference0[column]));
        REQUIRE(std::bit_cast<std::uint32_t>(
                    routed_output[hidden_columns + column]) ==
                std::bit_cast<std::uint32_t>(reference1[column]));
        REQUIRE(std::bit_cast<std::uint32_t>(shared_output[column]) ==
                std::bit_cast<std::uint32_t>(reference_shared[column]));
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
                before.deepseek_moe_h2d_transfers == 2U);
    REQUIRE(after.deepseek_moe_d2h_transfers -
                before.deepseek_moe_d2h_transfers == 2U);
    REQUIRE(after.deepseek_moe_h2d_bytes - before.deepseek_moe_h2d_bytes >
            hidden_columns * sizeof(float));
    REQUIRE(after.deepseek_moe_d2h_bytes - before.deepseek_moe_d2h_bytes ==
            3U * hidden_columns * sizeof(float) + sizeof(unsigned int));
    REQUIRE(after.matmul_calls - before.matmul_calls == 9U);
    REQUIRE(after.activation_h2d_bytes - before.activation_h2d_bytes ==
            hidden_columns * sizeof(float));
    REQUIRE(after.activation_d2h_bytes - before.activation_d2h_bytes ==
            3U * hidden_columns * sizeof(float));
    REQUIRE(after.workspace_allocation_calls -
                before.workspace_allocation_calls == 7U);
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

TEST_CASE("native CUDA backend batches repeated DeepSeek experts across rows") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;
    constexpr std::uint64_t hidden_columns = 32U;
    constexpr std::uint64_t intermediate_columns = 32U;
    constexpr std::uint32_t hidden_rows = 3U;
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

    std::array<float, hidden_rows * hidden_columns> hidden{};
    for (std::size_t index = 0U; index < hidden.size(); ++index) {
        hidden[index] = static_cast<float>(static_cast<int>(index % 13U) - 6) *
                        0.0625F;
    }
    constexpr std::array<strata::CudaDeepSeekMoeRow, 2> routed0_rows{{
        {0U, 0.75F}, {2U, 0.5F},
    }};
    constexpr std::array<strata::CudaDeepSeekMoeRow, 1> routed1_rows{{
        {1U, 0.3125F},
    }};
    const std::array<strata::CudaDeepSeekMoeGroup, 2> routed{{
        {&routed0_w1, &routed0_w3, &routed0_w2, routed0_rows},
        {&routed1_w1, &routed1_w3, &routed1_w2, routed1_rows},
    }};
    const strata::CudaDeepSeekMoeExpert shared{
        &shared_w1, &shared_w3, &shared_w2, 1.0F};

    std::array<std::vector<float>, 3> routed_reference{
        reference_expert(
            backend, routed0_w1, routed0_w3, routed0_w2,
            std::span<const float>(hidden).subspan(0U, hidden_columns),
            intermediate_columns, routed0_rows[0].coefficient, true),
        reference_expert(
            backend, routed0_w1, routed0_w3, routed0_w2,
            std::span<const float>(hidden).subspan(
                2U * hidden_columns, hidden_columns),
            intermediate_columns, routed0_rows[1].coefficient, true),
        reference_expert(
            backend, routed1_w1, routed1_w3, routed1_w2,
            std::span<const float>(hidden).subspan(
                hidden_columns, hidden_columns),
            intermediate_columns, routed1_rows[0].coefficient, true),
    };
    std::array<std::vector<float>, hidden_rows> shared_reference;
    for (std::size_t row = 0U; row < hidden_rows; ++row) {
        shared_reference[row] = reference_expert(
            backend, shared_w1, shared_w3, shared_w2,
            std::span<const float>(hidden).subspan(
                row * hidden_columns, hidden_columns),
            intermediate_columns, 1.0F, false);
    }

    std::array<float, 3U * hidden_columns> routed_output{};
    std::array<float, hidden_rows * hidden_columns> shared_output{};
    const auto before = backend.stats();
    REQUIRE(backend.enqueue_deepseek_moe_batch(
        device, hidden, hidden_rows, routed, &shared, 10.0F).ok());
    REQUIRE(backend.collect_deepseek_moe(
        device, routed_output, shared_output).ok());
    for (std::size_t row = 0U; row < routed_reference.size(); ++row) {
        for (std::size_t column = 0U; column < hidden_columns; ++column) {
            REQUIRE(std::bit_cast<std::uint32_t>(
                        routed_output[row * hidden_columns + column]) ==
                    std::bit_cast<std::uint32_t>(routed_reference[row][column]));
        }
    }
    for (std::size_t row = 0U; row < hidden_rows; ++row) {
        for (std::size_t column = 0U; column < hidden_columns; ++column) {
            REQUIRE(std::bit_cast<std::uint32_t>(
                        shared_output[row * hidden_columns + column]) ==
                    std::bit_cast<std::uint32_t>(shared_reference[row][column]));
        }
    }
    const auto after = backend.stats();
    REQUIRE(after.deepseek_moe_calls - before.deepseek_moe_calls == 1U);
    REQUIRE(after.deepseek_moe_kernel_launches -
                before.deepseek_moe_kernel_launches == 7U);
    REQUIRE(after.matmul_calls - before.matmul_calls == 18U);
}
