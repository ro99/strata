#include "test.hpp"

#include "strata/cuda_backend.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

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

}  // namespace

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
