#include "test.hpp"

#include "strata/compressed_tensors.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

void store_le_u32(std::byte* output, std::uint32_t value) {
    for (std::size_t index = 0; index < 4U; ++index) {
        output[index] = static_cast<std::byte>((value >> (index * 8U)) & 0xFFU);
    }
}

void store_le_u64(std::byte* output, std::uint64_t value) {
    for (std::size_t index = 0; index < 8U; ++index) {
        output[index] = static_cast<std::byte>((value >> (index * 8U)) & 0xFFU);
    }
}

}  // namespace

TEST_CASE("real QuantTrio INT4 group-128 expert layout is accepted") {
    // Revision 1d3bcfe5, model-00005-of-00124, layers.11.mlp.experts.16.down_proj.
    const auto model = strata::quanttrio_glm52_int4_int8_mix_spec();
    strata::CompressedTensorLayout layout;
    layout.logical_rows = 6144;
    layout.logical_columns = 2048;
    layout.packed_rows = 6144;
    layout.packed_columns = 256;
    layout.scale_rows = 6144;
    layout.scale_columns = 16;
    REQUIRE(strata::validate_compressed_tensor_layout(
                layout, model.mixed_quantization.routed_experts)
                .ok());

    ++layout.packed_columns;
    REQUIRE(!strata::validate_compressed_tensor_layout(
                 layout, model.mixed_quantization.routed_experts)
                 .ok());
}

TEST_CASE("real QuantTrio INT8 group-128 attention layout is accepted") {
    // Revision 1d3bcfe5, model-00037-of-00124, layers.3.self_attn.q_a_proj.
    const auto model = strata::quanttrio_glm52_int4_int8_mix_spec();
    strata::CompressedTensorLayout layout;
    layout.logical_rows = 2048;
    layout.logical_columns = 6144;
    layout.packed_rows = 2048;
    layout.packed_columns = 1536;
    layout.scale_rows = 2048;
    layout.scale_columns = 48;
    REQUIRE(strata::validate_compressed_tensor_layout(
                layout, model.mixed_quantization.linears)
                .ok());
}

TEST_CASE("real QuantTrio channelwise INT8 MTP layout is accepted") {
    // Revision 1d3bcfe5, mtp-00001-of-00004, layers.78.mlp.experts.0.gate_proj.
    const auto model = strata::quanttrio_glm52_int4_int8_mix_spec();
    strata::CompressedTensorLayout layout;
    layout.logical_rows = 2048;
    layout.logical_columns = 6144;
    layout.packed_rows = 2048;
    layout.packed_columns = 1536;
    layout.scale_rows = 2048;
    layout.scale_columns = 1;
    REQUIRE(strata::validate_compressed_tensor_layout(layout,
                                                       model.mixed_quantization.mtp)
                .ok());

    layout.scale_columns = 48;
    REQUIRE(!strata::validate_compressed_tensor_layout(layout,
                                                        model.mixed_quantization.mtp)
                 .ok());
}

TEST_CASE("compressed logical shape decodes exact little-endian I64 values") {
    std::array<std::byte, 16> encoded{};
    store_le_u64(encoded.data(), 6144U);
    store_le_u64(encoded.data() + 8U, 2048U);
    std::array<std::uint64_t, 2> shape{};
    REQUIRE(strata::decode_compressed_logical_shape(encoded, shape).ok());
    REQUIRE(shape[0] == 6144U);
    REQUIRE(shape[1] == 2048U);
}

TEST_CASE("INT4 group-128 matvec follows least-significant-lane offset packing") {
    std::array<std::byte, 64> packed{};
    for (std::size_t word_index = 0; word_index < 16U; ++word_index) {
        std::uint32_t word = 0;
        for (std::uint32_t lane = 0; lane < 8U; ++lane) {
            const auto value = static_cast<std::int32_t>((word_index * 8U + lane) % 16U) - 8;
            const auto nibble = static_cast<std::uint32_t>(value + 8);
            word |= nibble << (lane * 4U);
        }
        store_le_u32(packed.data() + word_index * 4U, word);
    }
    const std::array<std::byte, 2> scales{std::byte{0x00}, std::byte{0x40}};  // BF16 2.0
    const std::array<float, 128> input = [] {
        std::array<float, 128> values{};
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] = static_cast<float>(index + 1U);
        }
        return values;
    }();
    std::array<float, 1> output{};
    strata::CompressedTensorLayout layout;
    layout.logical_rows = 1;
    layout.logical_columns = 128;
    layout.packed_rows = 1;
    layout.packed_columns = 16;
    layout.scale_rows = 1;
    layout.scale_columns = 1;
    const auto quantization = strata::QuantizedWeightSpec{
        4, strata::QuantizationGranularity::Group, 128, true};
    REQUIRE(strata::compressed_tensor_matvec_f32(output, input, packed, scales, layout,
                                                  quantization)
                .ok());
    float expected = 0.0F;
    for (std::size_t index = 0; index < input.size(); ++index) {
        expected += static_cast<float>(static_cast<std::int32_t>(index % 16U) - 8) *
                    2.0F * input[index];
    }
    REQUIRE(output[0] == expected);
}

TEST_CASE("INT8 channelwise matvec uses a distinct BF16 scale per output row") {
    std::array<std::byte, 8> packed{};
    store_le_u32(packed.data(), 0x7C83'7E81U);      // [1, -2, 3, -4] + 128
    store_le_u32(packed.data() + 4U, 0x8483'8281U);  // [1, 2, 3, 4] + 128
    const std::array<std::byte, 4> scales{
        std::byte{0x80}, std::byte{0x3F},  // BF16 1.0
        std::byte{0x00}, std::byte{0x3F},  // BF16 0.5
    };
    const std::array<float, 4> input{1.0F, 1.0F, 1.0F, 1.0F};
    std::array<float, 2> output{};
    strata::CompressedTensorLayout layout;
    layout.logical_rows = 2;
    layout.logical_columns = 4;
    layout.packed_rows = 2;
    layout.packed_columns = 1;
    layout.scale_rows = 2;
    layout.scale_columns = 1;
    const auto quantization = strata::QuantizedWeightSpec{
        8, strata::QuantizationGranularity::Channel, 0, true};
    REQUIRE(strata::compressed_tensor_matvec_f32(output, input, packed, scales, layout,
                                                  quantization)
                .ok());
    REQUIRE(output[0] == -2.0F);
    REQUIRE(output[1] == 5.0F);
}
