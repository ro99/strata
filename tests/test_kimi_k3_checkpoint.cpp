#include "test.hpp"

#include "strata/platform/compressed_tensors.hpp"
#include "strata/models/kimi_k3/kimi_k3_checkpoint.hpp"
#include "strata/models/common/model_adapter.hpp"

#include <bit>
#include <cstddef>
#include <filesystem>
#include <vector>

namespace {

std::string kimi_directory() {
    return (std::filesystem::path(STRATA_SOURCE_DIR) / "models/kimi-k3").string();
}

bool kimi_present() {
    return std::filesystem::exists(
        std::filesystem::path(kimi_directory()) / "model.safetensors.index.json");
}

strata::CompressedTensorLayout synthetic_layout(std::uint64_t rows,
                                                std::uint64_t columns) {
    return strata::kimi_expert_layout(rows, columns);
}

}  // namespace

TEST_CASE("E8M0 block scales decode as exact powers of two") {
    // The encoding is the float32 exponent field: 2^(bits - 127).
    REQUIRE(strata::mxfp4_scale_from_e8m0(127U) == 1.0F);
    REQUIRE(strata::mxfp4_scale_from_e8m0(128U) == 2.0F);
    REQUIRE(strata::mxfp4_scale_from_e8m0(126U) == 0.5F);
    REQUIRE(strata::mxfp4_scale_from_e8m0(121U) == 0.015625F);
    REQUIRE(strata::mxfp4_scale_from_e8m0(0U) == 0.0F);
    // The encoding reserves 0xFF, and the reference's bit shift renders it as
    // infinity. It must not become a finite scale, and the decoder must refuse
    // a block that carries it.
    REQUIRE(!std::isfinite(strata::mxfp4_scale_from_e8m0(255U)));

    const auto layout = synthetic_layout(1U, 32U);
    const auto quantization = strata::kimi_expert_quantization();
    const std::vector<std::byte> packed(16U, std::byte{0x22});
    const std::vector<std::byte> reserved(1U, std::byte{255U});
    std::vector<float> row(32U);
    REQUIRE(!strata::mxfp4_dequantize_row(row, packed, reserved, layout,
                                          quantization, 0U).ok());
}

TEST_CASE("MXFP4 nibble order, sign, and magnitude table are exact") {
    // One row, 32 columns, one group, unit scale.
    const auto layout = synthetic_layout(1U, 32U);
    const auto quantization = strata::kimi_expert_quantization();
    std::vector<std::byte> packed(16U, std::byte{0});
    std::vector<std::byte> scales(1U, std::byte{127U});

    // Low nibble is the even element, high nibble the odd one. Byte 0 encodes
    // column 0 = index 1 (0.5) and column 1 = index 2 | sign (-1.0).
    packed[0] = std::byte{static_cast<unsigned char>(0x01U | (0x0AU << 4U))};
    // Every magnitude index, so a transposed table cannot pass.
    packed[1] = std::byte{static_cast<unsigned char>(0x03U | (0x04U << 4U))};
    packed[2] = std::byte{static_cast<unsigned char>(0x05U | (0x06U << 4U))};
    packed[3] = std::byte{static_cast<unsigned char>(0x07U | (0x0FU << 4U))};

    std::vector<float> row(32U);
    REQUIRE(strata::mxfp4_dequantize_row(row, packed, scales, layout,
                                         quantization, 0U).ok());
    REQUIRE(row[0] == 0.5F);
    REQUIRE(row[1] == -1.0F);
    REQUIRE(row[2] == 1.5F);
    REQUIRE(row[3] == 2.0F);
    REQUIRE(row[4] == 3.0F);
    REQUIRE(row[5] == 4.0F);
    REQUIRE(row[6] == 6.0F);
    REQUIRE(row[7] == -6.0F);
    for (std::size_t index = 8U; index < row.size(); ++index) {
        REQUIRE(row[index] == 0.0F);
    }
}

TEST_CASE("MXFP4 group scales apply per 32 columns") {
    const auto layout = synthetic_layout(1U, 64U);
    const auto quantization = strata::kimi_expert_quantization();
    std::vector<std::byte> packed(32U, std::byte{0});
    // Index 2 (1.0) in every element of both groups.
    for (auto& byte : packed) {
        byte = std::byte{static_cast<unsigned char>(0x02U | (0x02U << 4U))};
    }
    std::vector<std::byte> scales{std::byte{127U}, std::byte{129U}};
    std::vector<float> row(64U);
    REQUIRE(strata::mxfp4_dequantize_row(row, packed, scales, layout,
                                         quantization, 0U).ok());
    REQUIRE(row[0] == 1.0F);
    REQUIRE(row[31] == 1.0F);
    REQUIRE(row[32] == 4.0F);
    REQUIRE(row[63] == 4.0F);
}

TEST_CASE("MXFP4 layout validation rejects sub-four-bit and mismatched shapes") {
    auto layout = synthetic_layout(4U, 64U);
    auto quantization = strata::kimi_expert_quantization();
    REQUIRE(strata::validate_compressed_tensor_layout(layout, quantization).ok());

    // Below four bits is forbidden everywhere, including storage codecs.
    auto narrow = quantization;
    narrow.bits = 2U;
    REQUIRE(!strata::validate_compressed_tensor_layout(layout, narrow).ok());

    auto wrong_scale_dtype = layout;
    wrong_scale_dtype.scale_dtype = strata::SafetensorsDtype::Bf16;
    REQUIRE(!strata::validate_compressed_tensor_layout(wrong_scale_dtype,
                                                       quantization).ok());

    auto wrong_packed = layout;
    wrong_packed.packed_columns = 64U;
    REQUIRE(!strata::validate_compressed_tensor_layout(wrong_packed,
                                                       quantization).ok());

    auto wrong_group = quantization;
    wrong_group.group_size = 64U;
    REQUIRE(!strata::validate_compressed_tensor_layout(layout, wrong_group).ok());
}

TEST_CASE("MXFP4 matvec agrees with dequantize-then-dot") {
    const auto layout = synthetic_layout(3U, 32U);
    const auto quantization = strata::kimi_expert_quantization();
    std::vector<std::byte> packed(3U * 16U);
    for (std::size_t index = 0U; index < packed.size(); ++index) {
        packed[index] = std::byte{static_cast<unsigned char>((index * 37U) & 0xFFU)};
    }
    std::vector<std::byte> scales{std::byte{126U}, std::byte{127U}, std::byte{128U}};
    std::vector<float> input(32U);
    for (std::size_t index = 0U; index < input.size(); ++index) {
        input[index] = static_cast<float>(index % 7U) - 3.0F;
    }

    std::vector<float> product(3U);
    REQUIRE(strata::compressed_tensor_matvec_f32(product, input, packed, scales,
                                                 layout, quantization).ok());
    for (std::uint64_t row = 0U; row < 3U; ++row) {
        std::vector<float> decoded(32U);
        REQUIRE(strata::mxfp4_dequantize_row(decoded, packed, scales, layout,
                                             quantization, row).ok());
        float sum = 0.0F;
        for (std::size_t index = 0U; index < decoded.size(); ++index) {
            sum += decoded[index] * input[index];
        }
        REQUIRE(product[static_cast<std::size_t>(row)] == sum);
    }
}

TEST_CASE("real Kimi-K3 checkpoint opens and pins its geometry") {
    if (!kimi_present()) SKIP("pinned Kimi-K3 checkpoint is absent");
    const auto opened = strata::KimiCheckpointReader::open(kimi_directory());
    REQUIRE(opened.ok());
    const auto& reader = *opened.value;
    const auto& manifest = reader.manifest();

    REQUIRE(manifest.tensors.size() == 497'220U);
    REQUIRE(manifest.shards.size() == 96U);
    REQUIRE(manifest.kda_layers == 69U);
    REQUIRE(manifest.full_attention_layers == 24U);
    REQUIRE(manifest.moe_layers == 92U);
    REQUIRE(manifest.routed_expert_modules == 896U * 3U * 92U);

    // The byte split is the whole placement problem: the routed experts are
    // 92.7% of the checkpoint and the only quantized part of it.
    const double total = static_cast<double>(manifest.tensor_payload_bytes);
    const double routed = static_cast<double>(manifest.routed_expert_bytes);
    REQUIRE(manifest.routed_expert_bytes + manifest.dense_spine_bytes ==
            manifest.tensor_payload_bytes);
    REQUIRE(routed / total > 0.92);
    REQUIRE(routed / total < 0.93);
    // 1347.12 GiB of experts against 106.55 GiB of BF16 spine.
    REQUIRE(routed > 1440.0e9 && routed < 1450.0e9);

    // Every routed-expert module is MXFP4; nothing else is quantized at all.
    for (const auto& tensor : manifest.tensors) {
        const bool routed_expert = tensor.role == strata::KimiTensorRole::RoutedExpert;
        REQUIRE(routed_expert ==
                (tensor.encoding == strata::KimiTensorEncoding::Mxfp4Group32));
        if (!routed_expert) {
            REQUIRE(tensor.source_dtype == strata::SafetensorsDtype::Bf16 ||
                    tensor.source_dtype == strata::SafetensorsDtype::F32);
        }
    }
}

TEST_CASE("real Kimi-K3 A_log is per-head and zero-padded to head_dim") {
    if (!kimi_present()) SKIP("pinned Kimi-K3 checkpoint is absent");
    const auto opened = strata::KimiCheckpointReader::open(kimi_directory());
    REQUIRE(opened.ok());
    const auto& contract = strata::kKimiK3ExecutionContract;

    // Reading it as a per-channel tensor would apply exp(0) = 1 to the padding
    // and corrupt the decay on a quarter of the channels, so the padding being
    // exactly zero is what makes the per-head reading falsifiable.
    for (const auto layer : {0U, 1U, 2U, 90U}) {
        const auto name = "language_model.model.layers." + std::to_string(layer) +
                          ".self_attn.A_log";
        const auto values = opened.value->read_f32(name, 1024U);
        REQUIRE(values.ok());
        REQUIRE(values.value.size() == contract.linear_head_dim);
        for (std::size_t index = contract.attention_heads;
             index < values.value.size(); ++index) {
            REQUIRE(values.value[index] == 0.0F);
        }
        bool any_nonzero = false;
        for (std::size_t index = 0U; index < contract.attention_heads; ++index) {
            if (values.value[index] != 0.0F) any_nonzero = true;
        }
        REQUIRE(any_nonzero);
    }
}

TEST_CASE("real Kimi-K3 expert dequantizes to the reference values") {
    if (!kimi_present()) SKIP("pinned Kimi-K3 checkpoint is absent");
    const auto opened = strata::KimiCheckpointReader::open(kimi_directory());
    REQUIRE(opened.ok());
    const auto& reader = *opened.value;

    strata::KimiExpertModules modules{};
    REQUIRE(reader.expert_modules(1U, 0U, modules));
    REQUIRE(reader.expert_source_bytes() == 17'547'264ULL);
    REQUIRE(modules.gate.packed_bytes + modules.gate.scale_bytes +
                modules.up.packed_bytes + modules.up.scale_bytes +
                modules.down.packed_bytes + modules.down.scale_bytes ==
            reader.expert_source_bytes());

    // Oracle: `compressed_tensors.compressors.nvfp4.helpers.unpack_fp4_from_uint8`
    // on the real packed payload, scaled by (E8M0 byte << 23) reinterpreted as
    // float32. These are that function's outputs for layer 1, expert 0, not a
    // restatement of Strata's own decode.
    const float gate_row0[8] = {0.0625F,   0.0078125F, -0.0078125F, -0.0234375F,
                                -0.0625F, -0.03125F,   0.0234375F,  0.015625F};
    const float up_row0[8] = {0.0078125F, -0.0078125F, 0.03125F,   -0.046875F,
                              -0.0234375F, 0.0078125F, 0.046875F, -0.0234375F};
    const float down_row0[8] = {0.0078125F, -0.046875F, 0.0234375F, 0.046875F,
                                -0.03125F, -0.046875F,  0.046875F,  0.015625F};
    const float gate_last_row[4] = {0.0078125F, -0.0F, 0.0625F, 0.0F};
    const float down_last_row[4] = {0.03125F, -0.0234375F, -0.0F, -0.0625F};

    const auto& contract = strata::kKimiK3ExecutionContract;
    const auto inner = static_cast<std::uint64_t>(contract.expert_intermediate_size);
    const auto latent =
        static_cast<std::uint64_t>(contract.routed_expert_hidden_size);

    std::vector<float> gate(static_cast<std::size_t>(inner * latent));
    REQUIRE(reader.read_expert_module_f32(modules.gate, inner, latent, gate).ok());
    std::vector<float> up(static_cast<std::size_t>(inner * latent));
    REQUIRE(reader.read_expert_module_f32(modules.up, inner, latent, up).ok());
    std::vector<float> down(static_cast<std::size_t>(latent * inner));
    REQUIRE(reader.read_expert_module_f32(modules.down, latent, inner, down).ok());

    for (std::size_t index = 0U; index < 8U; ++index) {
        REQUIRE(gate[index] == gate_row0[index]);
        REQUIRE(up[index] == up_row0[index]);
        REQUIRE(down[index] == down_row0[index]);
    }
    for (std::size_t index = 0U; index < 4U; ++index) {
        REQUIRE(gate[static_cast<std::size_t>((inner - 1U) * latent) + index] ==
                gate_last_row[index]);
        REQUIRE(down[static_cast<std::size_t>((latent - 1U) * inner) + index] ==
                down_last_row[index]);
    }

    // Every decoded value is a table entry times a power of two, so the whole
    // module must stay finite and inside the representable magnitude.
    float magnitude = 0.0F;
    for (const auto value : gate) {
        REQUIRE(std::isfinite(value));
        magnitude = std::max(magnitude, std::fabs(value));
    }
    REQUIRE(magnitude == 0.125F);
}
