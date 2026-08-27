#include "test.hpp"

#include "strata/models/laguna/laguna_checkpoint.hpp"
#include "strata/models/laguna/laguna_ops.hpp"
#include "strata/models/common/model.hpp"
#include "strata/models/common/model_adapter.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <vector>

TEST_CASE("real Laguna S 2.1-NVFP4 checkpoint validates target tensors") {
    const auto path =
        std::filesystem::path(STRATA_SOURCE_DIR) / "models/laguna-s-21";
    if (!std::filesystem::exists(path / "model.safetensors.index.json")) {
        SKIP("pinned Laguna S 2.1-NVFP4 checkpoint is absent");
    }
    const auto& contract = strata::kLagunaExecutionContract;
    const auto spec = strata::laguna_s21_nvfp4_spec();
    const auto checkpoint =
        strata::LagunaCheckpointReader::open(path.string());
    REQUIRE(checkpoint.ok());
    REQUIRE(checkpoint.value->tensors().size() == spec.source.tensor_count);
    REQUIRE(checkpoint.value->shard_file_bytes() == spec.source.shard_file_bytes);

    // Layer 1 attention is a sliding layer: 72 query heads, 8 KV heads, and a
    // per-head gate projection.
    const auto sliding_query = checkpoint.value->linear(
        "model.layers.1.self_attn.q_proj", 72U * contract.head_dim,
        contract.hidden_size);
    REQUIRE(sliding_query.ok());
    REQUIRE(sliding_query.value.encoding ==
            strata::LagunaTensorEncoding::Plain);
    const auto gate = checkpoint.value->linear(
        "model.layers.1.self_attn.g_proj", 72U, contract.hidden_size);
    REQUIRE(gate.ok());
    // Layer 4 is global attention with 48 heads, so the sliding shape must fail.
    REQUIRE(!checkpoint.value->linear("model.layers.4.self_attn.q_proj",
                                      72U * contract.head_dim,
                                      contract.hidden_size).ok());
    REQUIRE(checkpoint.value->linear("model.layers.4.self_attn.q_proj",
                                     48U * contract.head_dim,
                                     contract.hidden_size).ok());

    // Layer 0 is dense; it has no router and no experts.
    REQUIRE(checkpoint.value->linear("model.layers.0.mlp.gate_proj",
                                     contract.dense_intermediate_size,
                                     contract.hidden_size).ok());
    REQUIRE(checkpoint.value->find("model.layers.0.mlp.gate.weight") == nullptr);

    // Routed experts below layer 40 are NVFP4 and from 40 on are plain BF16.
    const auto quantized = checkpoint.value->linear(
        "model.layers.39.mlp.experts.7.down_proj", contract.hidden_size,
        contract.expert_intermediate_size);
    REQUIRE(quantized.ok());
    REQUIRE(quantized.value.encoding ==
            strata::LagunaTensorEncoding::Nvfp4Group16);
    REQUIRE(quantized.value.packed->dtype == strata::SafetensorsDtype::U8);
    REQUIRE(quantized.value.packed->shape ==
            std::vector<std::uint64_t>(
                {contract.hidden_size, contract.expert_intermediate_size / 2U}));
    REQUIRE(quantized.value.scale->dtype == strata::SafetensorsDtype::F8E4M3);
    REQUIRE(quantized.value.scale->shape ==
            std::vector<std::uint64_t>(
                {contract.hidden_size,
                 contract.expert_intermediate_size / contract.nvfp4_group_size}));

    const auto plain = checkpoint.value->linear(
        "model.layers.40.mlp.experts.7.down_proj", contract.hidden_size,
        contract.expert_intermediate_size);
    REQUIRE(plain.ok());
    REQUIRE(plain.value.encoding == strata::LagunaTensorEncoding::Plain);
    REQUIRE(plain.value.weight->dtype == strata::SafetensorsDtype::Bf16);

    // The NVFP4 host view must present a usable positive global scale.
    const auto view = checkpoint.value->nvfp4_view(quantized.value);
    REQUIRE(view.ok());
    REQUIRE(view.value.global_scale > 0.0F);
    REQUIRE(std::isfinite(view.value.global_scale));
    REQUIRE(view.value.packed.size() ==
            view.value.rows * view.value.packed_columns);
    REQUIRE(view.value.scales.size() ==
            view.value.rows * view.value.scale_columns);
    checkpoint.value->release_mapped_views();
}

TEST_CASE("Laguna NVFP4 dequantization matches the target-format oracle") {
    const auto path =
        std::filesystem::path(STRATA_SOURCE_DIR) / "models/laguna-s-21";
    if (!std::filesystem::exists(path / "model.safetensors.index.json")) {
        SKIP("pinned Laguna S 2.1-NVFP4 checkpoint is absent");
    }
    const auto& contract = strata::kLagunaExecutionContract;
    const auto checkpoint = strata::LagunaCheckpointReader::open(path.string());
    REQUIRE(checkpoint.ok());
    const auto module = checkpoint.value->linear(
        "model.layers.1.mlp.experts.3.gate_proj",
        contract.expert_intermediate_size, contract.hidden_size);
    REQUIRE(module.ok());
    const auto view = checkpoint.value->nvfp4_view(module.value);
    REQUIRE(view.ok());
    // Independently reproduced by decoding the same tensor with NumPy against
    // the compressed-tensors rule w = e2m1 * (e4m3_scale / global_scale) and
    // multiplying in float64. Values below are that oracle's output.
    REQUIRE(std::fabs(view.value.global_scale - 11613.97461F) < 1.0e-2F);

    std::vector<float> input(contract.hidden_size);
    std::uint32_t state = 12345U;
    for (auto& value : input) {
        state = state * 1664525U + 1013904223U;
        value = static_cast<float>(
            static_cast<int>(state >> 16U) % 2001 - 1000) / 1000.0F;
    }
    std::vector<float> output(contract.expert_intermediate_size);
    REQUIRE(strata::laguna_nvfp4_matvec_reference(view.value, input, output).ok());
    const std::array<float, 4> oracle{0.86184775F, -0.471345946F, 0.788716189F,
                                      0.730643516F};
    for (std::size_t index = 0U; index < oracle.size(); ++index) {
        REQUIRE(std::fabs(output[index] - oracle[index]) < 1.0e-4F);
    }
    checkpoint.value->release_mapped_views();
}

TEST_CASE("real Laguna S 2.1-MXFP4 checkpoint validates without replacing NVFP4") {
    const auto path =
        std::filesystem::path(STRATA_SOURCE_DIR) / "models/laguna";
    if (!std::filesystem::exists(path / "model.safetensors.index.json")) {
        SKIP("pinned Laguna S 2.1-MXFP4 checkpoint is absent");
    }
    const auto& contract = strata::kLagunaExecutionContract;
    const auto spec = strata::laguna_s21_mxfp4_spec();
    REQUIRE(strata::validate_laguna_s21_mxfp4(spec).ok());
    const auto checkpoint = strata::LagunaCheckpointReader::open(path.string());
    REQUIRE(checkpoint.ok());
    REQUIRE(checkpoint.value->format() ==
            strata::LagunaCheckpointFormat::Mxfp4Group32);
    REQUIRE(checkpoint.value->tensors().size() == spec.source.tensor_count);
    REQUIRE(checkpoint.value->shard_file_bytes() == spec.source.shard_file_bytes);

    const auto early = checkpoint.value->linear(
        "model.layers.1.mlp.experts.7.gate_proj",
        contract.expert_intermediate_size, contract.hidden_size);
    REQUIRE(early.ok());
    REQUIRE(early.value.encoding == strata::LagunaTensorEncoding::Mxfp4Group32);
    REQUIRE(early.value.packed->dtype == strata::SafetensorsDtype::U8);
    REQUIRE(early.value.packed->shape ==
            std::vector<std::uint64_t>(
                {contract.expert_intermediate_size,
                 contract.hidden_size / 2U}));
    REQUIRE(early.value.scale->dtype == strata::SafetensorsDtype::U8);
    REQUIRE(early.value.scale->shape ==
            std::vector<std::uint64_t>(
                {contract.expert_intermediate_size,
                 contract.hidden_size / 32U}));
    REQUIRE(early.value.global_scale == nullptr);

    // The old checkpoint switches layers 40-47 back to BF16. MXFP4 does not.
    const auto late = checkpoint.value->linear(
        "model.layers.47.mlp.experts.7.down_proj", contract.hidden_size,
        contract.expert_intermediate_size);
    REQUIRE(late.ok());
    REQUIRE(late.value.encoding == strata::LagunaTensorEncoding::Mxfp4Group32);
    REQUIRE(late.value.scale->shape ==
            std::vector<std::uint64_t>(
                {contract.hidden_size,
                 contract.expert_intermediate_size / 32U}));

    const auto attention = checkpoint.value->linear(
        "model.layers.1.self_attn.q_proj", 72U * contract.head_dim,
        contract.hidden_size);
    REQUIRE(attention.ok());
    REQUIRE(attention.value.encoding == strata::LagunaTensorEncoding::Plain);
}
