#include "test.hpp"

#include "strata/inkling_checkpoint.hpp"
#include "strata/inkling_ops.hpp"
#include "strata/model.hpp"
#include "strata/model_adapter.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace {

std::filesystem::path inkling_model_path() {
    return std::filesystem::path(STRATA_SOURCE_DIR) / "models/inkling-s";
}

}  // namespace

TEST_CASE("Inkling-Small spec validates against itself") {
    const auto spec = strata::inkling_small_nvfp4_spec();
    const auto result = strata::validate_inkling_small_nvfp4(spec);
    for (const auto& error : result.errors) {
        std::cerr << "unexpected validation error: " << error << '\n';
    }
    REQUIRE(result.ok());
}

TEST_CASE("Inkling-Small validation rejects silent contract drift") {
    const auto require_rejected = [](strata::ModelSpec spec) {
        REQUIRE(!strata::validate_inkling_small_nvfp4(spec).ok());
    };
    // Every one of these is a different model, not a configuration knob.
    auto spec = strata::inkling_small_nvfp4_spec();
    spec.router.experts_per_token = 8U;
    require_rejected(spec);

    spec = strata::inkling_small_nvfp4_spec();
    spec.router.routed_scale = 1.0F;
    require_rejected(spec);

    spec = strata::inkling_small_nvfp4_spec();
    spec.inkling.shared_expert_sink = false;
    require_rejected(spec);

    spec = strata::inkling_small_nvfp4_spec();
    spec.inkling.log_sigmoid_renormalization = false;
    require_rejected(spec);

    spec = strata::inkling_small_nvfp4_spec();
    spec.inkling.interleaved_gate_up = false;
    require_rejected(spec);

    spec = strata::inkling_small_nvfp4_spec();
    spec.inkling.short_conv_streams = false;
    require_rejected(spec);

    // Position lives only in the relative branch, so zeroing it is fatal.
    spec = strata::inkling_small_nvfp4_spec();
    spec.inkling.global_relative_extent = 0U;
    require_rejected(spec);

    // The local extent must span exactly the window it masks.
    spec = strata::inkling_small_nvfp4_spec();
    spec.inkling.local_relative_extent = 256U;
    require_rejected(spec);

    // 1/sqrt(head_dim) is the usual scale and the wrong one here.
    spec = strata::inkling_small_nvfp4_spec();
    spec.inkling.attention_scale = 1.0F / std::sqrt(128.0F);
    require_rejected(spec);

    spec = strata::inkling_small_nvfp4_spec();
    spec.quant_bits = 8U;
    require_rejected(spec);

    spec = strata::inkling_small_nvfp4_spec();
    spec.mixed_quantization.routed_experts.group_size = 32U;
    require_rejected(spec);
}

TEST_CASE("real Inkling-Small checkpoint validates every declared module") {
    const auto path = inkling_model_path();
    if (!std::filesystem::exists(path / "model.safetensors.index.json")) {
        SKIP("pinned Inkling-Small-NVFP4 checkpoint is absent");
    }
    const auto& contract = strata::kInklingExecutionContract;
    const auto spec = strata::inkling_small_nvfp4_spec();
    const auto checkpoint = strata::InklingCheckpointReader::open(path.string());
    for (const auto& error : checkpoint.errors) {
        std::cerr << "checkpoint error: " << error << '\n';
    }
    REQUIRE(checkpoint.ok());
    REQUIRE(checkpoint.value->tensors().size() == spec.source.tensor_count);
    REQUIRE(checkpoint.value->shard_file_bytes() == spec.source.shard_file_bytes);

    const auto hidden = static_cast<std::uint64_t>(contract.hidden_size);
    const auto& reader = *checkpoint.value;

    // The attention block has four projections, not three: q, k, v and the
    // relative branch r.
    REQUIRE(reader.linear("model.llm.layers.0.attn.wq_du.weight",
                          contract.attention_heads * contract.head_dim, hidden)
                .ok());
    REQUIRE(reader.linear("model.llm.layers.0.attn.wk_dv.weight",
                          contract.key_value_heads * contract.head_dim, hidden)
                .ok());
    REQUIRE(reader.linear("model.llm.layers.0.attn.wr_du.weight",
                          contract.attention_heads * contract.relative_dim,
                          hidden)
                .ok());

    // Layer 5 is global and layer 4 is local, so their relative projections
    // must differ in extent. Getting this backwards is silent at load time and
    // wrong at every position, so it is pinned here.
    REQUIRE(reader.linear("model.llm.layers.5.attn.rel_logits_proj.proj",
                          contract.relative_dim, contract.global_relative_extent)
                .ok());
    REQUIRE(!reader.linear("model.llm.layers.5.attn.rel_logits_proj.proj",
                           contract.relative_dim, contract.local_relative_extent)
                 .ok());
    REQUIRE(reader.linear("model.llm.layers.4.attn.rel_logits_proj.proj",
                          contract.relative_dim, contract.local_relative_extent)
                .ok());
    REQUIRE(!reader.linear("model.llm.layers.4.attn.rel_logits_proj.proj",
                           contract.relative_dim, contract.global_relative_extent)
                 .ok());

    // There is no rotary tensor anywhere; position is entirely relative.
    REQUIRE(reader.find("model.llm.layers.0.attn.rotary_emb.inv_freq") == nullptr);

    // Layers 0 and 1 are dense; they have no gate and no experts.
    REQUIRE(reader.linear("model.llm.layers.0.mlp.w13_dn.weight",
                          2U * contract.dense_intermediate_size, hidden)
                .ok());
    REQUIRE(reader.find("model.llm.layers.0.mlp.gate.weight") == nullptr);
    REQUIRE(reader.find("model.llm.layers.1.mlp.experts.w13_weight") == nullptr);

    // The gate scores routed experts and both sinks; the correction bias
    // covers only the routed range.
    const auto gate = reader.linear("model.llm.layers.2.mlp.gate.weight",
                                    contract.routed_experts +
                                        contract.shared_experts,
                                    hidden);
    REQUIRE(gate.ok());
    const auto* bias = reader.find("model.llm.layers.2.mlp.gate.bias");
    REQUIRE(bias != nullptr);
    REQUIRE(bias->shape ==
            std::vector<std::uint64_t>{contract.routed_experts});

    // Layer 2's experts are plain BF16; layer 3's are NVFP4.
    const auto plain = reader.expert_stack(
        "model.llm.layers.2.mlp.experts.w13_weight", 2U, contract.routed_experts,
        2U * contract.expert_intermediate_size, hidden);
    REQUIRE(plain.ok());
    REQUIRE(plain.value.encoding == strata::InklingTensorEncoding::Plain);

    const auto quantized = reader.expert_stack(
        "model.llm.layers.3.mlp.experts.w13_weight", 3U, contract.routed_experts,
        2U * contract.expert_intermediate_size, hidden);
    REQUIRE(quantized.ok());
    REQUIRE(quantized.value.encoding ==
            strata::InklingTensorEncoding::Nvfp4Group16);
    REQUIRE(quantized.value.packed->dtype == strata::SafetensorsDtype::U8);
    REQUIRE(quantized.value.scale->dtype == strata::SafetensorsDtype::F8E4M3);
    REQUIRE(quantized.value.global_scale->dtype == strata::SafetensorsDtype::F32);
    // One FP32 multiplier per expert, not one per tensor.
    REQUIRE(quantized.value.global_scale->shape ==
            std::vector<std::uint64_t>{contract.routed_experts});

    // The last layer must also be quantized; an off-by-one on the start layer
    // would leave it BF16 and silently change memory footprint.
    REQUIRE(reader
                .expert_stack("model.llm.layers.41.mlp.experts.w2_weight", 41U,
                              contract.routed_experts, hidden,
                              contract.expert_intermediate_size)
                .ok());

    // All eight MTP depths are present with their own attention and dense MLP.
    for (std::uint32_t depth = 0U; depth < contract.mtp_layers; ++depth) {
        const auto prefix = strata::inkling_mtp_prefix(depth);
        REQUIRE(reader.linear(prefix + "input_proj.weight", hidden, 2U * hidden)
                    .ok());
        REQUIRE(reader
                    .linear(prefix + "transformer_block.attn.rel_logits_proj.proj",
                            contract.relative_dim,
                            strata::inkling_mtp_relative_extent(depth))
                    .ok());
    }
}

TEST_CASE("real Inkling-Small NVFP4 experts dequantize to the BF16 scale") {
    const auto path = inkling_model_path();
    if (!std::filesystem::exists(path / "model.safetensors.index.json")) {
        SKIP("pinned Inkling-Small-NVFP4 checkpoint is absent");
    }
    const auto& contract = strata::kInklingExecutionContract;
    const auto checkpoint = strata::InklingCheckpointReader::open(path.string());
    REQUIRE(checkpoint.ok());
    const auto& reader = *checkpoint.value;
    const auto hidden = static_cast<std::uint64_t>(contract.hidden_size);

    const auto stack = reader.expert_stack(
        "model.llm.layers.20.mlp.experts.w13_weight", 20U,
        contract.routed_experts, 2U * contract.expert_intermediate_size, hidden);
    REQUIRE(stack.ok());
    const auto matrix = reader.nvfp4_expert_view(stack.value, 0U);
    REQUIRE(matrix.ok());
    REQUIRE(matrix.value.columns == hidden);
    REQUIRE(matrix.value.group_size == contract.nvfp4_group_size);
    REQUIRE(matrix.value.global_scale > 0.0F);

    // ModelOpt stores scale2 as amax / (6 * 448), so it is a small multiplier.
    // The compressed-tensors reciprocal convention would put it near 10^3 and
    // inflate every dequantized weight by roughly seven orders of magnitude.
    REQUIRE(matrix.value.global_scale < 1.0F);

    // Dequantize one row via a basis vector per column group and compare the
    // magnitude against the unquantized BF16 experts of layer 2. If the scale
    // direction were inverted this ratio would be astronomically off.
    std::vector<float> input(static_cast<std::size_t>(hidden), 0.0F);
    std::vector<float> output(1U, 0.0F);
    strata::InklingNvfp4MatrixView row = matrix.value;
    row.rows = 1U;
    row.packed = row.packed.subspan(0U, static_cast<std::size_t>(hidden / 2U));
    row.scales = row.scales.subspan(
        0U, static_cast<std::size_t>(hidden / contract.nvfp4_group_size));

    double squared = 0.0;
    for (std::uint64_t column = 0U; column < hidden; ++column) {
        input[static_cast<std::size_t>(column)] = 1.0F;
        auto status = strata::inkling_nvfp4_matvec_reference(row, input, output);
        REQUIRE(status.ok());
        squared += static_cast<double>(output[0]) * output[0];
        input[static_cast<std::size_t>(column)] = 0.0F;
    }
    const double rms = std::sqrt(squared / static_cast<double>(hidden));
    // Trained BF16 weights of this shape sit well inside this band; the
    // inverted convention lands near 10^5.
    REQUIRE(rms > 1.0e-3);
    REQUIRE(rms < 1.0);
}
