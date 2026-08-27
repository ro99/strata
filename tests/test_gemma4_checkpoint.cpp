#include "test.hpp"

#include "strata/models/gemma4/gemma4_checkpoint.hpp"
#include "strata/models/common/model_adapter.hpp"

#include <filesystem>

TEST_CASE("real Gemma 4 31B checkpoint validates either pinned weight format") {
    const auto path = std::filesystem::path(STRATA_SOURCE_DIR) / "models/gemma4";
    if (!std::filesystem::exists(path / "model.safetensors.index.json") &&
        !std::filesystem::exists(path / "model.safetensors")) {
        SKIP("pinned Gemma 4 checkpoint is absent");
    }
    const auto checkpoint = strata::Gemma4CheckpointReader::open(path.string());
    REQUIRE(checkpoint.ok());
    REQUIRE((checkpoint.value->tensors().size() == 2008U ||
             checkpoint.value->tensors().size() == 1598U));
    const auto* global_q = checkpoint.value->find(
        "model.language_model.layers.5.self_attn.q_proj.weight_packed");
    REQUIRE(global_q != nullptr);
    if (global_q->dtype == strata::SafetensorsDtype::I32) {
        REQUIRE(global_q->shape == std::vector<std::uint64_t>({16384U, 1344U}));
    } else {
        REQUIRE(global_q->dtype == strata::SafetensorsDtype::U8);
        REQUIRE(global_q->shape == std::vector<std::uint64_t>({16384U, 2688U}));
    }
    REQUIRE(checkpoint.value->find(
        "model.language_model.layers.5.self_attn.v_proj.weight_packed") == nullptr);
}

TEST_CASE("Gemma 4 CUDA descriptors preserve W8A16 and map MXFP4 exactly") {
    strata::Gemma4Tensor w8_packed;
    w8_packed.dtype = strata::SafetensorsDtype::I32;
    w8_packed.shape = {16U, 16U};
    strata::Gemma4Tensor w8_scales;
    w8_scales.dtype = strata::SafetensorsDtype::Bf16;
    w8_scales.shape = {16U, 2U};
    const auto w8 = strata::describe_gemma4_cuda_linear(
        w8_packed, w8_scales, 16U, 64U);
    REQUIRE(w8.ok());
    REQUIRE(w8.value.encoding == strata::CudaWeightEncoding::OffsetPackedInt8);
    REQUIRE(w8.value.packed_columns == 16U);
    REQUIRE(w8.value.scale_columns == 2U);
    REQUIRE(w8.value.group_size == 32U);

    strata::Gemma4Tensor fp4_packed;
    fp4_packed.dtype = strata::SafetensorsDtype::U8;
    fp4_packed.shape = {16U, 32U};
    strata::Gemma4Tensor fp4_scales;
    fp4_scales.dtype = strata::SafetensorsDtype::U8;
    fp4_scales.shape = {16U, 2U};
    const auto fp4 = strata::describe_gemma4_cuda_linear(
        fp4_packed, fp4_scales, 16U, 64U);
    REQUIRE(fp4.ok());
    REQUIRE(fp4.value.encoding == strata::CudaWeightEncoding::Fp4E2m1Group32);
    REQUIRE(fp4.value.packed_columns == 32U);
    REQUIRE(fp4.value.scale_columns == 2U);
    REQUIRE(fp4.value.group_size == 32U);
}
