#pragma once

#include "strata/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata {

enum class ArchitectureKind : std::uint8_t {
    Dense,
    StandardMoe,
    DeepSeek,
};

enum class AttentionKind : std::uint8_t {
    Mha,
    Gqa,
    Mqa,
    Mla,
};

enum class RouterKind : std::uint8_t {
    None,
    SoftmaxTopK,
    SigmoidTopK,
    GroupLimitedSoftmax,
    NoAuxTc,
};

struct RouterSpec {
    RouterKind kind{RouterKind::None};
    std::uint32_t routed_experts{};
    std::uint32_t experts_per_token{};
    std::uint32_t groups{1};
    std::uint32_t selected_groups{1};
    bool normalize_topk{true};
    bool selection_bias{false};
    float routed_scale{1.0F};
};

struct ModelSpec {
    std::string name;
    ArchitectureKind architecture{ArchitectureKind::Dense};
    AttentionKind attention{AttentionKind::Mha};
    RouterSpec router;
    std::uint32_t quant_bits{kMinimumQuantBits};
    std::uint32_t hidden_size{};
    std::uint32_t layer_count{};
    std::uint32_t dense_prefix_layers{};
    std::uint32_t shared_experts{};
    std::uint32_t expert_intermediate_size{};
};

struct ValidationResult {
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

[[nodiscard]] ValidationResult validate_model(const ModelSpec& spec);

}  // namespace strata
