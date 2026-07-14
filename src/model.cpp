#include "strata/model.hpp"

#include <cmath>

namespace strata {

std::string_view to_string(Tier tier) noexcept {
    switch (tier) {
        case Tier::Vram: return "vram";
        case Tier::Ram: return "ram";
        case Tier::Peer: return "peer";
        case Tier::Nvme: return "nvme";
    }
    return "unknown";
}

std::string_view to_string(ReplacementPolicy policy) noexcept {
    switch (policy) {
        case ReplacementPolicy::Lru: return "lru";
        case ReplacementPolicy::Lfu: return "lfu";
        case ReplacementPolicy::Lease: return "lease";
    }
    return "unknown";
}

ValidationResult validate_model(const ModelSpec& spec) {
    ValidationResult result;
    if (spec.name.empty()) result.errors.emplace_back("model name is empty");
    if (!quantization_allowed(spec.quant_bits)) {
        result.errors.emplace_back("weight precision below four bits is forbidden");
    }
    if (spec.hidden_size == 0) result.errors.emplace_back("hidden_size must be positive");
    if (spec.layer_count == 0) result.errors.emplace_back("layer_count must be positive");
    if (spec.dense_prefix_layers > spec.layer_count) {
        result.errors.emplace_back("dense_prefix_layers exceeds layer_count");
    }
    if (!std::isfinite(spec.router.routed_scale) || spec.router.routed_scale <= 0.0F) {
        result.errors.emplace_back("routed_scale must be finite and positive");
    }

    if (spec.architecture == ArchitectureKind::Dense) {
        if (spec.router.kind != RouterKind::None || spec.router.routed_experts != 0 ||
            spec.router.experts_per_token != 0) {
            result.errors.emplace_back("dense architecture cannot declare routed experts");
        }
        if (spec.shared_experts != 0) {
            result.errors.emplace_back("dense architecture cannot declare shared experts");
        }
        return result;
    }

    if (spec.router.kind == RouterKind::None) {
        result.errors.emplace_back("MoE architecture requires an explicit router kind");
    }
    if (spec.router.routed_experts == 0) {
        result.errors.emplace_back("MoE architecture requires routed experts");
    }
    if (spec.router.experts_per_token == 0 ||
        spec.router.experts_per_token > spec.router.routed_experts) {
        result.errors.emplace_back("experts_per_token is outside the routed expert range");
    }
    if (spec.expert_intermediate_size == 0) {
        result.errors.emplace_back("MoE architecture requires expert_intermediate_size");
    }
    if (spec.router.groups == 0 || spec.router.selected_groups == 0 ||
        spec.router.selected_groups > spec.router.groups) {
        result.errors.emplace_back("invalid router group selection");
    }

    if (spec.architecture == ArchitectureKind::DeepSeek) {
        if (spec.attention != AttentionKind::Mla) {
            result.errors.emplace_back("DeepSeek adapter requires MLA attention");
        }
        if (spec.shared_experts == 0) {
            result.errors.emplace_back("DeepSeek adapter requires explicit shared experts");
        }
        if (spec.router.kind != RouterKind::NoAuxTc &&
            spec.router.kind != RouterKind::GroupLimitedSoftmax &&
            spec.router.kind != RouterKind::SigmoidTopK) {
            result.errors.emplace_back("unsupported DeepSeek router semantics");
        }
    }
    return result;
}

}  // namespace strata
