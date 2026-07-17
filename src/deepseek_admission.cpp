#include "strata/deepseek_admission.hpp"

#include <algorithm>
#include <limits>
#include <string_view>
#include <unordered_map>

namespace strata {

namespace {

[[nodiscard]] bool ends_with(std::string_view text,
                             std::string_view suffix) noexcept {
    return text.size() >= suffix.size() &&
           text.substr(text.size() - suffix.size()) == suffix;
}

[[nodiscard]] bool add(std::uint64_t& target, std::uint64_t value) noexcept {
    if (value > std::numeric_limits<std::uint64_t>::max() - target) return false;
    target += value;
    return true;
}

}  // namespace

Dsv4AdmissionResult plan_dsv4_resident_topology(
    const Dsv4IndexManifest& manifest, const Dsv4AdmissionConfig& config) {
    Dsv4AdmissionResult result;
    result.plan.maximum_context_tokens = config.maximum_context_tokens;
    result.plan.dspark_enabled = config.enable_dspark;
    if (manifest.resolved_tensors != manifest.tensors.size() ||
        manifest.validated_layouts != manifest.quantized_modules) {
        result.errors.emplace_back(
            "DeepSeek admission requires a fully resolved and validated checkpoint manifest");
        return result;
    }
    if (config.host_memory_ceiling_bytes == 0U) {
        result.errors.emplace_back("DeepSeek admission requires an explicit host-memory ceiling");
    }
    if (config.vram_weight_budgets.empty() ||
        std::any_of(config.vram_weight_budgets.begin(),
                    config.vram_weight_budgets.end(),
                    [](std::uint64_t bytes) { return bytes == 0U; })) {
        result.errors.emplace_back("DeepSeek admission requires positive per-device VRAM budgets");
    }
    const auto model_context =
        deepseek_v4_flash_dspark_spec().max_context_tokens;
    if (config.maximum_context_tokens == 0U ||
        config.maximum_context_tokens > model_context) {
        result.errors.emplace_back(
            "DeepSeek context must be within the model limit [1, " +
            std::to_string(model_context) + "] tokens");
    }
    if (!result.errors.empty()) return result;

    for (const auto budget : config.vram_weight_budgets) {
        if (!add(result.plan.total_vram_budget_bytes, budget)) {
            result.errors.emplace_back("aggregate DeepSeek VRAM budget overflows");
            return result;
        }
    }
    std::unordered_map<std::uint64_t, std::uint64_t> expert_placement_bytes;
    for (const auto& tensor : manifest.tensors) {
        if (tensor.dspark && !config.enable_dspark) continue;
        if (tensor.role == Dsv4TensorRole::RoutedExpert) {
            if (!add(result.plan.routed_expert_host_bytes, tensor.source_bytes)) {
                result.errors.emplace_back("DeepSeek routed expert byte count overflows");
                return result;
            }
            if (tensor.layer < 0 || tensor.expert < 0) {
                result.errors.emplace_back(
                    "DeepSeek routed tensor lacks an atomic placement identity");
                return result;
            }
            const auto placement =
                (static_cast<std::uint64_t>(tensor.layer) << 32U) |
                static_cast<std::uint32_t>(tensor.expert);
            if (!add(expert_placement_bytes[placement], tensor.source_bytes)) {
                result.errors.emplace_back(
                    "DeepSeek expert placement byte count overflows");
                return result;
            }
            continue;
        }

        const bool host_parameter = tensor.role == Dsv4TensorRole::Embedding ||
            tensor.role == Dsv4TensorRole::Mhc ||
            tensor.role == Dsv4TensorRole::Norm ||
            (tensor.role == Dsv4TensorRole::Router &&
             tensor.component != Dsv4TensorComponent::Weight) ||
            ends_with(tensor.name, ".attn_sink") || ends_with(tensor.name, ".ape") ||
            ends_with(tensor.name, ".compressor.norm.weight");
        if (host_parameter) {
            std::uint64_t bytes = tensor.source_bytes;
            if (tensor.source_dtype == SafetensorsDtype::Bf16 ||
                tensor.source_dtype == SafetensorsDtype::F16) {
                bytes *= 2U;
            } else if (tensor.source_dtype == SafetensorsDtype::I64) {
                bytes /= 2U;
            }
            if (!add(result.plan.host_parameter_bytes, bytes)) {
                result.errors.emplace_back("DeepSeek host parameter byte count overflows");
                return result;
            }
        } else if (tensor.component == Dsv4TensorComponent::Weight ||
                   tensor.component == Dsv4TensorComponent::Scale) {
            std::uint64_t bytes = tensor.source_bytes;
            // The target executor dequantizes wo_a to BF16 once and performs
            // its grouped einsum without FP8 activation quantization.
            if (ends_with(tensor.name, ".attn.wo_a.weight")) {
                if (bytes > std::numeric_limits<std::uint64_t>::max() / 2U) {
                    result.errors.emplace_back("DeepSeek wo_a byte count overflows");
                    return result;
                }
                bytes *= 2U;
            } else if (ends_with(tensor.name, ".attn.wo_a.scale")) {
                bytes = 0U;
            }
            if (!add(result.plan.resident_spine_vram_bytes, bytes)) {
                result.errors.emplace_back("DeepSeek resident spine byte count overflows");
                return result;
            }
        }
    }
    for (const auto& [placement, bytes] : expert_placement_bytes) {
        static_cast<void>(placement);
        result.plan.maximum_expert_bytes = std::max(
            result.plan.maximum_expert_bytes, bytes);
    }

    constexpr std::uint64_t layers = 43U;
    constexpr std::uint64_t head_dim = 512U;
    constexpr std::uint64_t window = 128U;
    constexpr std::uint64_t fp32 = 4U;
    result.plan.kv_state_bytes = layers * window * head_dim * fp32;
    const auto& deepseek = deepseek_v4_flash_dspark_spec().deepseek_v4;
    const auto& ratios = deepseek.compression_ratios;
    for (std::uint32_t layer = 0U; layer < 43U; ++layer) {
        const auto ratio = ratios[layer];
        if (ratio == 0U) continue;
        const auto compressed =
            (static_cast<std::uint64_t>(config.maximum_context_tokens) + ratio - 1U) / ratio;
        if (!add(result.plan.kv_state_bytes, compressed * head_dim * fp32)) {
            result.errors.emplace_back("DeepSeek compressed KV byte count overflows");
            return result;
        }
        const auto overlap = ratio == 4U ? 2U : 1U;
        const auto compressor_state = static_cast<std::uint64_t>(overlap) * ratio *
                                      overlap * head_dim * fp32 * 2U;
        if (!add(result.plan.kv_state_bytes, compressor_state)) {
            result.errors.emplace_back("DeepSeek compressor state byte count overflows");
            return result;
        }
        if (ratio == 4U &&
            config.maximum_context_tokens > deepseek.index_topk * ratio) {
            constexpr std::uint64_t index_head_dim = 128U;
            const auto index_cache = compressed * index_head_dim * fp32;
            constexpr std::uint64_t index_compressor_state =
                2U * 4U * 2U * index_head_dim * fp32 * 2U;
            if (!add(result.plan.index_state_bytes, index_cache) ||
                !add(result.plan.index_state_bytes, index_compressor_state)) {
                result.errors.emplace_back(
                    "DeepSeek sparse-index state byte count overflows");
                return result;
            }
        }
    }
    if (!add(result.plan.kv_state_bytes, result.plan.index_state_bytes)) {
        result.errors.emplace_back("DeepSeek total KV/index state byte count overflows");
        return result;
    }
    if (config.enable_dspark) {
        if (!add(result.plan.kv_state_bytes, 3U * window * head_dim * fp32)) {
            result.errors.emplace_back("DSpark KV state byte count overflows");
            return result;
        }
    }
    result.plan.host_workspace_bytes = 1ULL << 30U;
    result.plan.vram_workspace_bytes =
        static_cast<std::uint64_t>(config.vram_weight_budgets.size()) * (256ULL << 20U);
    result.plan.required_host_bytes = result.plan.routed_expert_host_bytes;
    if (!add(result.plan.required_host_bytes, result.plan.host_parameter_bytes) ||
        !add(result.plan.required_host_bytes, result.plan.kv_state_bytes) ||
        !add(result.plan.required_host_bytes, result.plan.host_workspace_bytes)) {
        result.errors.emplace_back("DeepSeek total host-memory admission overflows");
        return result;
    }
    if (result.plan.required_host_bytes > config.host_memory_ceiling_bytes) {
        result.errors.emplace_back(
            "DeepSeek zero-read resident set exceeds the host-memory ceiling");
    }
    const auto required_vram = result.plan.resident_spine_vram_bytes +
                               result.plan.vram_workspace_bytes;
    if (required_vram > result.plan.total_vram_budget_bytes) {
        result.errors.emplace_back(
            "DeepSeek resident spine and workspaces exceed aggregate VRAM budget");
    } else {
        result.plan.expert_vram_cache_bytes =
            result.plan.total_vram_budget_bytes - required_vram;
        if (result.plan.expert_vram_cache_bytes < result.plan.maximum_expert_bytes) {
            result.errors.emplace_back(
                "DeepSeek VRAM budget cannot hold one routed expert projection triplet");
        }
    }
    result.plan.zero_nvme_decode = result.errors.empty() &&
                                   config.require_zero_nvme_decode;
    result.plan.steady_state_nvme_bytes = result.plan.zero_nvme_decode ? 0U :
        result.plan.routed_expert_host_bytes;
    return result;
}

}  // namespace strata
