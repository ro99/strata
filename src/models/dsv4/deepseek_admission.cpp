#include "strata/deepseek_admission.hpp"

#include "strata/deepseek_kv_cache.hpp"
#include "strata/model_adapter.hpp"

#include <algorithm>
#include <limits>
#include <string_view>
#include <unordered_map>

namespace strata {

Dsv4E8m0Admission dsv4_admit_e8m0_scales(
    std::span<const std::byte> scales) noexcept {
    Dsv4E8m0Admission admission;
    for (std::size_t index = 0U; index < scales.size(); ++index) {
        const auto code = static_cast<std::uint8_t>(scales[index]);
        if (code >= kDsv4E8m0AdmissibleMinimum &&
            code <= kDsv4E8m0AdmissibleMaximum) {
            continue;
        }
        if (admission.inadmissible == 0U) {
            admission.first_offset = index;
            admission.first_code = code;
        }
        ++admission.inadmissible;
        if (code == 0U) {
            ++admission.code_zero;
        } else {
            ++admission.code_255;
        }
    }
    return admission;
}

ValidationResult dsv4_admit_e8m0_scales_for(
    std::string_view tensor_name, std::span<const std::byte> scales) {
    ValidationResult result;
    const auto admission = dsv4_admit_e8m0_scales(scales);
    if (admission.admitted()) return result;
    result.errors.emplace_back(
        std::string(tensor_name) + " carries " +
        std::to_string(admission.inadmissible) +
        " inadmissible E8M0 scale codes (first: code " +
        std::to_string(static_cast<unsigned>(admission.first_code)) +
        " at byte offset " + std::to_string(admission.first_offset) +
        "); exact mode reports failure rather than substituting");
    return result;
}


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

[[nodiscard]] bool multiply(std::uint64_t first, std::uint64_t second,
                            std::uint64_t& output) noexcept {
    if (first != 0U && second >
            std::numeric_limits<std::uint64_t>::max() / first) {
        return false;
    }
    output = first * second;
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
    if (!config.device_kv_cache_bytes.empty() &&
        config.device_kv_cache_bytes.size() != config.vram_weight_budgets.size()) {
        result.errors.emplace_back(
            "DeepSeek KV and VRAM device budget counts must match");
    }
    if (config.physical_kv_cache && !config.compact_kv_cache) {
        result.errors.emplace_back(
            "DeepSeek physical KV admission requires the block cache");
    }
    if (config.device_resident_mhc && !config.physical_kv_cache) {
        result.errors.emplace_back(
            "DeepSeek device-resident mHC requires physical KV admission");
    }
    const auto model_context =
        deepseek_v4_flash_0731_spec().max_context_tokens;
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
            // The output-tiled decode layout duplicates each group-32 scale
            // for its two group-16 inner loops. It replaces the canonical
            // routed tensor arena, so only the scale expansion is additional.
            if (config.host_routed_experts &&
                tensor.component == Dsv4TensorComponent::Scale &&
                !add(result.plan.routed_expert_host_bytes,
                     tensor.source_bytes)) {
                result.errors.emplace_back(
                    "DeepSeek tiled expert scale byte count overflows");
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

    constexpr std::uint64_t layers = kDeepSeekV4ExecutionContract.layer_count;
    constexpr std::uint64_t head_dim = kDeepSeekV4ExecutionContract.head_dim;
    constexpr std::uint64_t window = kDeepSeekV4ExecutionContract.sliding_window;
    constexpr std::uint64_t fp32 = sizeof(float);
    const auto add_compact_blocks = [&](Dsv4KvBlockKind kind,
                                        std::uint32_t ratio,
                                        std::uint64_t blocks,
                                        std::uint64_t& target) {
        const auto format = dsv4_kv_format(
            kind, false, config.physical_kv_cache);
        const auto capacity_rows = config.physical_kv_cache
            ? dsv4_kv_block_rows(kind, ratio, true) : kDsv4KvBlockRows;
        const auto row_bytes = dsv4_kv_row_bytes(kind, format);
        const auto block_bytes = dsv4_kv_block_bytes(
            kind, format, capacity_rows);
        std::uint64_t payload = 0U;
        std::uint64_t metadata = 0U;
        std::uint64_t physical = 0U;
        if (block_bytes == 0U ||
            capacity_rows == 0U ||
            !multiply(row_bytes * capacity_rows, blocks, payload) ||
            !multiply(kDsv4KvBlockHeaderBytes, blocks, metadata) ||
            !multiply(block_bytes, blocks, physical) ||
            physical < payload || physical - payload < metadata ||
            !add(result.plan.kv_cache_payload_bytes, payload) ||
            !add(result.plan.kv_cache_metadata_bytes, metadata) ||
            !add(result.plan.kv_cache_alignment_bytes,
                 physical - payload - metadata) ||
            !add(target, physical)) {
            return false;
        }
        return true;
    };
    if (config.compact_kv_cache) {
        const auto sliding_capacity = config.physical_kv_cache
            ? dsv4_kv_block_rows(Dsv4KvBlockKind::Sliding, 1U, true)
            : kDsv4KvBlockRows;
        // A page of N rows retains [base + 1 - window, base + N - 1] for the
        // whole page, so the live span is N + window - 1 rows before block
        // rounding -- not window alone.
        const auto page_rows = std::max<std::uint64_t>(
            1U, config.prefill_page_tokens);
        const auto maximum_sliding_rows = std::min<std::uint64_t>(
            config.maximum_context_tokens,
            window + page_rows + sliding_capacity - 2U);
        const auto sliding_blocks =
            (maximum_sliding_rows + sliding_capacity - 1U) /
            sliding_capacity;
        if (!add_compact_blocks(Dsv4KvBlockKind::Sliding,
                                1U,
                                layers * sliding_blocks,
                                result.plan.kv_state_bytes)) {
            result.errors.emplace_back(
                "DeepSeek compact sliding KV byte count overflows");
            return result;
        }
    } else {
        result.plan.kv_state_bytes = layers * window * head_dim * fp32;
    }
    const auto& deepseek = deepseek_v4_flash_0731_spec().deepseek_v4;
    const auto& ratios = deepseek.compression_ratios;
    for (std::uint32_t layer = 0U;
         layer < kDeepSeekV4ExecutionContract.layer_count; ++layer) {
        const auto ratio = ratios[layer];
        if (ratio == 0U) continue;
        const auto compressed =
            (static_cast<std::uint64_t>(config.maximum_context_tokens) + ratio - 1U) / ratio;
        const auto kind = ratio == 4U ? Dsv4KvBlockKind::Csa
                                     : Dsv4KvBlockKind::Hca;
        const auto capacity_rows = config.physical_kv_cache
            ? dsv4_kv_block_rows(kind, ratio, true) : kDsv4KvBlockRows;
        const auto blocks = (compressed + capacity_rows - 1U) /
                            capacity_rows;
        if (config.compact_kv_cache
                ? !add_compact_blocks(kind, ratio, blocks,
                                      result.plan.kv_state_bytes)
                : !add(result.plan.kv_state_bytes,
                       compressed * head_dim * fp32)) {
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
            constexpr std::uint64_t index_head_dim =
                kDeepSeekV4ExecutionContract.index_head_dim;
            constexpr std::uint64_t index_compressor_state =
                2U * 4U * 2U * index_head_dim * fp32 * 2U;
            const bool index_cache_ok = config.compact_kv_cache
                ? add_compact_blocks(Dsv4KvBlockKind::LearnedIndex, ratio,
                                     blocks,
                                     result.plan.index_state_bytes)
                : add(result.plan.index_state_bytes,
                      compressed * index_head_dim * fp32);
            if (!index_cache_ok ||
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
    auto minimum_host_kv_cache_bytes = result.plan.kv_state_bytes;
    if (config.compact_kv_cache) {
        minimum_host_kv_cache_bytes = result.plan.kv_cache_payload_bytes;
        if (!add(minimum_host_kv_cache_bytes,
                 result.plan.kv_cache_metadata_bytes) ||
            !add(minimum_host_kv_cache_bytes,
                 result.plan.kv_cache_alignment_bytes)) {
            result.errors.emplace_back(
                "DeepSeek compact KV cache byte count overflows");
            return result;
        }
    }
    result.plan.host_kv_cache_bytes = config.host_kv_cache_bytes == 0U
        ? minimum_host_kv_cache_bytes : config.host_kv_cache_bytes;
    if (result.plan.host_kv_cache_bytes < minimum_host_kv_cache_bytes) {
        result.errors.emplace_back(
            "DeepSeek exact KV state exceeds the host KV cache budget");
    }
    result.plan.per_device_kv_cache_bytes = config.device_kv_cache_bytes;
    if (result.plan.per_device_kv_cache_bytes.empty()) {
        result.plan.per_device_kv_cache_bytes.resize(
            config.vram_weight_budgets.size());
    }
    for (std::size_t slot = 0U;
         slot < result.plan.per_device_kv_cache_bytes.size(); ++slot) {
        const auto bytes = result.plan.per_device_kv_cache_bytes[slot];
        constexpr std::uint64_t workspace = 256ULL << 20U;
        if (config.vram_weight_budgets[slot] <= workspace ||
            bytes >= config.vram_weight_budgets[slot] - workspace ||
            !add(result.plan.device_kv_cache_bytes, bytes)) {
            result.errors.emplace_back(
                "DeepSeek device KV cache budget exceeds admitted VRAM");
            return result;
        }
    }
    result.plan.host_workspace_bytes = 1ULL << 30U;
    if (config.enable_mhc_prepack) {
        constexpr std::uint64_t mhc_prepack_bytes =
            2U * layers * kDeepSeekV4ExecutionContract.mix_width *
            kDeepSeekV4ExecutionContract.mhc_multiplier *
            kDeepSeekV4ExecutionContract.hidden_size * fp32;
        result.plan.mhc_prepack_bytes = mhc_prepack_bytes;
        if (!add(result.plan.host_workspace_bytes, mhc_prepack_bytes)) {
            result.errors.emplace_back(
                "DeepSeek mHC prepack byte count overflows");
            return result;
        }
    }
    if (config.device_resident_mhc) {
        constexpr std::uint64_t mhc_projection_bytes =
            2U * layers * kDeepSeekV4ExecutionContract.mix_width *
            kDeepSeekV4ExecutionContract.mhc_multiplier *
            kDeepSeekV4ExecutionContract.hidden_size * fp32;
        constexpr std::uint64_t mhc_auxiliary_bytes =
            2U * layers *
            (112U + kDeepSeekV4ExecutionContract.hidden_size *
                        sizeof(std::uint16_t));
        result.plan.mhc_device_bytes =
            mhc_projection_bytes + mhc_auxiliary_bytes;
        if (!add(result.plan.resident_spine_vram_bytes,
                 result.plan.mhc_device_bytes)) {
            result.errors.emplace_back(
                "DeepSeek device mHC device byte count overflows");
            return result;
        }
    }
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
    if (result.plan.resident_spine_vram_bytes >
            std::numeric_limits<std::uint64_t>::max() -
                result.plan.vram_workspace_bytes ||
        result.plan.resident_spine_vram_bytes +
                result.plan.vram_workspace_bytes >
            std::numeric_limits<std::uint64_t>::max() -
                result.plan.device_kv_cache_bytes) {
        result.errors.emplace_back("DeepSeek required VRAM byte count overflows");
        return result;
    }
    const auto required_vram = result.plan.resident_spine_vram_bytes +
                               result.plan.vram_workspace_bytes +
                               result.plan.device_kv_cache_bytes;
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
