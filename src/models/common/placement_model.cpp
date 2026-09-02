#include "strata/engine/placement.hpp"

#include "strata/platform/hardware_profile.hpp"

#include "strata/models/common/checkpoint.hpp"
#include "strata/device/cuda_backend.hpp"
#include "strata/models/deepseek/deepseek_admission.hpp"
#include "strata/models/deepseek/deepseek_checkpoint.hpp"
#include "strata/models/gemma4/gemma4_checkpoint.hpp"
#include "strata/models/glm53/glm53_checkpoint.hpp"
#include "strata/models/kimi_k3/kimi_k3_checkpoint.hpp"
#include "strata/models/inkling/inkling_checkpoint.hpp"
#include "strata/models/laguna/laguna_checkpoint.hpp"
#include "strata/models/common/model_adapter.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <limits>
#include <map>
#include <string>
#include <system_error>

namespace strata {
namespace {

constexpr std::uint64_t kFlashAttentionWorkspaceReserve = 768ULL << 20U;
constexpr std::uint64_t kDeepSeekDeviceWorkspaceReserve = 256ULL << 20U;
// Kimi-K3 stages one routed expert triple (16.7 MiB) per admitted slot
// alongside 7168- and 3584-wide activations and the 96-head KDA state.
constexpr std::uint64_t kKimiDeviceWorkspaceReserve = 1536ULL << 20U;
constexpr std::uint64_t kMinimumDeviceBudget = 2ULL << 30U;
// The same ceiling Dsv4RuntimeConfig applies, taken from the same place, so
// the two cannot drift. Planning against all free host memory would report a
// fit that DeepSeek's own admission then refuses.

[[nodiscard]] std::string module_base(std::string_view name) {
    const auto dot = name.rfind('.');
    return std::string(dot == std::string_view::npos ? name : name.substr(0U, dot));
}

// Groups a checkpoint's tensors into the modules the backend actually uploads,
// so device bytes carry the arena's payload and scale alignment rather than a
// bare sum of shard extents.
struct ModuleSizes {
    std::uint64_t weight_bytes{};
    std::uint64_t scale_bytes{};
    std::uint64_t host_bytes{};
    std::int32_t layer{-1};

    [[nodiscard]] std::uint64_t device_bytes() const noexcept {
        return CudaBackend::weight_storage_bytes(weight_bytes, scale_bytes);
    }
    [[nodiscard]] std::uint64_t source_bytes() const noexcept {
        return weight_bytes + scale_bytes + host_bytes;
    }
};

[[nodiscard]] PlacementItem make_item(PlacementClass component,
                                      const ModuleSizes& sizes,
                                      std::uint64_t decode_reads,
                                      PlacementTier tier, bool spillable) {
    PlacementItem item;
    item.component = component;
    item.layer = sizes.layer;
    item.device_bytes = tier == PlacementTier::Device ? sizes.device_bytes()
                                                      : sizes.source_bytes();
    item.host_bytes = tier == PlacementTier::Device ? sizes.host_bytes
                                                    : sizes.source_bytes();
    item.source_bytes = sizes.source_bytes();
    item.decode_read_bytes = decode_reads;
    item.preferred_tier = tier;
    item.spillable = spillable;
    return item;
}

// ---------------------------------------------------------------- Gemma 4

struct Gemma4Linear {
    std::uint64_t weight_bytes{};
    std::uint64_t scale_bytes{};
    bool found{};
};

[[nodiscard]] Gemma4Linear gemma4_linear(const Gemma4CheckpointReader& checkpoint,
                                         const std::string& base) {
    Gemma4Linear result;
    if (const auto* plain = checkpoint.find(base + ".weight"); plain != nullptr) {
        result.weight_bytes = plain->bytes;
        result.found = true;
        return result;
    }
    const auto* packed = checkpoint.find(base + ".weight_packed");
    const auto* scales = checkpoint.find(base + ".weight_scale");
    if (packed == nullptr || scales == nullptr) return result;
    result.weight_bytes = packed->bytes;
    result.scale_bytes = scales->bytes;
    result.found = true;
    return result;
}

[[nodiscard]] ParseResult<PlacementInventory> build_gemma4_inventory(
    const Gemma4CheckpointReader& checkpoint, std::uint32_t context_tokens) {
    ParseResult<PlacementInventory> result;
    constexpr auto& c = kGemma4ExecutionContract;
    auto& inventory = result.value;
    inventory.model = PlacementModel::Gemma4;
    inventory.model_name = gemma4_31b_it_w8a16_spec().name;
    inventory.layer_count = c.layer_count;
    inventory.maximum_context_tokens = context_tokens;
    inventory.per_device_workspace_bytes = kFlashAttentionWorkspaceReserve;
    inventory.minimum_device_budget_bytes = kMinimumDeviceBudget;
    inventory.contiguous_layer_blocks = true;
    // Gemma 4 is dense and, at the sizes it ships in, entirely device resident.
    // The runtime consumes this assignment, so the plan is prescriptive.
    inventory.prescriptive = true;

    const auto add_linear = [&](PlacementClass component, std::int32_t layer,
                                const std::string& base) {
        const auto linear = gemma4_linear(checkpoint, base);
        if (!linear.found) {
            result.errors.push_back("Gemma 4 checkpoint is missing " + base);
            return;
        }
        ModuleSizes sizes;
        sizes.layer = layer;
        sizes.weight_bytes = linear.weight_bytes;
        sizes.scale_bytes = linear.scale_bytes;
        const auto device = sizes.device_bytes();
        inventory.items.push_back(
            make_item(component, sizes, layer < 0 ? 0U : device,
                      PlacementTier::Device, false));
        if (layer < 0) inventory.items.back().decode_read_bytes = device;
    };

    for (std::uint32_t layer = 0U; layer < c.layer_count; ++layer) {
        const auto index = static_cast<std::int32_t>(layer);
        const bool global = gemma4_global_attention_layer(layer);
        const auto head_dim = global ? c.global_head_dim : c.local_head_dim;
        const auto kv_heads = global ? c.global_key_value_heads
                                     : c.local_key_value_heads;
        const auto prefix = "model.language_model.layers." +
            std::to_string(layer) + ".";
        const auto attention = prefix + "self_attn.";
        add_linear(PlacementClass::Attention, index, attention + "q_proj");
        add_linear(PlacementClass::Attention, index, attention + "k_proj");
        if (!global) {
            add_linear(PlacementClass::Attention, index, attention + "v_proj");
        }
        add_linear(PlacementClass::Attention, index, attention + "o_proj");
        add_linear(PlacementClass::FeedForward, index, prefix + "mlp.gate_proj");
        add_linear(PlacementClass::FeedForward, index, prefix + "mlp.up_proj");
        add_linear(PlacementClass::FeedForward, index, prefix + "mlp.down_proj");
        if (!result.ok()) return result;

        // Four hidden-width norms and two head-width norms per layer. The
        // runtime keeps each as an F32 host vector and an F32 device buffer.
        ModuleSizes norms;
        norms.layer = index;
        norms.weight_bytes = (4ULL * c.hidden_size + 2ULL * head_dim) *
                             sizeof(float);
        norms.host_bytes = norms.weight_bytes;
        auto norm_item = make_item(PlacementClass::Norm, norms,
                                   norms.weight_bytes, PlacementTier::Device,
                                   false);
        norm_item.device_bytes = norms.weight_bytes;
        inventory.items.push_back(norm_item);

        const auto rows = global ? context_tokens
                                 : std::min(context_tokens, c.sliding_window);
        ModuleSizes cache;
        cache.layer = index;
        cache.weight_bytes = static_cast<std::uint64_t>(rows) * kv_heads *
                             head_dim * 2ULL * sizeof(std::uint16_t);
        // The runtime holds the same rows in host vectors as the authoritative
        // copy and mirrors them into the device buffer.
        cache.host_bytes = cache.weight_bytes;
        auto cache_item = make_item(PlacementClass::KvCache, cache,
                                    cache.weight_bytes, PlacementTier::Device,
                                    false);
        cache_item.device_bytes = cache.weight_bytes;
        inventory.items.push_back(cache_item);
    }

    // Gemma 4 ties the output head to the embedding table. It consumes the
    // final hidden state, so it stays on the device that owns the last layer.
    add_linear(PlacementClass::OutputHead,
               static_cast<std::int32_t>(c.layer_count - 1U),
               "model.language_model.embed_tokens");
    if (!result.ok()) return result;
    inventory.items.back().decode_read_bytes = inventory.items.back().device_bytes;

    // The vision tower is loaded onto device 0 by the runtime and is not read
    // during text decode.
    std::vector<std::string> vision_bases{
        "model.vision_tower.patch_embedder.input_proj",
        "model.embed_vision.embedding_projection"};
    for (std::uint32_t layer = 0U; layer < c.vision_layer_count; ++layer) {
        const auto prefix = "model.vision_tower.encoder.layers." +
            std::to_string(layer) + ".";
        const auto attention = prefix + "self_attn.";
        for (const auto& suffix : {attention + "q_proj.linear",
                                   attention + "k_proj.linear",
                                   attention + "v_proj.linear",
                                   attention + "o_proj.linear",
                                   prefix + "mlp.gate_proj.linear",
                                   prefix + "mlp.up_proj.linear",
                                   prefix + "mlp.down_proj.linear"}) {
            vision_bases.push_back(suffix);
        }
    }
    ModuleSizes vision;
    for (const auto& base : vision_bases) {
        const auto linear = gemma4_linear(checkpoint, base);
        if (!linear.found) {
            result.errors.push_back("Gemma 4 checkpoint is missing " + base);
            return result;
        }
        vision.weight_bytes += CudaBackend::weight_storage_bytes(
            linear.weight_bytes, linear.scale_bytes);
    }
    auto vision_item = make_item(PlacementClass::Vision, vision, 0U,
                                 PlacementTier::Device, false);
    vision_item.device_bytes = vision.weight_bytes;
    vision_item.fixed_device_slot = 0;
    inventory.items.push_back(vision_item);
    return result;
}

// ------------------------------------------------------------------- GLM

[[nodiscard]] PlacementClass glm_component(GlmTensorRole role) noexcept {
    switch (role) {
        case GlmTensorRole::Embedding: return PlacementClass::Embedding;
        case GlmTensorRole::OutputHead: return PlacementClass::OutputHead;
        case GlmTensorRole::Norm: return PlacementClass::Norm;
        case GlmTensorRole::Attention:
        case GlmTensorRole::AttentionIndexer: return PlacementClass::Attention;
        case GlmTensorRole::DenseMlp: return PlacementClass::FeedForward;
        case GlmTensorRole::Router: return PlacementClass::Router;
        case GlmTensorRole::SharedExpert: return PlacementClass::SharedExpert;
        case GlmTensorRole::RoutedExpert: return PlacementClass::RoutedExpert;
        case GlmTensorRole::MtpState:
        case GlmTensorRole::Count: break;
    }
    return PlacementClass::Norm;
}

[[nodiscard]] ParseResult<PlacementInventory> build_glm_inventory(
    const GlmIndexManifest& manifest, std::uint32_t context_tokens,
    bool flash_attention) {
    ParseResult<PlacementInventory> result;
    constexpr auto& c = kGlm52ExecutionContract;
    auto& inventory = result.value;
    inventory.model = PlacementModel::Glm52;
    inventory.model_name = glm52_w4a16_spec().name;
    inventory.layer_count = c.layer_count;
    inventory.maximum_context_tokens = context_tokens;
    inventory.per_device_workspace_bytes =
        flash_attention ? kFlashAttentionWorkspaceReserve : 0U;
    inventory.minimum_device_budget_bytes = kMinimumDeviceBudget;
    // The GLM runtime interleaves layers over a VRAM-weighted round robin and
    // its expert cache is tuned against that schedule. Report it; do not
    // silently re-place a validated runtime.
    inventory.contiguous_layer_blocks = false;
    inventory.prescriptive = false;

    std::map<std::string, ModuleSizes> modules;
    std::map<std::string, PlacementClass> module_class;
    ModuleSizes routed;
    std::uint64_t routed_modules = 0U;
    for (const auto& tensor : manifest.tensors) {
        if (tensor.mtp || tensor.role == GlmTensorRole::MtpState) continue;
        if (tensor.component == GlmTensorComponent::LogicalShape) continue;
        if (tensor.role == GlmTensorRole::RoutedExpert) {
            if (tensor.component == GlmTensorComponent::Scale) {
                routed.scale_bytes += tensor.source_bytes;
            } else {
                routed.weight_bytes += tensor.source_bytes;
                ++routed_modules;
            }
            continue;
        }
        const auto base = module_base(tensor.name);
        auto& sizes = modules[base];
        sizes.layer = tensor.layer;
        module_class[base] = glm_component(tensor.role);
        if (tensor.component == GlmTensorComponent::Scale) {
            sizes.scale_bytes += tensor.source_bytes;
        } else if (tensor.role == GlmTensorRole::Norm ||
                   tensor.role == GlmTensorRole::Embedding ||
                   tensor.component == GlmTensorComponent::Bias) {
            sizes.host_bytes += tensor.source_bytes;
        } else {
            sizes.weight_bytes += tensor.source_bytes;
        }
    }
    for (const auto& [base, sizes] : modules) {
        const auto component = module_class.at(base);
        const bool host = sizes.weight_bytes == 0U;
        auto item_sizes = sizes;
        if (component == PlacementClass::OutputHead && sizes.layer < 0) {
            item_sizes.layer = static_cast<std::int32_t>(c.layer_count - 1U);
        }
        const auto tier = host ? PlacementTier::Host : PlacementTier::Device;
        const auto reads = host ? 0U : item_sizes.device_bytes();
        inventory.items.push_back(
            make_item(component, item_sizes, reads, tier, false));
    }
    if (routed.weight_bytes != 0U) {
        const auto sparse_layers = c.layer_count - c.dense_prefix_layers;
        const auto triplets = routed_modules == 0U ? 1U : routed_modules;
        const auto per_module = (routed.weight_bytes + routed.scale_bytes) /
                                triplets;
        // Three projections per expert, top-k experts per sparse layer, once
        // per decode step.
        const auto reads = per_module * 3ULL * c.experts_per_token * sparse_layers;
        auto item = make_item(PlacementClass::RoutedExpert, routed, reads,
                              PlacementTier::Device, true);
        item.device_bytes = routed.weight_bytes + routed.scale_bytes;
        item.host_bytes = item.device_bytes;
        item.device_cache_only = true;
        inventory.items.push_back(item);
    }

    // Compact latent KV plus RoPE per layer, and one indexer key set per
    // indexer layer, all F32 in host memory.
    constexpr std::uint32_t kIndexDim = 128U;
    ModuleSizes cache;
    cache.host_bytes = static_cast<std::uint64_t>(context_tokens) *
        (c.kv_lora_rank + c.rope_head_dim) * sizeof(float) * c.layer_count;
    cache.host_bytes += static_cast<std::uint64_t>(context_tokens) * kIndexDim *
        sizeof(float) * c.layer_count;
    inventory.items.push_back(make_item(PlacementClass::KvCache, cache, 0U,
                                        PlacementTier::Host, false));
    return result;
}


// ------------------------------------------------------------- GLM-5.3

[[nodiscard]] PlacementClass glm53_component(
    Glm53TensorRole role) noexcept {
    switch (role) {
        case Glm53TensorRole::Embedding: return PlacementClass::Embedding;
        case Glm53TensorRole::OutputHead: return PlacementClass::OutputHead;
        case Glm53TensorRole::Norm:
        case Glm53TensorRole::Mhc: return PlacementClass::Norm;
        case Glm53TensorRole::KdaAttention:
        case Glm53TensorRole::SparseAttention:
        case Glm53TensorRole::AttentionIndexer:
            return PlacementClass::Attention;
        case Glm53TensorRole::DenseMlp: return PlacementClass::FeedForward;
        case Glm53TensorRole::Router: return PlacementClass::Router;
        case Glm53TensorRole::SharedExpert: return PlacementClass::SharedExpert;
        case Glm53TensorRole::RoutedExpert: return PlacementClass::RoutedExpert;
        case Glm53TensorRole::Vision: return PlacementClass::Vision;
        case Glm53TensorRole::Mtp:
        case Glm53TensorRole::Count: break;
    }
    return PlacementClass::Norm;
}

[[nodiscard]] std::string glm53_module_base(std::string_view name) {
    for (const auto suffix : {std::string_view{".weight_scale_inv"},
                              std::string_view{".weight"},
                              std::string_view{".bias"}}) {
        if (name.size() >= suffix.size() &&
            name.substr(name.size() - suffix.size()) == suffix) {
            return std::string(name.substr(0U, name.size() - suffix.size()));
        }
    }
    return std::string(name);
}

[[nodiscard]] ParseResult<PlacementInventory> build_glm53_inventory(
    const Glm53IndexManifest& manifest, std::uint32_t context_tokens) {
    ParseResult<PlacementInventory> result;
    auto& inventory = result.value;
    inventory.model = PlacementModel::Glm53;
    inventory.model_name = "GLM-5.3-Flash";
    inventory.layer_count = 45U;
    inventory.maximum_context_tokens = context_tokens;
    inventory.per_device_workspace_bytes = 2ULL << 30U;
    inventory.minimum_device_budget_bytes = kMinimumDeviceBudget;
    inventory.contiguous_layer_blocks = false;
    // The checkpoint remains the canonical backing store. At runtime the
    // non-expert spine is pinned opportunistically and experts enter a
    // free-VRAM-sized demand cache, so this inventory is a cold-cache bound;
    // it does not prescribe or claim a fixed resident layout.
    inventory.prescriptive = false;

    std::map<std::string, ModuleSizes> modules;
    std::map<std::string, PlacementClass> classes;
    for (const auto& tensor : manifest.tensors) {
        if (tensor.role == Glm53TensorRole::Vision ||
            tensor.role == Glm53TensorRole::Mtp) {
            continue;
        }
        const auto base = glm53_module_base(tensor.name);
        auto& sizes = modules[base];
        sizes.layer = tensor.layer;
        classes[base] = glm53_component(tensor.role);
        if (tensor.component == Glm53TensorComponent::Scale) {
            sizes.scale_bytes += tensor.source_bytes;
        } else if (tensor.component == Glm53TensorComponent::Weight) {
            sizes.weight_bytes += tensor.source_bytes;
        } else {
            sizes.host_bytes += tensor.source_bytes;
        }
    }
    for (const auto& [base, sizes] : modules) {
        auto source = sizes;
        source.host_bytes = sizes.source_bytes();
        source.weight_bytes = 0U;
        source.scale_bytes = 0U;
        auto reads = source.host_bytes;
        const auto component = classes.at(base);
        if (component == PlacementClass::Embedding) {
            reads = 4096ULL * sizeof(std::uint16_t);
        } else if (component == PlacementClass::RoutedExpert) {
            // The manifest has every expert projection as its own module, but
            // exact decode reads only the eight selected experts in each MoE
            // layer. Summing this fraction over all 288 experts gives eight
            // complete gate/up/down triplets per sparse layer.
            reads = source.host_bytes * 8ULL / 288ULL;
        }
        inventory.items.push_back(make_item(
            component, source, reads, PlacementTier::Storage, false));
    }

    ModuleSizes state;
    constexpr std::uint64_t kKdaLayers = 34U;
    constexpr std::uint64_t kSparseLayers = 11U;
    state.host_bytes = kKdaLayers * 64ULL * 128ULL * 128ULL * sizeof(float);
    state.host_bytes += kKdaLayers * 3ULL * 8192ULL * 3ULL * sizeof(float);
    state.host_bytes += kSparseLayers * context_tokens * 512ULL * sizeof(float);
    inventory.items.push_back(make_item(PlacementClass::KvCache, state, 0U,
                                        PlacementTier::Host, false));
    return result;
}


// --------------------------------------------------------------- Inkling

[[nodiscard]] ParseResult<PlacementInventory> build_inkling_inventory(
    const InklingCheckpointReader& checkpoint, std::uint32_t context_tokens) {
    ParseResult<PlacementInventory> result;
    constexpr auto& c = kInklingExecutionContract;
    auto& inventory = result.value;
    inventory.model = PlacementModel::Inkling;
    const bool mxfp4 = checkpoint.format() ==
                       InklingCheckpointFormat::Mxfp4Group32;
    inventory.model_name = mxfp4 ? inkling_small_mxfp4_spec().name
                                 : inkling_small_nvfp4_spec().name;
    inventory.layer_count = c.layer_count;
    inventory.maximum_context_tokens = context_tokens;
    inventory.per_device_workspace_bytes = kFlashAttentionWorkspaceReserve;
    inventory.minimum_device_budget_bytes = kMinimumDeviceBudget;
    inventory.contiguous_layer_blocks = false;
    inventory.prescriptive = false;

    const auto add_linear = [&](PlacementClass component, std::int32_t layer,
                                const std::string& name, std::uint64_t rows,
                                std::uint64_t columns) {
        auto module = checkpoint.linear(name, rows, columns);
        if (!module.ok()) {
            result.errors.push_back("Inkling checkpoint is missing " + name);
            return;
        }
        ModuleSizes sizes;
        sizes.layer = layer;
        sizes.weight_bytes = module.value.source_bytes();
        inventory.items.push_back(make_item(component, sizes,
                                            sizes.device_bytes(),
                                            PlacementTier::Device, false));
    };

    const auto hidden = static_cast<std::uint64_t>(c.hidden_size);
    const auto query_columns =
        static_cast<std::uint64_t>(c.attention_heads) * c.head_dim;
    const auto kv_columns =
        static_cast<std::uint64_t>(c.key_value_heads) * c.head_dim;
    ModuleSizes routed;
    std::uint64_t routed_modules = 0U;
    for (std::uint32_t layer = 0U; layer < c.layer_count; ++layer) {
        const auto index = static_cast<std::int32_t>(layer);
        const auto prefix = mxfp4
            ? "language_model.model.layers." + std::to_string(layer) + "."
            : inkling_layer_prefix(layer);
        const auto attention = prefix + (mxfp4 ? "self_attn." : "attn.");
        const auto mlp = prefix + "mlp.";
        add_linear(PlacementClass::Attention, index,
                   attention + (mxfp4 ? "q_proj" : "wq_du.weight"),
                   query_columns, hidden);
        add_linear(PlacementClass::Attention, index,
                   attention + (mxfp4 ? "k_proj" : "wk_dv.weight"),
                   kv_columns, hidden);
        add_linear(PlacementClass::Attention, index,
                   attention + (mxfp4 ? "v_proj" : "wv_dv.weight"),
                   kv_columns, hidden);
        add_linear(PlacementClass::Attention, index,
                   attention + (mxfp4 ? "o_proj" : "wo_ud.weight"),
                   hidden, query_columns);
        // The relative branch is a fourth projection, not an optional extra:
        // without it the layer has no position signal at all.
        add_linear(PlacementClass::Attention, index,
                   attention + (mxfp4 ? "r_proj" : "wr_du.weight"),
                   static_cast<std::uint64_t>(c.attention_heads) * c.relative_dim,
                   hidden);
        if (!inkling_sparse_layer(layer)) {
            if (mxfp4) {
                add_linear(PlacementClass::FeedForward, index,
                           mlp + "gate_proj", c.dense_intermediate_size, hidden);
                add_linear(PlacementClass::FeedForward, index,
                           mlp + "up_proj", c.dense_intermediate_size, hidden);
            } else {
                add_linear(PlacementClass::FeedForward, index,
                           mlp + "w13_dn.weight",
                           2ULL * c.dense_intermediate_size, hidden);
            }
            add_linear(PlacementClass::FeedForward, index,
                       mlp + (mxfp4 ? "down_proj" : "w2_md.weight"),
                       hidden, c.dense_intermediate_size);
        } else {
            add_linear(PlacementClass::Router, index,
                       mlp + (mxfp4 ? "gate_weight" : "gate.weight"),
                       static_cast<std::uint64_t>(c.routed_experts) +
                           c.shared_experts,
                       hidden);
            // Both sinks run on every token, so they are resident like a
            // shared expert rather than cached like a routed one.
            const auto shared_projections = mxfp4
                ? std::vector<std::pair<std::string,
                       std::pair<std::uint64_t, std::uint64_t>>>{
                      {mlp + "shared_experts.gate_proj",
                       {c.expert_intermediate_size, hidden}},
                      {mlp + "shared_experts.up_proj",
                       {c.expert_intermediate_size, hidden}},
                      {mlp + "shared_experts.down_proj",
                       {hidden, c.expert_intermediate_size}}}
                : std::vector<std::pair<std::string,
                       std::pair<std::uint64_t, std::uint64_t>>>{
                      {mlp + "shared_experts.shared_w13_weight",
                       {2ULL * c.expert_intermediate_size, hidden}},
                      {mlp + "shared_experts.shared_w2_weight",
                       {hidden, c.expert_intermediate_size}}};
            for (const auto& projection : shared_projections) {
                auto stack = checkpoint.expert_stack(
                    projection.first, layer, c.shared_experts,
                    projection.second.first, projection.second.second);
                if (!stack.ok()) {
                    result.errors.push_back("Inkling checkpoint is missing " +
                                            projection.first);
                    return result;
                }
                ModuleSizes shared;
                shared.layer = index;
                shared.weight_bytes = stack.value.source_bytes();
                inventory.items.push_back(make_item(
                    PlacementClass::SharedExpert, shared, shared.device_bytes(),
                    PlacementTier::Device, false));
            }
            const auto routed_projections = mxfp4
                ? std::vector<std::pair<std::string,
                       std::pair<std::uint64_t, std::uint64_t>>>{
                      {mlp + "switch_mlp.gate_proj",
                       {c.expert_intermediate_size, hidden}},
                      {mlp + "switch_mlp.up_proj",
                       {c.expert_intermediate_size, hidden}},
                      {mlp + "switch_mlp.down_proj",
                       {hidden, c.expert_intermediate_size}}}
                : std::vector<std::pair<std::string,
                       std::pair<std::uint64_t, std::uint64_t>>>{
                      {mlp + "experts.w13_weight",
                       {2ULL * c.expert_intermediate_size, hidden}},
                      {mlp + "experts.w2_weight",
                       {hidden, c.expert_intermediate_size}}};
            for (const auto& projection : routed_projections) {
                auto stack = checkpoint.expert_stack(
                    projection.first, layer, c.routed_experts,
                    projection.second.first, projection.second.second);
                if (!stack.ok()) {
                    result.errors.push_back("Inkling checkpoint is missing " +
                                            projection.first);
                    return result;
                }
                if (stack.value.encoding == InklingTensorEncoding::Plain) {
                    routed.weight_bytes += stack.value.weight->bytes;
                } else {
                    routed.weight_bytes += stack.value.packed->bytes;
                    routed.scale_bytes += stack.value.scale->bytes;
                }
                routed_modules += c.routed_experts;
            }
        }
        if (!result.ok()) return result;

        // Two hidden-width norms, two head-width norms, four convolution
        // kernels, and on sparse layers the routing correction bias.
        ModuleSizes norms;
        norms.layer = index;
        norms.host_bytes =
            (2ULL * hidden + 2ULL * c.head_dim +
             2ULL * kv_columns * c.short_conv_kernel +
             2ULL * hidden * c.short_conv_kernel +
             (inkling_sparse_layer(layer) ? c.routed_experts : 0U)) *
            sizeof(float);
        inventory.items.push_back(
            make_item(PlacementClass::Norm, norms, 0U, PlacementTier::Host, false));

        const auto rows = inkling_global_attention_layer(layer)
            ? context_tokens
            : std::min(context_tokens, c.sliding_window);
        ModuleSizes cache;
        cache.layer = index;
        cache.host_bytes = static_cast<std::uint64_t>(rows) * kv_columns * 2ULL *
                           sizeof(std::uint16_t);
        inventory.items.push_back(make_item(PlacementClass::KvCache, cache, 0U,
                                            PlacementTier::Host, false));
    }

    add_linear(PlacementClass::OutputHead,
               static_cast<std::int32_t>(c.layer_count - 1U),
               mxfp4 ? "language_model.lm_head" : "model.llm.unembed.weight",
               c.padded_vocabulary_size, hidden);
    if (!result.ok()) return result;
    inventory.items.back().decode_read_bytes =
        inventory.items.back().device_bytes;

    // The embedding table is read one row per token from the host mapping and
    // is never uploaded.
    const auto embedding_name = mxfp4
        ? "language_model.model.embed_tokens.weight"
        : "model.llm.embed.weight";
    if (const auto* embedding = checkpoint.find(embedding_name);
        embedding != nullptr) {
        ModuleSizes table;
        table.layer = -1;
        table.host_bytes = embedding->bytes;
        if (mxfp4) {
            if (const auto* scales = checkpoint.find(
                    "language_model.model.embed_tokens.scales");
                scales != nullptr) {
                table.host_bytes += scales->bytes;
            }
        }
        inventory.items.push_back(make_item(PlacementClass::Embedding, table, 0U,
                                            PlacementTier::Host, false));
    }

    if (routed_modules != 0U) {
        ModuleSizes experts;
        experts.layer = -1;
        experts.weight_bytes = routed.weight_bytes;
        experts.scale_bytes = routed.scale_bytes;
        // Six of 256 experts per sparse layer are read per token. NVFP4 stores
        // fused gate/up plus down; MXFP4 stores gate, up, and down separately.
        const auto per_module = experts.device_bytes() / routed_modules;
        const auto projections_per_layer = mxfp4 ? 3ULL : 2ULL;
        inventory.items.push_back(make_item(PlacementClass::RoutedExpert, experts,
                                            per_module * c.experts_per_token *
                                                projections_per_layer *
                                                (c.layer_count -
                                                 c.dense_prefix_layers),
                                            PlacementTier::Host, true));
    }
    return result;
}

// ---------------------------------------------------------------- Laguna

[[nodiscard]] ParseResult<PlacementInventory> build_laguna_inventory(
    const LagunaCheckpointReader& checkpoint, std::uint32_t context_tokens) {
    ParseResult<PlacementInventory> result;
    constexpr auto& c = kLagunaExecutionContract;
    auto& inventory = result.value;
    inventory.model = PlacementModel::Laguna;
    inventory.model_name =
        checkpoint.format() == LagunaCheckpointFormat::Mxfp4Group32
            ? laguna_s21_mxfp4_spec().name
            : laguna_s21_nvfp4_spec().name;
    inventory.layer_count = c.layer_count;
    inventory.maximum_context_tokens = context_tokens;
    inventory.per_device_workspace_bytes = kFlashAttentionWorkspaceReserve;
    inventory.minimum_device_budget_bytes = kMinimumDeviceBudget;
    // Like GLM, the Laguna runtime interleaves layers over a VRAM-weighted
    // round robin and sizes its expert cache from what the spine leaves behind.
    // Report that placement; do not re-place a validated runtime.
    inventory.contiguous_layer_blocks = false;
    inventory.prescriptive = false;

    const auto add_linear = [&](PlacementClass component, std::int32_t layer,
                                const std::string& base, std::uint64_t rows,
                                std::uint64_t columns) {
        auto module = checkpoint.linear(base, rows, columns);
        if (!module.ok()) {
            result.errors.push_back("Laguna checkpoint is missing " + base);
            return;
        }
        ModuleSizes sizes;
        sizes.layer = layer;
        if (module.value.encoding == LagunaTensorEncoding::Plain) {
            sizes.weight_bytes = module.value.weight->bytes;
        } else {
            sizes.weight_bytes = module.value.packed->bytes;
            sizes.scale_bytes = module.value.scale->bytes;
        }
        const auto device = sizes.device_bytes();
        inventory.items.push_back(
            make_item(component, sizes, device, PlacementTier::Device, false));
    };

    const auto hidden = static_cast<std::uint64_t>(c.hidden_size);
    const auto kv_columns =
        static_cast<std::uint64_t>(c.key_value_heads) * c.head_dim;
    ModuleSizes routed;
    std::uint64_t routed_modules = 0U;
    for (std::uint32_t layer = 0U; layer < c.layer_count; ++layer) {
        const auto index = static_cast<std::int32_t>(layer);
        const auto prefix = "model.layers." + std::to_string(layer) + ".";
        const auto attention = prefix + "self_attn.";
        const auto mlp = prefix + "mlp.";
        const auto heads = static_cast<std::uint64_t>(laguna_attention_heads(layer));
        add_linear(PlacementClass::Attention, index, attention + "q_proj",
                   heads * c.head_dim, hidden);
        add_linear(PlacementClass::Attention, index, attention + "k_proj",
                   kv_columns, hidden);
        add_linear(PlacementClass::Attention, index, attention + "v_proj",
                   kv_columns, hidden);
        add_linear(PlacementClass::Attention, index, attention + "o_proj",
                   hidden, heads * c.head_dim);
        add_linear(PlacementClass::Attention, index, attention + "g_proj",
                   heads, hidden);
        if (!laguna_sparse_layer(layer)) {
            add_linear(PlacementClass::FeedForward, index, mlp + "gate_proj",
                       c.dense_intermediate_size, hidden);
            add_linear(PlacementClass::FeedForward, index, mlp + "up_proj",
                       c.dense_intermediate_size, hidden);
            add_linear(PlacementClass::FeedForward, index, mlp + "down_proj",
                       hidden, c.dense_intermediate_size);
        } else {
            add_linear(PlacementClass::Router, index, mlp + "gate",
                       c.routed_experts, hidden);
            add_linear(PlacementClass::SharedExpert, index,
                       mlp + "shared_expert.gate_proj",
                       c.shared_expert_intermediate_size, hidden);
            add_linear(PlacementClass::SharedExpert, index,
                       mlp + "shared_expert.up_proj",
                       c.shared_expert_intermediate_size, hidden);
            add_linear(PlacementClass::SharedExpert, index,
                       mlp + "shared_expert.down_proj", hidden,
                       c.shared_expert_intermediate_size);
            for (std::uint32_t expert = 0U; expert < c.routed_experts; ++expert) {
                const auto base = mlp + "experts." + std::to_string(expert) + ".";
                for (const auto& projection :
                     {std::pair<std::string, std::pair<std::uint64_t, std::uint64_t>>{
                          base + "gate_proj", {c.expert_intermediate_size, hidden}},
                      {base + "up_proj", {c.expert_intermediate_size, hidden}},
                      {base + "down_proj", {hidden, c.expert_intermediate_size}}}) {
                    auto module = checkpoint.linear(projection.first,
                                                    projection.second.first,
                                                    projection.second.second);
                    if (!module.ok()) {
                        result.errors.push_back("Laguna checkpoint is missing " +
                                                projection.first);
                        return result;
                    }
                    if (module.value.encoding == LagunaTensorEncoding::Plain) {
                        routed.weight_bytes += module.value.weight->bytes;
                    } else {
                        routed.weight_bytes += module.value.packed->bytes;
                        routed.scale_bytes += module.value.scale->bytes;
                    }
                    ++routed_modules;
                }
            }
        }
        if (!result.ok()) return result;

        // Two hidden-width norms, two head-width norms, and (on sparse layers)
        // the routing correction bias, all kept as F32 host vectors.
        ModuleSizes norms;
        norms.layer = index;
        norms.host_bytes = (2ULL * hidden + 2ULL * c.head_dim +
                            (laguna_sparse_layer(layer) ? c.routed_experts : 0U)) *
                           sizeof(float);
        inventory.items.push_back(
            make_item(PlacementClass::Norm, norms, 0U, PlacementTier::Host, false));

        // Sliding layers retain at most one window of BF16 keys and values.
        const auto rows = laguna_global_attention_layer(layer)
            ? context_tokens
            : std::min(context_tokens, c.sliding_window);
        ModuleSizes cache;
        cache.layer = index;
        cache.host_bytes = static_cast<std::uint64_t>(rows) * kv_columns * 2ULL *
                           sizeof(std::uint16_t);
        inventory.items.push_back(make_item(PlacementClass::KvCache, cache, 0U,
                                            PlacementTier::Host, false));
    }

    add_linear(PlacementClass::OutputHead,
               static_cast<std::int32_t>(c.layer_count - 1U), "lm_head",
               c.vocabulary_size, hidden);
    if (!result.ok()) return result;
    inventory.items.back().decode_read_bytes =
        inventory.items.back().device_bytes;

    // The embedding table is read one row per token from the host mapping and
    // is never uploaded.
    if (const auto* embedding = checkpoint.find("model.embed_tokens.weight");
        embedding != nullptr) {
        ModuleSizes sizes;
        sizes.host_bytes = embedding->bytes;
        inventory.items.push_back(make_item(PlacementClass::Embedding, sizes, 0U,
                                            PlacementTier::Host, false));
    }

    if (routed.weight_bytes != 0U) {
        const auto sparse_layers = c.layer_count - c.dense_prefix_layers;
        const auto triplets = routed_modules == 0U ? 1U : routed_modules;
        const auto per_module = (routed.weight_bytes + routed.scale_bytes) /
                                triplets;
        // Three projections per expert, top-k experts per sparse layer, once
        // per decode step. Whether those reads hit VRAM or the host mapping is
        // exactly what the cache size decides.
        const auto reads = per_module * 3ULL * c.experts_per_token * sparse_layers;
        auto item = make_item(PlacementClass::RoutedExpert, routed, reads,
                              PlacementTier::Device, true);
        item.device_bytes = routed.weight_bytes + routed.scale_bytes;
        item.host_bytes = item.device_bytes;
        item.device_cache_only = true;
        inventory.items.push_back(item);
    }
    return result;
}

// -------------------------------------------------------------- Kimi-K3

[[nodiscard]] PlacementClass kimi_component(KimiTensorRole role) noexcept {
    switch (role) {
        case KimiTensorRole::Embedding: return PlacementClass::Embedding;
        case KimiTensorRole::OutputHead: return PlacementClass::OutputHead;
        case KimiTensorRole::Norm:
        case KimiTensorRole::AttentionResidual: return PlacementClass::Norm;
        case KimiTensorRole::MlaAttention:
        case KimiTensorRole::KdaAttention: return PlacementClass::Attention;
        case KimiTensorRole::DenseMlp: return PlacementClass::FeedForward;
        case KimiTensorRole::Router: return PlacementClass::Router;
        case KimiTensorRole::SharedExpert:
        // The latent down/up projections are read on every token like the
        // shared experts are, not once per routed hit, so they belong with the
        // dense spine rather than with the routed set.
        case KimiTensorRole::LatentMoeProjection: return PlacementClass::SharedExpert;
        case KimiTensorRole::RoutedExpert: return PlacementClass::RoutedExpert;
        case KimiTensorRole::Vision:
        case KimiTensorRole::VisionProjector: return PlacementClass::Vision;
        case KimiTensorRole::Count: break;
    }
    return PlacementClass::Norm;
}

[[nodiscard]] ParseResult<PlacementInventory> build_kimi_k3_inventory(
    const KimiIndexManifest& manifest, std::uint32_t context_tokens) {
    ParseResult<PlacementInventory> result;
    constexpr auto& c = kKimiK3ExecutionContract;
    auto& inventory = result.value;
    inventory.model = PlacementModel::KimiK3;
    inventory.model_name = kimi_k3_mxfp4_spec().name;
    inventory.layer_count = c.layer_count;
    inventory.maximum_context_tokens = context_tokens;
    inventory.per_device_workspace_bytes = kKimiDeviceWorkspaceReserve;
    inventory.minimum_device_budget_bytes = kMinimumDeviceBudget;
    inventory.contiguous_layer_blocks = false;
    // Descriptive until stage 4's runtime consumes the assignment and a real
    // load validates it. Promoting it before then would have the planner
    // prescribe a placement nothing has executed.
    inventory.prescriptive = false;

    // The dense spine is plain BF16 and 103 GiB of it, against 64 GiB of VRAM.
    // Half of it must live in host memory and stream over PCIe every step, so
    // it is spillable — but only as far as the host tier.
    std::map<std::string, ModuleSizes> modules;
    std::map<std::string, PlacementClass> module_class;
    ModuleSizes routed;
    std::uint64_t routed_modules = 0U;
    for (const auto& tensor : manifest.tensors) {
        if (tensor.role == KimiTensorRole::RoutedExpert) {
            if (tensor.component == KimiTensorComponent::Scale) {
                routed.scale_bytes += tensor.source_bytes;
            } else {
                routed.weight_bytes += tensor.source_bytes;
                ++routed_modules;
            }
            continue;
        }
        const auto base = module_base(tensor.name);
        auto& sizes = modules[base];
        sizes.layer = tensor.layer;
        module_class[base] = kimi_component(tensor.role);
        // Norms, the attention-residual pseudo-queries, and the KDA scalar
        // state are decoded to F32 host vectors and mirrored to the device.
        const bool host_parameter = tensor.role == KimiTensorRole::Norm ||
            tensor.role == KimiTensorRole::AttentionResidual;
        if (host_parameter) {
            auto bytes = tensor.source_bytes;
            if (tensor.source_dtype == SafetensorsDtype::Bf16) bytes *= 2U;
            sizes.host_bytes += bytes;
        } else {
            sizes.weight_bytes += tensor.source_bytes;
        }
    }
    for (const auto& [base, sizes] : modules) {
        const auto component = module_class.at(base);
        auto item_sizes = sizes;
        if (component == PlacementClass::OutputHead && sizes.layer < 0) {
            item_sizes.layer = static_cast<std::int32_t>(c.layer_count - 1U);
        }
        const bool host_only = item_sizes.weight_bytes == 0U;
        if (host_only) {
            inventory.items.push_back(make_item(component, item_sizes, 0U,
                                                PlacementTier::Host, false));
            continue;
        }
        auto item = make_item(component, item_sizes, item_sizes.device_bytes(),
                              PlacementTier::Device, true);
        // The vision tower is not read during text decode and is small enough
        // to pin, so it stays where the runtime puts it.
        if (component == PlacementClass::Vision) {
            item.spillable = false;
            item.fixed_device_slot = 0;
            item.decode_read_bytes = 0U;
        }
        item.deepest_tier = PlacementTier::Host;
        item.host_bytes = item_sizes.source_bytes();
        inventory.items.push_back(std::move(item));
    }

    if (routed.weight_bytes != 0U) {
        const auto triplets = routed_modules == 0U ? 1U : routed_modules;
        const auto per_module = (routed.weight_bytes + routed.scale_bytes) / triplets;
        // Three modules per expert, top-k experts per MoE layer, once per step.
        const auto reads = per_module * 3ULL * c.experts_per_token *
                           (c.layer_count - c.dense_prefix_layers);
        auto item = make_item(PlacementClass::RoutedExpert, routed, reads,
                              PlacementTier::Device, true);
        item.device_bytes = routed.weight_bytes + routed.scale_bytes;
        item.host_bytes = item.device_bytes;
        item.device_cache_only = true;
        item.deepest_tier = PlacementTier::Storage;
        inventory.items.push_back(item);
    }

    // KV is cheap here and the recurrent state does not grow with context at
    // all. 24 gated MLA layers hold a 512-wide latent plus a 64-wide shared
    // rope row per token; the 69 KDA layers hold a fixed [heads, d_k, d_v]
    // state each. That is why a 1M-token context is not the constraint the
    // layer count suggests.
    ModuleSizes cache;
    cache.host_bytes = static_cast<std::uint64_t>(context_tokens) *
        (c.kv_lora_rank + c.rope_head_dim) * sizeof(float) * 24ULL;
    const auto kda_state_bytes = static_cast<std::uint64_t>(c.linear_attention_heads) *
        c.linear_head_dim * c.value_head_dim * sizeof(float);
    const auto kda_conv_bytes = static_cast<std::uint64_t>(c.linear_attention_heads) *
        c.linear_head_dim * c.short_conv_kernel * 3ULL * sizeof(float);
    cache.host_bytes += (kda_state_bytes + kda_conv_bytes) * 69ULL;
    inventory.items.push_back(make_item(PlacementClass::KvCache, cache, 0U,
                                        PlacementTier::Host, false));
    return result;
}

// -------------------------------------------------------------- DeepSeek

[[nodiscard]] PlacementClass deepseek_component(Dsv4TensorRole role) noexcept {
    switch (role) {
        case Dsv4TensorRole::Embedding: return PlacementClass::Embedding;
        case Dsv4TensorRole::OutputHead: return PlacementClass::OutputHead;
        case Dsv4TensorRole::Norm: return PlacementClass::Norm;
        case Dsv4TensorRole::Attention:
        case Dsv4TensorRole::AttentionCompressor:
        case Dsv4TensorRole::AttentionIndexer: return PlacementClass::Attention;
        case Dsv4TensorRole::Router: return PlacementClass::Router;
        case Dsv4TensorRole::SharedExpert: return PlacementClass::SharedExpert;
        case Dsv4TensorRole::RoutedExpert: return PlacementClass::RoutedExpert;
        case Dsv4TensorRole::Mhc: return PlacementClass::FeedForward;
        case Dsv4TensorRole::DsparkState:
        case Dsv4TensorRole::Count: break;
    }
    return PlacementClass::Norm;
}

[[nodiscard]] ParseResult<PlacementInventory> build_deepseek_inventory(
    const Dsv4IndexManifest& manifest, const PlacementRequest& request,
    std::span<const std::uint64_t> budgets, std::uint64_t host_ceiling_bytes,
    std::vector<std::string>& admission_notes) {
    ParseResult<PlacementInventory> result;
    constexpr auto& c = kDeepSeekV4ExecutionContract;
    auto& inventory = result.value;
    inventory.model = PlacementModel::DeepSeekV4;
    inventory.model_name = deepseek_v4_flash_0731_spec().name;
    inventory.layer_count = c.layer_count;
    inventory.maximum_context_tokens = request.maximum_context_tokens;
    inventory.per_device_workspace_bytes = kDeepSeekDeviceWorkspaceReserve;
    inventory.minimum_device_budget_bytes = kMinimumDeviceBudget;
    inventory.contiguous_layer_blocks = false;
    inventory.prescriptive = false;
    inventory.host_capacity_bytes = host_ceiling_bytes;

    std::map<std::string, ModuleSizes> modules;
    std::map<std::string, PlacementClass> module_class;
    ModuleSizes routed;
    std::uint64_t routed_modules = 0U;
    for (const auto& tensor : manifest.tensors) {
        if (tensor.dspark || tensor.role == Dsv4TensorRole::DsparkState) continue;
        if (tensor.role == Dsv4TensorRole::RoutedExpert) {
            if (tensor.component == Dsv4TensorComponent::Scale) {
                routed.scale_bytes += tensor.source_bytes;
            } else {
                routed.weight_bytes += tensor.source_bytes;
                ++routed_modules;
            }
            continue;
        }
        const auto base = module_base(tensor.name);
        auto& sizes = modules[base];
        sizes.layer = tensor.layer;
        module_class[base] = deepseek_component(tensor.role);
        const bool host_parameter = tensor.role == Dsv4TensorRole::Embedding ||
            tensor.role == Dsv4TensorRole::Mhc ||
            tensor.role == Dsv4TensorRole::Norm;
        if (host_parameter) {
            auto bytes = tensor.source_bytes;
            if (tensor.source_dtype == SafetensorsDtype::Bf16 ||
                tensor.source_dtype == SafetensorsDtype::F16) {
                bytes *= 2U;
            } else if (tensor.source_dtype == SafetensorsDtype::I64) {
                bytes /= 2U;
            }
            sizes.host_bytes += bytes;
        } else if (tensor.component == Dsv4TensorComponent::Scale) {
            // wo_a is dequantized to BF16 once, so its FP8 scales never reach
            // the device. Mirrors the DeepSeek admission contract.
            if (tensor.name.find(".attn.wo_a.") == std::string::npos) {
                sizes.scale_bytes += tensor.source_bytes;
            }
        } else if (tensor.component == Dsv4TensorComponent::Weight) {
            sizes.weight_bytes +=
                tensor.name.find(".attn.wo_a.") == std::string::npos
                    ? tensor.source_bytes : tensor.source_bytes * 2U;
        } else {
            sizes.host_bytes += tensor.source_bytes;
        }
    }
    for (const auto& [base, sizes] : modules) {
        const bool host = sizes.weight_bytes == 0U;
        const auto tier = host ? PlacementTier::Host : PlacementTier::Device;
        inventory.items.push_back(
            make_item(module_class.at(base), sizes,
                      host ? 0U : sizes.device_bytes(), tier, false));
    }
    if (routed.weight_bytes != 0U) {
        const auto triplets = routed_modules == 0U ? 1U : routed_modules;
        const auto per_module = (routed.weight_bytes + routed.scale_bytes) /
                                triplets;
        const auto reads = per_module * 3ULL * c.experts_per_token * c.layer_count;
        auto item = make_item(PlacementClass::RoutedExpert, routed, reads,
                              PlacementTier::Device, true);
        item.device_bytes = routed.weight_bytes + routed.scale_bytes;
        item.host_bytes = item.device_bytes;
        item.device_cache_only = true;
        inventory.items.push_back(item);
    }

    // DeepSeek's own admission contract owns the KV and compressor state
    // sizing; reuse it rather than restating a second, divergent model.
    Dsv4AdmissionConfig admission_config;
    admission_config.host_memory_ceiling_bytes = host_ceiling_bytes;
    admission_config.vram_weight_budgets.assign(budgets.begin(), budgets.end());
    admission_config.maximum_context_tokens = request.maximum_context_tokens;
    admission_config.compact_kv_cache = request.block_kv_cache;
    admission_config.enable_mhc_prepack = true;
    admission_config.require_zero_nvme_decode = true;
    auto admission = plan_dsv4_resident_topology(manifest, admission_config);
    ModuleSizes cache;
    cache.host_bytes = admission.plan.kv_state_bytes;
    inventory.items.push_back(make_item(PlacementClass::KvCache, cache, 0U,
                                        PlacementTier::Host, false));
    inventory.host_workspace_bytes = admission.plan.host_workspace_bytes;
    for (auto& error : admission.errors) {
        admission_notes.push_back("DeepSeek admission: " + std::move(error));
    }
    return result;
}

[[nodiscard]] std::vector<std::uint64_t> admitted_budgets(
    const PlacementHardware& hardware, double fraction) {
    std::vector<std::uint64_t> budgets;
    budgets.reserve(hardware.devices.size());
    for (const auto& device : hardware.devices) {
        budgets.push_back(static_cast<std::uint64_t>(
            static_cast<double>(device.free_bytes) * fraction));
    }
    return budgets;
}

struct OpenCheckpoints {
    std::unique_ptr<Gemma4CheckpointReader> gemma4;
    std::unique_ptr<GlmCheckpointReader> glm;
    std::unique_ptr<Glm53CheckpointReader> glm53;
    std::unique_ptr<Dsv4CheckpointReader> deepseek;
    std::unique_ptr<KimiCheckpointReader> kimi;
    std::unique_ptr<LagunaCheckpointReader> laguna;
    std::unique_ptr<InklingCheckpointReader> inkling;
};

[[nodiscard]] ParseResult<PlacementInventory> build_inventory(
    const OpenCheckpoints& checkpoints, const PlacementRequest& request,
    std::uint32_t context_tokens, const PlacementHardware& hardware,
    std::vector<std::string>& admission_notes) {
    PlacementRequest scoped(request);
    scoped.maximum_context_tokens = context_tokens;
    switch (request.model) {
        case PlacementModel::Gemma4:
            return build_gemma4_inventory(*checkpoints.gemma4, context_tokens);
        case PlacementModel::Glm52:
            return build_glm_inventory(checkpoints.glm->manifest(), context_tokens,
                                       request.flash_attention);
        case PlacementModel::Glm53:
            return build_glm53_inventory(checkpoints.glm53->manifest(),
                                         context_tokens);
        case PlacementModel::Laguna:
            return build_laguna_inventory(*checkpoints.laguna, context_tokens);
        case PlacementModel::Inkling:
            return build_inkling_inventory(*checkpoints.inkling, context_tokens);
        case PlacementModel::DeepSeekV4:
            return build_deepseek_inventory(
                checkpoints.deepseek->manifest(), scoped,
                admitted_budgets(hardware, request.vram_cache_fraction),
                std::min(hardware.host_available_bytes,
                         host_hardware_profile().host_usable_bytes()),
                admission_notes);
        case PlacementModel::KimiK3:
            return build_kimi_k3_inventory(checkpoints.kimi->manifest(),
                                           context_tokens);
    }
    ParseResult<PlacementInventory> result;
    result.errors.emplace_back("unknown placement model");
    return result;
}

[[nodiscard]] std::uint32_t model_context_ceiling(PlacementModel model) noexcept {
    switch (model) {
        case PlacementModel::Gemma4:
            return kGemma4ExecutionContract.maximum_context_tokens;
        case PlacementModel::Glm52:
            return kGlm52ExecutionContract.maximum_context_tokens;
        case PlacementModel::Glm53:
            // Was 2,048 while the checkpoint's k-pool sparse indexer was
            // unimplemented and attention ran densely, which is exact only up
            // to `index_topk`. With the indexer the MLA workspace no longer
            // grows with history; the bound is host sequence state, about
            // 33.5 KB per token across the eleven sparse layers.
            return 262144U;
        case PlacementModel::Laguna:
            return kLagunaExecutionContract.maximum_context_tokens;
        case PlacementModel::Inkling:
            return kInklingExecutionContract.maximum_context_tokens;
        case PlacementModel::DeepSeekV4:
            return kDeepSeekV4ExecutionContract.maximum_context_tokens;
        case PlacementModel::KimiK3:
            return kKimiK3ExecutionContract.maximum_context_tokens;
    }
    return 0U;
}

}  // namespace


namespace {

PlacementPlanResult plan_model_placement_impl(const PlacementRequest& request,
                                              const PlacementHardware& hardware) {
    PlacementPlanResult result;
    const auto ceiling = model_context_ceiling(request.model);
    if (request.maximum_context_tokens == 0U ||
        request.maximum_context_tokens > ceiling) {
        result.errors.push_back(
            std::string(to_string(request.model)) +
            " context must be within [1, " + std::to_string(ceiling) + "] tokens");
        return result;
    }
    OpenCheckpoints checkpoints;
    switch (request.model) {
        case PlacementModel::Gemma4: {
            auto opened = Gemma4CheckpointReader::open(request.model_directory);
            if (!opened.ok()) {
                result.errors = std::move(opened.errors);
                return result;
            }
            checkpoints.gemma4 = std::move(opened.value);
            break;
        }
        case PlacementModel::Glm52: {
            auto opened = GlmCheckpointReader::open(request.model_directory);
            if (!opened.ok()) {
                result.errors = std::move(opened.errors);
                return result;
            }
            checkpoints.glm = std::move(opened.value);
            break;
        }
        case PlacementModel::Glm53: {
            auto opened = Glm53CheckpointReader::open(request.model_directory);
            if (!opened.ok()) {
                result.errors = std::move(opened.errors);
                return result;
            }
            checkpoints.glm53 = std::move(opened.value);
            break;
        }
        case PlacementModel::Laguna: {
            auto opened = LagunaCheckpointReader::open(request.model_directory);
            if (!opened.ok()) {
                result.errors = std::move(opened.errors);
                return result;
            }
            checkpoints.laguna = std::move(opened.value);
            break;
        }
        case PlacementModel::Inkling: {
            auto opened = InklingCheckpointReader::open(request.model_directory);
            if (!opened.ok()) {
                result.errors = std::move(opened.errors);
                return result;
            }
            checkpoints.inkling = std::move(opened.value);
            break;
        }
        case PlacementModel::DeepSeekV4: {
            auto opened = Dsv4CheckpointReader::open(request.model_directory);
            if (!opened.ok()) {
                result.errors = std::move(opened.errors);
                return result;
            }
            checkpoints.deepseek = std::move(opened.value);
            break;
        }
        case PlacementModel::KimiK3: {
            auto opened = KimiCheckpointReader::open(request.model_directory);
            if (!opened.ok()) {
                result.errors = std::move(opened.errors);
                return result;
            }
            checkpoints.kimi = std::move(opened.value);
            break;
        }
    }

    std::vector<std::string> admission_notes;
    auto inventory = build_inventory(checkpoints, request,
                                     request.maximum_context_tokens, hardware,
                                     admission_notes);
    if (!inventory.ok()) {
        result.errors = std::move(inventory.errors);
        return result;
    }
    result = solve_placement(inventory.value, hardware, request);
    if (!result.ok()) return result;

    auto identity = probe_model_identity(request.model_directory);
    if (!identity.ok()) {
        result.errors = std::move(identity.errors);
        return result;
    }
    result.value.model_identity = std::move(identity.value);

    // Largest context that still spills nothing. Re-sizes the inventory at
    // every probe rather than extrapolating one measured constant: Gemma 4's
    // local layers stop growing at the sliding window, so KV bytes are not
    // linear in context. A plan that already reaches NVMe has no such
    // threshold — every extra token simply trades cache residency for KV — so
    // the search is skipped rather than reported as a limit it is not.
    if (result.value.fits) {
        std::uint32_t low = request.maximum_context_tokens;
        std::uint32_t high = ceiling;
        while (low < high) {
            const auto middle = low + (high - low + 1U) / 2U;
            std::vector<std::string> probe_notes;
            auto probe_inventory = build_inventory(checkpoints, request, middle,
                                                   hardware, probe_notes);
            bool admitted = probe_inventory.ok();
            if (admitted) {
                PlacementRequest probe_request(request);
                probe_request.maximum_context_tokens = middle;
                const auto probe = solve_placement(probe_inventory.value,
                                                   hardware, probe_request);
                admitted = probe.ok() && probe.value.fits;
            }
            if (admitted) low = middle; else high = middle - 1U;
        }
        result.value.maximum_context_tokens_that_fit = low;
    } else {
        result.value.notes.emplace_back(
            "no context ceiling is reported: this plan already spills, so more "
            "context trades cache residency rather than crossing a threshold");
    }
    for (auto& note : admission_notes) {
        result.value.fits = false;
        result.value.notes.push_back(std::move(note));
    }
    return result;
}


const PlacementPlannerRegistrar planner_registrar{&plan_model_placement_impl};

}  // namespace

}  // namespace strata
