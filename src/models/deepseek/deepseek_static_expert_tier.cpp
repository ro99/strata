#include "strata/models/deepseek/deepseek_static_expert_tier.hpp"

#include "strata/models/common/model_adapter.hpp"

#include <cstdio>
#include <string>
#include <utility>

namespace strata {
namespace {

constexpr std::size_t kHidden =
    static_cast<std::size_t>(kDeepSeekV4ExecutionContract.hidden_size);
constexpr std::size_t kIntermediate = static_cast<std::size_t>(
    kDeepSeekV4ExecutionContract.expert_intermediate_size);
constexpr std::uint32_t kLayers = kDeepSeekV4ExecutionContract.layer_count;
constexpr std::uint32_t kExperts = kDeepSeekV4ExecutionContract.routed_experts;
// w1/w3 packed + scales and w2 packed + scales, FP4 with one E8M0 per 32.
constexpr std::uint64_t kTripletBytes = 13369344ULL;

[[nodiscard]] std::string expert_prefix(std::uint32_t layer,
                                        std::uint32_t expert) {
    return "layers." + std::to_string(layer) + ".ffn.experts." +
           std::to_string(expert) + ".";
}

}  // namespace

ValidationResult Dsv4StaticExpertTier::initialize(
    int device, CudaBackend& backend, const Dsv4CheckpointReader& checkpoint,
    Dsv4ExpertResidencyPlan plan, std::uint64_t vram_budget_bytes,
    std::size_t slice_offset, std::size_t slice_stride) {
    ValidationResult result;
    if (active_) {
        result.errors.emplace_back("DeepSeek expert tier is already initialized");
        return result;
    }

    // Take this device's disjoint share first, then cut to what fits. Slicing
    // before truncating keeps each device's share spread across the ranking
    // rather than giving one device the whole hot head.
    plan.slice(slice_offset, slice_stride);
    plan.truncate(static_cast<std::size_t>(vram_budget_bytes / kTripletBytes));
    device_ = device;
    if (plan.empty()) {
        // Legal and quiet: this is exactly today's behaviour.
        plan_ = std::move(plan);
        return result;
    }

    auto reserved = backend.dsv4_tier_reserve(device, kLayers, kExperts);
    if (!reserved.ok()) {
        result.errors = std::move(reserved.errors);
        return result;
    }

    weights_.resize(plan.size() * 3U);
    std::size_t slot = 0U;
    for (const auto& [layer, expert] : plan.pairs()) {
        const auto prefix = expert_prefix(layer, expert);
        auto& w1 = weights_[slot * 3U];
        auto& w3 = weights_[slot * 3U + 1U];
        auto& w2 = weights_[slot * 3U + 2U];
        const auto load = [&](const char* suffix, std::uint64_t rows,
                              std::uint64_t columns, CudaWeight& into) {
            // Straight from the checkpoint: in the host-routed configuration
            // the resident store holds the tiled decode layout, which is not
            // the shape a device weight wants.
            auto loaded = load_dsv4_cuda_linear(
                checkpoint, nullptr, prefix + suffix, rows, columns, device,
                backend, into, false);
            if (!loaded.ok()) {
                for (auto& error : loaded.errors) {
                    result.errors.emplace_back(
                        "expert tier " + prefix + suffix + ": " + error);
                }
                return false;
            }
            return true;
        };
        if (!load("w1", kIntermediate, kHidden, w1) ||
            !load("w3", kIntermediate, kHidden, w3) ||
            !load("w2", kHidden, kIntermediate, w2)) {
            weights_.clear();
            return result;
        }
        auto added = backend.dsv4_tier_add(device, layer, expert, w1, w3, w2);
        if (!added.ok()) {
            result.errors = std::move(added.errors);
            weights_.clear();
            return result;
        }
        ++slot;
    }

    auto committed = backend.dsv4_tier_commit(device);
    if (!committed.ok()) {
        result.errors = std::move(committed.errors);
        weights_.clear();
        return result;
    }

    admitted_ = plan.size();
    bytes_ = plan.bytes(kTripletBytes);
    // The tier build was previously silent, which left "how big is the tier
    // that was actually built" unanswerable from a run: experiment 0127 could
    // compute 802 pairs from the plan and the truncation math but could not
    // observe them, so a partial build and a full one looked identical.
    std::fprintf(stderr,
                 "[deepseek-tier] device=%d rank=%zu/%zu pairs=%zu bytes=%llu "
                 "(%.2f GiB)\n",
                 device, slice_offset, slice_stride, admitted_,
                 static_cast<unsigned long long>(bytes_),
                 static_cast<double>(bytes_) / static_cast<double>(1ULL << 30U));
    plan_ = std::move(plan);
    active_ = true;
    return result;
}

}  // namespace strata
