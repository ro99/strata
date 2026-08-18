#include "strata/dsv4_static_expert_tier.hpp"

#include "strata/model_adapter.hpp"

#include <chrono>
#include <cmath>
#include <string>
#include <utility>

namespace strata {
namespace {

constexpr std::size_t kHidden =
    static_cast<std::size_t>(kDeepSeekV4ExecutionContract.hidden_size);
constexpr std::size_t kIntermediate =
    static_cast<std::size_t>(kDeepSeekV4ExecutionContract.expert_intermediate_size);
// w1 and w3 are [intermediate, hidden]; w2 is [hidden, intermediate]. FP4 with
// one E8M0 scale per 32 values, which is what the triplet byte count encodes.
constexpr std::uint64_t kTripletBytes = 13369344ULL;

[[nodiscard]] std::uint64_t key_of(std::uint32_t layer,
                                   std::uint32_t expert) noexcept {
    return (static_cast<std::uint64_t>(layer) << 32U) | expert;
}

[[nodiscard]] std::string expert_prefix(std::uint32_t layer,
                                        std::uint32_t expert) {
    return "layers." + std::to_string(layer) + ".ffn.experts." +
           std::to_string(expert) + ".";
}

[[nodiscard]] std::uint64_t elapsed_ns(
    std::chrono::steady_clock::time_point started) noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count());
}

}  // namespace

Dsv4StaticExpertTier::~Dsv4StaticExpertTier() { shutdown(); }

void Dsv4StaticExpertTier::shutdown() {
    if (!worker_.joinable()) return;
    {
        std::lock_guard<std::mutex> guard(mutex_);
        stop_ = true;
    }
    request_.notify_all();
    worker_.join();
}

ValidationResult Dsv4StaticExpertTier::initialize(
    int device, CudaBackend& backend, const Dsv4CheckpointReader& checkpoint,
    Dsv4ExpertResidencyPlan plan, std::uint64_t vram_budget_bytes,
    float swiglu_limit) {
    ValidationResult result;
    if (active_ || worker_.joinable()) {
        result.errors.emplace_back(
            "DeepSeek static expert tier is already initialized");
        return result;
    }
    if (swiglu_limit <= 0.0F) {
        result.errors.emplace_back(
            "DeepSeek static expert tier needs a positive SwiGLU limit");
        return result;
    }

    // Admit only what fits. Truncating keeps the hottest, which is the order
    // the planner emits, so a tight budget degrades gracefully instead of
    // failing the run.
    const auto capacity =
        static_cast<std::size_t>(vram_budget_bytes / kTripletBytes);
    plan.truncate(capacity);
    if (plan.empty()) {
        // A tier that admits nothing is exactly today's behaviour. Report it
        // and stay inactive rather than treating it as an error.
        device_ = device;
        backend_ = &backend;
        swiglu_limit_ = swiglu_limit;
        plan_ = std::move(plan);
        return result;
    }

    triplets_.reserve(plan.size());
    for (const auto& [layer, expert] : plan.pairs()) {
        const auto prefix = expert_prefix(layer, expert);
        Triplet triplet;
        // Loaded straight from the checkpoint rather than from the resident
        // store: in the host-routed configuration that store holds the tiled
        // decode layout, which is not the shape a device weight wants.
        const auto load = [&](const char* suffix, std::uint64_t rows,
                              std::uint64_t columns, CudaWeight& into) {
            auto loaded = load_dsv4_cuda_linear(
                checkpoint, nullptr, prefix + suffix, rows, columns, device,
                backend, into, false);
            if (!loaded.ok()) {
                for (auto& error : loaded.errors) {
                    result.errors.emplace_back(
                        "static expert tier " + prefix + suffix + ": " + error);
                }
                return false;
            }
            return true;
        };
        if (!load("w1", kIntermediate, kHidden, triplet.w1) ||
            !load("w3", kIntermediate, kHidden, triplet.w3) ||
            !load("w2", kHidden, kIntermediate, triplet.w2)) {
            triplets_.clear();
            return result;
        }
        triplets_.emplace(key_of(layer, expert), std::move(triplet));
    }

    device_ = device;
    backend_ = &backend;
    swiglu_limit_ = swiglu_limit;
    admitted_ = plan.size();
    bytes_ = plan.bytes(kTripletBytes);
    plan_ = std::move(plan);
    request_input_.assign(kHidden, 0.0F);
    response_output_.assign(kHidden, 0.0F);
    request_experts_.reserve(kDeepSeekV4ExecutionContract.experts_per_token);
    active_ = true;
    worker_ = std::thread([this] { worker_loop(); });
    return result;
}

std::size_t Dsv4StaticExpertTier::select(
    std::uint32_t layer, std::span<const std::uint32_t> experts,
    std::span<bool> serve) const noexcept {
    std::size_t count = 0U;
    for (std::size_t slot = 0U; slot < serve.size(); ++slot) serve[slot] = false;
    if (!active_ || experts.size() != serve.size()) return 0U;
    for (std::size_t slot = 0U; slot < experts.size(); ++slot) {
        if (!plan_.resident(layer, experts[slot])) continue;
        // A route may name the same expert twice only if the router allows it;
        // serving both slots is still correct because each carries its own
        // coefficient and the sum is over slots, not over distinct experts.
        serve[slot] = true;
        ++count;
    }
    return count;
}

ValidationResult Dsv4StaticExpertTier::submit(
    std::uint32_t layer, std::span<const std::uint32_t> experts,
    std::span<const float> weights, std::span<const bool> serve,
    std::span<const float> input) {
    ValidationResult result;
    if (!active_) {
        result.errors.emplace_back(
            "DeepSeek static expert tier submit on an inactive tier");
        return result;
    }
    if (experts.size() != weights.size() || experts.size() != serve.size() ||
        input.size() != kHidden) {
        result.errors.emplace_back(
            "DeepSeek static expert tier submit has incompatible spans");
        return result;
    }
    std::unique_lock<std::mutex> guard(mutex_);
    if (pending_) {
        result.errors.emplace_back(
            "DeepSeek static expert tier already has work in flight");
        return result;
    }
    request_experts_.clear();
    request_weights_.clear();
    for (std::size_t slot = 0U; slot < experts.size(); ++slot) {
        if (!serve[slot]) continue;
        const auto found = triplets_.find(key_of(layer, experts[slot]));
        if (found == triplets_.end()) {
            // The residency map and the loaded set must agree exactly; a miss
            // here means one of them is stale and the expert would be dropped.
            result.errors.emplace_back(
                "DeepSeek static expert tier is missing layer " +
                std::to_string(layer) + " expert " +
                std::to_string(experts[slot]) + " it claims to hold");
            return result;
        }
        CudaDeepSeekMoeExpert descriptor;
        descriptor.w1 = &found->second.w1;
        descriptor.w3 = &found->second.w3;
        descriptor.w2 = &found->second.w2;
        // The command returns one unsummed row per routed expert, so the
        // route coefficient is applied here on the join rather than handed to
        // the kernel. Passing 1.0 makes the device output the bare expert
        // whether or not the backend folds the coefficient in, which removes
        // the ambiguity rather than depending on it.
        descriptor.coefficient = 1.0F;
        request_experts_.push_back(descriptor);
        request_weights_.push_back(weights[slot]);
    }
    if (request_experts_.empty()) {
        result.errors.emplace_back(
            "DeepSeek static expert tier submit selected no experts");
        return result;
    }
    request_input_.assign(input.begin(), input.end());
    pending_ = true;
    complete_ = false;
    worker_errors_.clear();
    guard.unlock();
    request_.notify_one();
    return result;
}

ValidationResult Dsv4StaticExpertTier::collect(std::span<float> destination) {
    ValidationResult result;
    if (destination.size() != kHidden) {
        result.errors.emplace_back(
            "DeepSeek static expert tier collect has the wrong width");
        return result;
    }
    const auto waited = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> guard(mutex_);
    response_.wait(guard, [this] { return complete_ || stop_; });
    wait_nanoseconds_.fetch_add(elapsed_ns(waited), std::memory_order_relaxed);
    if (!worker_errors_.empty()) {
        for (auto& error : worker_errors_) result.errors.push_back(error);
        worker_errors_.clear();
        pending_ = false;
        complete_ = false;
        // One failure disables the tier: continuing would leave some layers
        // served and others not, which is worse than serving none.
        active_ = false;
        return result;
    }
    for (std::size_t index = 0U; index < kHidden; ++index) {
        destination[index] += response_output_[index];
    }
    pending_ = false;
    complete_ = false;
    return result;
}

void Dsv4StaticExpertTier::worker_loop() {
    for (;;) {
        std::unique_lock<std::mutex> guard(mutex_);
        request_.wait(guard, [this] { return pending_ || stop_; });
        if (stop_) return;
        auto experts = request_experts_;
        auto expert_weights = request_weights_;
        auto input = request_input_;
        guard.unlock();

        std::vector<float> routed(experts.size() * kHidden, 0.0F);
        std::vector<std::string> errors;
        const auto started = std::chrono::steady_clock::now();
        auto enqueued = backend_->enqueue_deepseek_moe(
            device_, input, experts, nullptr, swiglu_limit_);
        if (!enqueued.ok()) {
            errors = std::move(enqueued.errors);
        } else {
            auto collected = backend_->collect_deepseek_moe(
                device_, routed, {});
            if (!collected.ok()) errors = std::move(collected.errors);
        }
        const auto elapsed = elapsed_ns(started);

        std::vector<float> summed(kHidden, 0.0F);
        if (errors.empty()) {
            for (std::size_t index = 0U; index < expert_weights.size(); ++index) {
                const auto* row = routed.data() + index * kHidden;
                const auto weight = expert_weights[index];
                for (std::size_t column = 0U; column < kHidden; ++column) {
                    summed[column] = std::fma(row[column], weight, summed[column]);
                }
            }
        }

        guard.lock();
        if (errors.empty()) {
            response_output_ = std::move(summed);
            submissions_.fetch_add(1U, std::memory_order_relaxed);
            experts_served_.fetch_add(experts.size(), std::memory_order_relaxed);
            device_nanoseconds_.fetch_add(elapsed, std::memory_order_relaxed);
        } else {
            worker_errors_ = std::move(errors);
        }
        complete_ = true;
        guard.unlock();
        response_.notify_one();
    }
}

Dsv4StaticExpertTier::Stats Dsv4StaticExpertTier::stats() const noexcept {
    Stats stats;
    stats.submissions = submissions_.load(std::memory_order_relaxed);
    stats.experts_served = experts_served_.load(std::memory_order_relaxed);
    stats.device_nanoseconds = device_nanoseconds_.load(std::memory_order_relaxed);
    stats.wait_nanoseconds = wait_nanoseconds_.load(std::memory_order_relaxed);
    return stats;
}

}  // namespace strata
