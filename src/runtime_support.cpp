#include "strata/runtime_support.hpp"

#include "strata/cuda_backend.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace strata {

ValidationResult validate_common_runtime_config(
    std::span<const int> devices, double vram_fraction,
    double sampling_temperature, std::string_view model_label) {
    ValidationResult result;
    if (devices.empty()) {
        result.errors.emplace_back(std::string(model_label) +
                                   " runtime requires at least one CUDA device");
    }
    std::unordered_set<int> unique;
    for (const int device : devices) {
        if (device < 0 || !unique.insert(device).second) {
            result.errors.emplace_back(std::string(model_label) +
                                       " runtime devices must be unique non-negative ids");
            break;
        }
    }
    if (!std::isfinite(vram_fraction) || vram_fraction <= 0.0 ||
        vram_fraction > 0.95) {
        result.errors.emplace_back("VRAM cache fraction must be in (0, 0.95]");
    }
    if (!std::isfinite(sampling_temperature) || sampling_temperature < 0.0 ||
        sampling_temperature > 10.0) {
        result.errors.emplace_back(std::string(model_label) +
                                   " sampling temperature must be within [0, 10]");
    }
    return result;
}

RuntimeDevicePlanResult plan_runtime_devices(
    std::span<const int> devices, double vram_fraction,
    std::uint64_t per_device_workspace_reserve,
    std::uint64_t minimum_device_budget,
    std::string_view model_label) {
    RuntimeDevicePlanResult result;
    std::vector<std::uint64_t> totals;
    result.value.devices.assign(devices.begin(), devices.end());
    for (const int device : devices) {
        auto memory = CudaBackend::device_memory(device);
        if (!memory.ok()) {
            result.errors = std::move(memory.errors);
            return result;
        }
        const auto budget = static_cast<std::uint64_t>(
            static_cast<double>(memory.value.free_bytes) * vram_fraction);
        if (budget < minimum_device_budget ||
            budget <= per_device_workspace_reserve) {
            result.errors.emplace_back(
                std::string(model_label) + " CUDA device " +
                std::to_string(device) +
                " does not meet the admitted VRAM budget");
            return result;
        }
        result.value.budgets.push_back(budget);
        result.value.weight_capacities.push_back(
            budget - per_device_workspace_reserve);
        totals.push_back(memory.value.total_bytes);
    }
    const auto smallest = *std::min_element(totals.begin(), totals.end());
    if (smallest == 0U) {
        result.errors.emplace_back(std::string(model_label) +
                                   " CUDA device reports zero total memory");
        return result;
    }
    for (std::size_t slot = 0U; slot < totals.size(); ++slot) {
        if (totals[slot] > (std::numeric_limits<std::uint64_t>::max() -
                            smallest / 2U) / 2U) {
            result.errors.emplace_back("CUDA device schedule weight overflows");
            return result;
        }
        const auto shares = std::max<std::uint64_t>(
            1U, (totals[slot] * 2U + smallest / 2U) / smallest);
        for (std::uint64_t count = 0U; count < shares; ++count) {
            result.value.weighted_schedule.push_back(slot);
        }
    }
    return result;
}

std::size_t incremental_kv_prefix_tokens(
    std::span<const std::uint32_t> cached_tokens,
    std::span<const std::uint32_t> prompt_tokens) noexcept {
    if (cached_tokens.empty() || cached_tokens.size() >= prompt_tokens.size() ||
        !std::equal(cached_tokens.begin(), cached_tokens.end(),
                    prompt_tokens.begin())) {
        return 0U;
    }
    return cached_tokens.size();
}

}  // namespace strata
