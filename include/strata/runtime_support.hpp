#pragma once

#include "strata/result.hpp"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace strata {

struct RuntimeDevicePlan {
    std::vector<int> devices;
    std::vector<std::uint64_t> budgets;
    std::vector<std::uint64_t> weight_capacities;
    std::vector<std::size_t> weighted_schedule;
};

using RuntimeDevicePlanResult = ParseResult<RuntimeDevicePlan>;

[[nodiscard]] ValidationResult validate_common_runtime_config(
    std::span<const int> devices, double vram_fraction,
    double sampling_temperature, std::string_view model_label);

[[nodiscard]] RuntimeDevicePlanResult plan_runtime_devices(
    std::span<const int> devices, double vram_fraction,
    std::uint64_t per_device_workspace_reserve,
    std::uint64_t minimum_device_budget,
    std::string_view model_label);

[[nodiscard]] std::size_t incremental_kv_prefix_tokens(
    std::span<const std::uint32_t> cached_tokens,
    std::span<const std::uint32_t> prompt_tokens) noexcept;

}  // namespace strata
