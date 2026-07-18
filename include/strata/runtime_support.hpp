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

struct RuntimeDevicePlanResult {
    RuntimeDevicePlan value;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

[[nodiscard]] ValidationResult validate_common_runtime_config(
    std::span<const int> devices, double vram_fraction,
    double sampling_temperature, std::string_view model_label);

[[nodiscard]] RuntimeDevicePlanResult plan_runtime_devices(
    std::span<const int> devices, double vram_fraction,
    std::uint64_t per_device_workspace_reserve,
    std::uint64_t minimum_device_budget,
    std::string_view model_label);

}  // namespace strata
