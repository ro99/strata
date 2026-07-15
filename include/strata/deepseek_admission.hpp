#pragma once

#include "strata/deepseek_manifest.hpp"

#include <cstdint>
#include <vector>

namespace strata {

struct Dsv4AdmissionConfig {
    std::uint64_t host_memory_ceiling_bytes{};
    std::vector<std::uint64_t> vram_weight_budgets;
    std::uint32_t maximum_context_tokens{2048U};
    bool enable_dspark{};
    bool require_zero_nvme_decode{true};
};

struct Dsv4MemoryPlan {
    std::uint64_t routed_expert_host_bytes{};
    std::uint64_t host_parameter_bytes{};
    std::uint64_t kv_state_bytes{};
    std::uint64_t host_workspace_bytes{};
    std::uint64_t required_host_bytes{};
    std::uint64_t resident_spine_vram_bytes{};
    std::uint64_t vram_workspace_bytes{};
    std::uint64_t total_vram_budget_bytes{};
    std::uint64_t expert_vram_cache_bytes{};
    std::uint64_t maximum_expert_bytes{};
    std::uint64_t steady_state_nvme_bytes{};
    std::uint32_t maximum_context_tokens{};
    bool dspark_enabled{};
    bool zero_nvme_decode{};
};

struct Dsv4AdmissionResult {
    Dsv4MemoryPlan plan;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

[[nodiscard]] Dsv4AdmissionResult plan_dsv4_resident_topology(
    const Dsv4IndexManifest& manifest, const Dsv4AdmissionConfig& config);

}  // namespace strata
