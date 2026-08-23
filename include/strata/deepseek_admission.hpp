#pragma once

#include "strata/deepseek_manifest.hpp"

#include "strata/result.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <string>
#include <vector>

namespace strata {

// E8M0 scale admission for FP4 regions.
//
// Both FP4 decoders map the E8M0 scale code straight into a BF16 exponent
// field. That is exact for codes 1-254 and silently wrong outside it: code 0
// means 2^-127, which is subnormal in BF16 whose smallest normal is 2^-126 and
// encodes as +0, and code 255 is the E8M0 NaN encoding and encodes as +inf.
// Exact mode must execute an approved route or report failure, never
// substitute, so a region carrying either code fails admission at load rather
// than producing a wrong value at decode.
//
// Measured against the real DeepSeek V4 checkpoint in experiment 0156:
// 1,409,286,144 scale bytes span codes [119, 125], entirely inside the window,
// so this is a guard against a malformed or differently quantized checkpoint
// rather than a constraint on the current one.
inline constexpr std::uint8_t kDsv4E8m0AdmissibleMinimum = 1U;
inline constexpr std::uint8_t kDsv4E8m0AdmissibleMaximum = 254U;

struct Dsv4E8m0Admission {
    std::uint64_t inadmissible{};
    std::uint64_t code_zero{};
    std::uint64_t code_255{};
    std::uint64_t first_offset{};
    std::uint8_t first_code{};
    [[nodiscard]] bool admitted() const noexcept { return inadmissible == 0U; }
};

[[nodiscard]] Dsv4E8m0Admission dsv4_admit_e8m0_scales(
    std::span<const std::byte> scales) noexcept;

// Convenience wrapper that turns a rejection into a ValidationResult naming the
// tensor, the offending code and its byte offset.
[[nodiscard]] ValidationResult dsv4_admit_e8m0_scales_for(
    std::string_view tensor_name, std::span<const std::byte> scales);

struct Dsv4AdmissionConfig {
    std::uint64_t host_memory_ceiling_bytes{};
    std::vector<std::uint64_t> vram_weight_budgets;
    std::uint64_t host_kv_cache_bytes{};
    std::vector<std::uint64_t> device_kv_cache_bytes;
    std::uint32_t maximum_context_tokens{2048U};
    // A batched prefill page pins its own sliding history: the page's
    // retention floor holds every row from position_base + 1 - kWindow until
    // the page's attend loop completes, so the sliding table carries
    // page_tokens + kWindow rows, not kWindow. Budgeting for the window alone
    // exhausts the host KV cache on the first page wider than a block.
    std::uint32_t prefill_page_tokens{1U};
    bool enable_dspark{};
    bool enable_mhc_prepack{};
    bool host_routed_experts{};
    bool compact_kv_cache{};
    bool physical_kv_cache{};
    bool device_resident_mhc{};
    bool require_zero_nvme_decode{true};
};

struct Dsv4MemoryPlan {
    std::uint64_t routed_expert_host_bytes{};
    std::uint64_t host_parameter_bytes{};
    std::uint64_t kv_state_bytes{};
    std::uint64_t index_state_bytes{};
    std::uint64_t kv_cache_payload_bytes{};
    std::uint64_t kv_cache_metadata_bytes{};
    std::uint64_t kv_cache_alignment_bytes{};
    std::uint64_t host_kv_cache_bytes{};
    std::uint64_t device_kv_cache_bytes{};
    std::vector<std::uint64_t> per_device_kv_cache_bytes;
    std::uint64_t host_workspace_bytes{};
    std::uint64_t mhc_prepack_bytes{};
    std::uint64_t mhc_device_bytes{};
    std::uint64_t required_host_bytes{};
    std::uint64_t resident_spine_vram_bytes{};
    std::uint64_t vram_workspace_bytes{};
    std::uint64_t total_vram_budget_bytes{};
    std::uint64_t expert_vram_cache_bytes{};
    std::uint64_t maximum_expert_bytes{};
    std::vector<std::uint64_t> fractional_vram_budget_bytes;
    std::vector<std::uint64_t> explicit_vram_budget_bytes;
    std::vector<std::uint64_t> applied_vram_budget_bytes;
    std::vector<std::string> vram_budget_bound;
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
