#pragma once

#include "strata/result.hpp"
#include "strata/chat_protocol.hpp"
#include "strata/types.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strata {

class StopSequenceBuffer {
public:
    explicit StopSequenceBuffer(std::span<const std::string> stops)
        : stops_(stops) {}

    void append(std::uint32_t token, std::string_view piece,
                const TokenStreamCallback& callback);
    void finish(const TokenStreamCallback& callback);

    [[nodiscard]] bool stopped() const noexcept { return stopped_; }
    [[nodiscard]] const std::string& text() const noexcept { return text_; }

private:
    void emit(std::size_t end, std::uint32_t token,
              const TokenStreamCallback& callback);

    std::span<const std::string> stops_;
    std::string text_;
    std::size_t emitted_{};
    bool stopped_{};
};

struct RuntimeDeviceBudget {
    std::uint64_t fractional_budget_bytes{};
    std::uint64_t explicit_budget_bytes{};
    std::uint64_t applied_budget_bytes{};
};

struct RuntimeDevicePlan {
    std::vector<int> devices;
    std::vector<std::uint64_t> budgets;
    std::vector<std::uint64_t> fractional_budgets;
    std::vector<std::uint64_t> explicit_budgets;
    std::vector<std::uint64_t> weight_capacities;
    std::vector<std::size_t> weighted_schedule;
};

using RuntimeDevicePlanResult = ParseResult<RuntimeDevicePlan>;

[[nodiscard]] ValidationResult validate_common_runtime_config(
    std::span<const int> devices, double vram_fraction,
    double sampling_temperature, std::string_view model_label);

[[nodiscard]] RuntimeDeviceBudget compute_runtime_device_budget(
    std::uint64_t free_bytes, double vram_fraction,
    std::uint64_t explicit_budget_bytes) noexcept;

[[nodiscard]] std::string_view runtime_budget_bound_name(
    std::uint64_t fractional_budget_bytes,
    std::uint64_t explicit_budget_bytes) noexcept;

[[nodiscard]] RuntimeDevicePlanResult plan_runtime_devices(
    std::span<const int> devices, double vram_fraction,
    std::uint64_t per_device_workspace_reserve,
    std::uint64_t minimum_device_budget,
    std::string_view model_label,
    std::uint64_t explicit_device_budget_bytes = 0U);

[[nodiscard]] std::uint64_t process_resident_set_bytes() noexcept;
[[nodiscard]] std::vector<std::uint64_t> device_vram_used_bytes(
    std::span<const int> devices);

[[nodiscard]] std::size_t incremental_kv_prefix_tokens(
    std::span<const std::uint32_t> cached_tokens,
    std::span<const std::uint32_t> prompt_tokens) noexcept;

[[nodiscard]] bool trim_oldest_chat_turn(
    std::vector<ChatMessage>& messages);

// Returns the complete valid UTF-8 prefix, or string_view::npos for invalid input.
[[nodiscard]] std::size_t complete_utf8_prefix(std::string_view text) noexcept;

}  // namespace strata
