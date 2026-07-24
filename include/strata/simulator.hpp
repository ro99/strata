#pragma once

#include "strata/residency.hpp"
#include "strata/trace.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata {

/*
 * Offline oracle for the shadow-speculative window described in
 * research/shadow-speculative-moe-offload.md.
 *
 * A window drafts `window_tokens` (gamma) tokens and verifies gamma + 1 rows:
 * the row for the last accepted token plus one row per draft. Accepting j
 * drafts advances j + 1 tokens, so a fully accepted window costs gamma + 1
 * verify rows and advances gamma + 1 positions.
 *
 * The oracle replays a recorded route trace, so every verify row inside a
 * window is charged the route that position took in the *accepted* sequence.
 * A real draft mispredicts, which perturbs the routes of the rows it will not
 * keep. The oracle therefore reports a strict UPPER BOUND on achievable
 * dedup and speedup, and `break_even_acceptance` is a LOWER BOUND on the
 * acceptance rate a real implementation needs. It cannot answer whether the
 * resident-shadow draft agrees with the target router; only the runtime probe
 * can.
 */
struct WindowOracleConfig {
    // Drafted tokens per window (gamma). Zero keeps the sequential path.
    std::uint32_t window_tokens{};
    // Fraction of drafts accepted; advance = floor(acceptance_rate * gamma) + 1.
    double acceptance_rate{1.0};
    // Issue the window's cold union as a coalesced prefetch before replay.
    bool prefetch_window_union{true};
    // Unknown replays every phase; Decode/Prefill filter the trace first.
    RoutePhase phase{RoutePhase::Unknown};
    // Cost model, milliseconds. All zero disables the projection.
    double base_step_ms{};
    double draft_token_ms{};
    double verify_row_ms{};
    double expert_acquire_ms{};
    double h2d_gigabytes_per_second{};
};

struct WindowOracleStats {
    bool enabled{};
    std::uint64_t steps{};
    std::uint64_t windows{};
    std::uint64_t verify_rows{};
    std::uint64_t window_accesses{};
    std::uint64_t window_unique_experts{};
    std::uint64_t window_cold_accesses{};
    std::uint64_t window_cold_unique_experts{};
    std::uint64_t window_read_bytes{};
    std::uint64_t advanced_tokens{};
    std::uint64_t baseline_steps{};
    std::uint64_t baseline_cold_accesses{};
    std::uint64_t baseline_read_bytes{};
    double dedup_factor{};
    double cold_dedup_factor{};
    double break_even_acceptance{};
    double projected_baseline_ms{};
    double projected_window_ms{};
    double projected_speedup{};
};

struct SimulationConfig {
    ResidencyConfig residency;
    std::size_t prefetch_limit{};
    double minimum_prediction_confidence{0.50};
    // Window mode ignores prefetch_limit: the manifest replaces the predictor.
    WindowOracleConfig window;
};

struct SimulationResult {
    ResidencyStats residency;
    std::uint64_t events{};
    std::uint64_t transitions_learned{};
    WindowOracleStats window;
};

[[nodiscard]] SimulationResult simulate(const std::vector<RouteEvent>& events,
                                        const SimulationConfig& config);

}  // namespace strata
