#include "strata/residency.hpp"
#include "strata/simulator.hpp"
#include "strata/trace.hpp"
#include "test.hpp"

#include <cstdint>
#include <vector>

namespace {

strata::RouteEvent make_event(std::uint64_t request, std::uint64_t position,
                              std::uint32_t layer, std::vector<std::uint32_t> experts,
                              strata::RoutePhase phase = strata::RoutePhase::Decode) {
    strata::RouteEvent event;
    event.request = request;
    event.token_position = position;
    event.layer = layer;
    event.experts = std::move(experts);
    event.coefficients.assign(event.experts.size(), 1.0F);
    event.phase = phase;
    return event;
}

strata::SimulationConfig base_config() {
    strata::SimulationConfig config;
    config.residency.vram_capacity_bytes = 64U * 1024U;
    config.residency.ram_capacity_bytes = 1024U * 1024U;
    config.residency.expert_bytes = 1024U;
    config.residency.lease_ticks = 8;
    config.residency.policy = strata::ReplacementPolicy::Lease;
    return config;
}

// One route event per (position, layer); every step touches `per_step` layers.
std::vector<strata::RouteEvent> single_expert_per_step(std::uint32_t steps,
                                                       std::uint32_t stride) {
    std::vector<strata::RouteEvent> events;
    for (std::uint32_t step = 0; step < steps; ++step) {
        events.push_back(make_event(1U, step, 0U, {step * stride}));
    }
    return events;
}

}  // namespace

TEST_CASE("window oracle disabled reproduces the sequential replay") {
    const auto events = single_expert_per_step(24U, 1U);
    auto config = base_config();
    const auto result = strata::simulate(events, config);

    strata::ResidencyManager reference(config.residency);
    std::uint64_t tick = 0;
    for (const auto& event : events) {
        for (const auto expert : event.experts) {
            ++tick;
            (void)reference.access(strata::ExpertKey{event.layer, expert}, tick);
        }
    }
    reference.finalize();
    const auto& expected = reference.stats();
    const auto& actual = result.residency;

    REQUIRE(!result.window.enabled);
    REQUIRE(actual.accesses == expected.accesses);
    REQUIRE(actual.vram_hits == expected.vram_hits);
    REQUIRE(actual.ram_hits == expected.ram_hits);
    REQUIRE(actual.peer_hits == expected.peer_hits);
    REQUIRE(actual.nvme_misses == expected.nvme_misses);
    REQUIRE(actual.nvme_read_bytes == expected.nvme_read_bytes);
    REQUIRE(actual.nvme_prefetch_bytes == expected.nvme_prefetch_bytes);
    REQUIRE(actual.weight_h2d_bytes == expected.weight_h2d_bytes);
    REQUIRE(actual.peer_activation_bytes == expected.peer_activation_bytes);
    REQUIRE(actual.prefetches == expected.prefetches);
    REQUIRE(actual.useful_prefetches == expected.useful_prefetches);
    REQUIRE(actual.wasted_prefetches == expected.wasted_prefetches);
    REQUIRE(actual.vram_evictions == expected.vram_evictions);
    REQUIRE(actual.ram_evictions == expected.ram_evictions);
    REQUIRE(actual.cold_budget_violations == expected.cold_budget_violations);
}

TEST_CASE("disjoint steps give unit window dedup") {
    const auto events = single_expert_per_step(8U, 1U);
    auto config = base_config();
    config.window.window_tokens = 1U;
    config.window.acceptance_rate = 1.0;

    const auto result = strata::simulate(events, config);
    const auto& window = result.window;
    REQUIRE(window.enabled);
    REQUIRE(window.steps == 8U);
    REQUIRE(window.windows == 4U);
    REQUIRE(window.verify_rows == 8U);
    REQUIRE(window.advanced_tokens == 8U);
    REQUIRE(window.window_accesses == 8U);
    REQUIRE(window.window_unique_experts == 8U);
    REQUIRE_NEAR(window.dedup_factor, 1.0, 1e-12);
    REQUIRE_NEAR(window.break_even_acceptance, 1.0, 1e-12);
}

TEST_CASE("repeated experts collapse to one manifest entry") {
    std::vector<strata::RouteEvent> events;
    for (std::uint32_t step = 0; step < 4U; ++step) {
        events.push_back(make_event(1U, step, 0U, {7U}));
    }
    auto config = base_config();
    config.window.window_tokens = 3U;
    config.window.acceptance_rate = 1.0;

    const auto result = strata::simulate(events, config);
    const auto& window = result.window;
    REQUIRE(window.windows == 1U);
    REQUIRE(window.verify_rows == 4U);
    REQUIRE(window.window_accesses == 4U);
    REQUIRE(window.window_unique_experts == 1U);
    REQUIRE(window.window_cold_unique_experts == 1U);
    REQUIRE(window.window_cold_accesses == 4U);
    REQUIRE_NEAR(window.dedup_factor, 4.0, 1e-12);
    REQUIRE_NEAR(window.cold_dedup_factor, 4.0, 1e-12);
    REQUIRE_NEAR(window.break_even_acceptance, 0.25, 1e-12);
    REQUIRE(window.window_read_bytes == config.residency.expert_bytes);
}

TEST_CASE("manifest prefetch converts cold misses into useful prefetches") {
    const auto events = single_expert_per_step(16U, 1U);
    auto config = base_config();
    config.window.window_tokens = 3U;
    config.window.acceptance_rate = 1.0;

    const auto prefetched = strata::simulate(events, config);
    REQUIRE(prefetched.residency.prefetches == 16U);
    REQUIRE(prefetched.residency.useful_prefetches == 16U);
    REQUIRE(prefetched.residency.nvme_misses == 0U);

    config.window.prefetch_window_union = false;
    const auto demand = strata::simulate(events, config);
    REQUIRE(demand.residency.prefetches == 0U);
    REQUIRE(demand.residency.nvme_misses == 16U);
    REQUIRE(demand.window.window_accesses == prefetched.window.window_accesses);
}

TEST_CASE("a manifest larger than VRAM is bounded, not fatal") {
    const auto events = single_expert_per_step(32U, 1U);
    auto config = base_config();
    config.residency.vram_capacity_bytes = 2U * config.residency.expert_bytes;
    config.window.window_tokens = 15U;
    config.window.acceptance_rate = 1.0;

    const auto result = strata::simulate(events, config);
    REQUIRE(result.window.windows == 2U);
    REQUIRE(result.window.advanced_tokens == 32U);
    REQUIRE(result.window.window_unique_experts == 32U);
    REQUIRE(result.residency.vram_evictions > 0U);
    REQUIRE(result.residency.accesses == 32U);
}

TEST_CASE("partial acceptance advances only the accepted prefix") {
    const auto events = single_expert_per_step(4U, 1U);
    auto config = base_config();
    config.window.window_tokens = 3U;
    config.window.acceptance_rate = 0.5;  // floor(1.5) = 1 accepted, advance 2

    const auto result = strata::simulate(events, config);
    const auto& window = result.window;
    REQUIRE(window.windows == 2U);
    REQUIRE(window.advanced_tokens == 4U);
    // Window one verifies four rows, window two verifies the trailing two.
    REQUIRE(window.verify_rows == 6U);
    REQUIRE(window.verify_rows > window.steps);
}

TEST_CASE("window transfer never exceeds the demand baseline") {
    std::vector<strata::RouteEvent> events;
    for (std::uint32_t step = 0; step < 32U; ++step) {
        events.push_back(make_event(1U, step, 0U, {step % 4U}));
        events.push_back(make_event(1U, step, 1U, {step % 3U}));
    }
    auto config = base_config();
    config.window.window_tokens = 7U;
    config.window.acceptance_rate = 1.0;

    const auto result = strata::simulate(events, config);
    const auto& window = result.window;
    REQUIRE(window.steps == 32U);
    REQUIRE(window.baseline_steps == 32U);
    REQUIRE(window.window_accesses == 64U);
    REQUIRE(window.window_read_bytes <= window.baseline_read_bytes);
    REQUIRE(window.dedup_factor > 1.0);
}

TEST_CASE("the decode phase filter excludes prefill rows") {
    std::vector<strata::RouteEvent> events;
    for (std::uint32_t step = 0; step < 6U; ++step) {
        events.push_back(make_event(1U, step, 0U, {step}, strata::RoutePhase::Prefill));
    }
    for (std::uint32_t step = 6U; step < 10U; ++step) {
        events.push_back(make_event(1U, step, 0U, {step}, strata::RoutePhase::Decode));
    }
    auto config = base_config();
    config.window.window_tokens = 1U;
    config.window.acceptance_rate = 1.0;

    config.window.phase = strata::RoutePhase::Decode;
    const auto decode = strata::simulate(events, config);
    REQUIRE(decode.window.steps == 4U);
    REQUIRE(decode.window.window_accesses == 4U);

    config.window.phase = strata::RoutePhase::Unknown;
    const auto all = strata::simulate(events, config);
    REQUIRE(all.window.steps == 10U);
    REQUIRE(all.window.window_accesses == 10U);
}

TEST_CASE("the projected cost model is reported against the demand baseline") {
    std::vector<strata::RouteEvent> events;
    for (std::uint32_t step = 0; step < 16U; ++step) {
        events.push_back(make_event(1U, step, 0U, {step % 2U}));
    }
    auto config = base_config();
    config.residency.vram_capacity_bytes = config.residency.expert_bytes;  // thrash
    config.window.window_tokens = 3U;
    config.window.acceptance_rate = 1.0;
    config.window.base_step_ms = 100.0;
    config.window.draft_token_ms = 10.0;
    config.window.verify_row_ms = 10.0;
    config.window.expert_acquire_ms = 50.0;
    config.window.h2d_gigabytes_per_second = 0.0;  // fixed acquisition cost only

    const auto result = strata::simulate(events, config);
    const auto& window = result.window;
    REQUIRE(window.baseline_steps == 16U);
    REQUIRE(window.projected_baseline_ms > 0.0);
    REQUIRE(window.projected_window_ms > 0.0);
    REQUIRE(window.projected_speedup > 1.0);
}
