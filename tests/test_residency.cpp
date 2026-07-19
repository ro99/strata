#include "test.hpp"

#include "strata/residency.hpp"
#include "strata/simulator.hpp"

TEST_CASE("residency promotes RAM hits into VRAM") {
    strata::ResidencyConfig config;
    config.expert_bytes = 10;
    config.vram_capacity_bytes = 10;
    config.ram_capacity_bytes = 20;
    config.policy = strata::ReplacementPolicy::Lru;
    strata::ResidencyManager manager(config);

    REQUIRE(manager.access({0, 1}, 1).tier == strata::Tier::Nvme);
    REQUIRE(manager.access({0, 2}, 2).tier == strata::Tier::Nvme);
    REQUIRE(manager.access({0, 1}, 3).tier == strata::Tier::Ram);
    REQUIRE(manager.access({0, 1}, 4).tier == strata::Tier::Vram);
    REQUIRE(manager.stats().nvme_read_bytes == 20);
    REQUIRE(manager.stats().weight_h2d_bytes == 30);
}

TEST_CASE("prefetched expert is accounted useful on demand") {
    strata::ResidencyConfig config;
    config.expert_bytes = 10;
    config.ram_capacity_bytes = 20;
    config.lease_ticks = 4;
    config.policy = strata::ReplacementPolicy::Lease;
    strata::ResidencyManager manager(config);

    REQUIRE(manager.prefetch({2, 7}, 1, 0.9));
    REQUIRE(manager.access({2, 7}, 2).tier == strata::Tier::Ram);
    REQUIRE(manager.stats().prefetches == 1);
    REQUIRE(manager.stats().useful_prefetches == 1);
    REQUIRE(manager.stats().nvme_prefetch_bytes == 10);
}

TEST_CASE("unused resident prefetch is wasted at finalization") {
    strata::ResidencyConfig config;
    config.expert_bytes = 10;
    config.ram_capacity_bytes = 20;
    strata::ResidencyManager manager(config);
    REQUIRE(manager.prefetch({2, 7}, 1, 0.9));
    manager.finalize();
    manager.finalize();
    REQUIRE(manager.stats().wasted_prefetches == 1);
}

TEST_CASE("strict cold-read budget reports refusal") {
    strata::ResidencyConfig config;
    config.expert_bytes = 10;
    config.ram_capacity_bytes = 10;
    config.cold_read_budget_bytes = 10;
    config.strict_cold_read_budget = true;
    strata::ResidencyManager manager(config);
    REQUIRE(!manager.access({0, 1}, 1).budget_exceeded);
    REQUIRE(manager.access({0, 2}, 2).budget_exceeded);
    REQUIRE(manager.stats().nvme_read_bytes == 10);
    REQUIRE(manager.stats().cold_budget_violations == 1);
}

TEST_CASE("simulator learns transitions and accounts all routes") {
    const std::vector<strata::RouteEvent> events{
        {0, 0, 3, {1}, {}}, {0, 0, 4, {2}, {}},
        {0, 1, 3, {1}, {}}, {0, 1, 4, {2}, {}},
    };
    strata::SimulationConfig config;
    config.residency.expert_bytes = 10;
    config.residency.ram_capacity_bytes = 20;
    config.residency.policy = strata::ReplacementPolicy::Lease;
    config.prefetch_limit = 2;
    config.minimum_prediction_confidence = 0.5;
    const auto result = strata::simulate(events, config);
    REQUIRE(result.events == 4);
    REQUIRE(result.residency.accesses == 4);
    REQUIRE(result.transitions_learned > 0);
}
