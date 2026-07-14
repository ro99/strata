#pragma once

#include "strata/residency.hpp"
#include "strata/trace.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata {

struct SimulationConfig {
    ResidencyConfig residency;
    std::size_t prefetch_limit{};
    double minimum_prediction_confidence{0.50};
};

struct SimulationResult {
    ResidencyStats residency;
    std::uint64_t events{};
    std::uint64_t transitions_learned{};
};

[[nodiscard]] SimulationResult simulate(const std::vector<RouteEvent>& events,
                                        const SimulationConfig& config);

}  // namespace strata
