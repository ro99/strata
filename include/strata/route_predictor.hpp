#pragma once

#include "strata/trace.hpp"
#include "strata/types.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace strata {

struct RoutePrediction {
    ExpertKey key;
    double confidence{};
    std::uint64_t evidence{};
};

/*
 * Online expert-transition model. It learns only from already observed routes;
 * no future trace information is consulted. The model is advisory and can
 * affect residency, never router output.
 */
class RoutePredictor {
public:
    void observe(const RouteEvent& event);

    [[nodiscard]] std::vector<RoutePrediction> predict(
        const RouteEvent& source, std::size_t limit, double minimum_confidence) const;

    [[nodiscard]] std::uint64_t transitions_observed() const noexcept {
        return transitions_observed_;
    }

private:
    struct PreviousEvent {
        RouteEvent event;
        bool valid{};
    };

    using DestinationCounts = std::unordered_map<ExpertKey, std::uint64_t, ExpertKeyHash>;
    std::unordered_map<ExpertKey, DestinationCounts, ExpertKeyHash> transitions_;
    std::unordered_map<std::uint64_t, PreviousEvent> previous_by_request_;
    std::uint64_t transitions_observed_{};
};

}  // namespace strata
