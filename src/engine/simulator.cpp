#include "strata/engine/simulator.hpp"

#include "strata/engine/route_predictor.hpp"

namespace strata {

SimulationResult simulate(const std::vector<RouteEvent>& events,
                          const SimulationConfig& config) {
    ResidencyManager residency(config.residency);
    RoutePredictor predictor;
    std::uint64_t tick = 0;
    std::uint64_t measured_events = 0U;
    bool measuring = config.measured_phase == RoutePhase::Unknown;

    for (const auto& event : events) {
        if (!measuring && event.phase == config.measured_phase) {
            residency.reset_stats();
            measuring = true;
        }
        predictor.observe(event);
        for (const auto expert : event.experts) {
            ++tick;
            (void)residency.access(ExpertKey{event.layer, expert}, tick);
        }
        if (config.prefetch_limit > 0) {
            const auto predictions = predictor.predict(
                event, config.prefetch_limit, config.minimum_prediction_confidence);
            for (const auto& prediction : predictions) {
                (void)residency.prefetch(prediction.key, tick, prediction.confidence);
            }
        }
        if (measuring) ++measured_events;
    }

    if (!measuring) residency.reset_stats();

    residency.finalize();

    return SimulationResult{residency.stats(), measured_events,
                            predictor.transitions_observed()};
}

}  // namespace strata
