#include "strata/simulator.hpp"

#include "strata/route_predictor.hpp"

namespace strata {

SimulationResult simulate(const std::vector<RouteEvent>& events,
                          const SimulationConfig& config) {
    ResidencyManager residency(config.residency);
    RoutePredictor predictor;
    std::uint64_t tick = 0;

    for (const auto& event : events) {
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
    }

    residency.finalize();

    return SimulationResult{residency.stats(), events.size(),
                            predictor.transitions_observed()};
}

}  // namespace strata
