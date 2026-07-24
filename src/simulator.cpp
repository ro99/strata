#include "strata/simulator.hpp"

#include "strata/route_predictor.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace strata {
namespace {

SimulationResult simulate_sequential(const std::vector<RouteEvent>& events,
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
                            predictor.transitions_observed(), WindowOracleStats{}};
}

// A decode step is one token position of one request. The experts consumed by
// that step are its route events concatenated layer-ascending, which is the
// order the runtime acquires them in.
std::vector<std::vector<ExpertKey>> build_steps(const std::vector<RouteEvent>& events,
                                                RoutePhase phase,
                                                RoutePredictor& predictor) {
    std::vector<const RouteEvent*> ordered;
    ordered.reserve(events.size());
    for (const auto& event : events) {
        if (phase != RoutePhase::Unknown && event.phase != phase) continue;
        predictor.observe(event);
        ordered.push_back(&event);
    }

    std::vector<std::size_t> bounds;
    for (std::size_t index = 0; index < ordered.size(); ++index) {
        const bool boundary = index == 0 ||
                              ordered[index]->request != ordered[index - 1]->request ||
                              ordered[index]->token_position !=
                                  ordered[index - 1]->token_position;
        if (boundary) bounds.push_back(index);
    }
    bounds.push_back(ordered.size());

    std::vector<std::vector<ExpertKey>> steps;
    steps.reserve(bounds.empty() ? 0 : bounds.size() - 1);
    for (std::size_t step = 0; step + 1 < bounds.size(); ++step) {
        const auto first = ordered.begin() + static_cast<std::ptrdiff_t>(bounds[step]);
        const auto last = ordered.begin() + static_cast<std::ptrdiff_t>(bounds[step + 1]);
        std::stable_sort(first, last, [](const RouteEvent* lhs, const RouteEvent* rhs) {
            return lhs->layer < rhs->layer;
        });
        std::vector<ExpertKey> keys;
        for (auto current = first; current != last; ++current) {
            for (const auto expert : (*current)->experts) {
                keys.push_back(ExpertKey{(*current)->layer, expert});
            }
        }
        steps.push_back(std::move(keys));
    }
    return steps;
}

void measure_demand_baseline(const std::vector<std::vector<ExpertKey>>& steps,
                             const ResidencyConfig& residency_config,
                             WindowOracleStats& stats) {
    ResidencyManager baseline(residency_config);
    std::uint64_t tick = 0;
    for (const auto& step : steps) {
        for (const auto key : step) {
            ++tick;
            const bool hot = baseline.vram_resident(key);
            (void)baseline.access(key, tick);
            if (!hot) ++stats.baseline_cold_accesses;
        }
    }
    baseline.finalize();
    stats.baseline_steps = steps.size();
    stats.baseline_read_bytes = stats.baseline_cold_accesses * residency_config.expert_bytes;
}

void project_costs(const WindowOracleConfig& config, std::uint64_t expert_bytes,
                   WindowOracleStats& stats) {
    const double gamma = static_cast<double>(config.window_tokens);
    double transfer_ms = config.expert_acquire_ms;
    if (config.h2d_gigabytes_per_second > 0.0) {
        transfer_ms += static_cast<double>(expert_bytes) * 1e3 /
                       (config.h2d_gigabytes_per_second * 1e9);
    }

    stats.projected_baseline_ms =
        static_cast<double>(stats.baseline_steps) * config.base_step_ms +
        static_cast<double>(stats.baseline_cold_accesses) * transfer_ms;
    stats.projected_window_ms =
        static_cast<double>(stats.windows) * gamma * config.draft_token_ms +
        static_cast<double>(stats.verify_rows) * config.verify_row_ms +
        static_cast<double>(stats.window_cold_unique_experts) * transfer_ms;

    if (stats.baseline_steps == 0 || stats.advanced_tokens == 0) return;
    const double baseline_per_token =
        stats.projected_baseline_ms / static_cast<double>(stats.baseline_steps);
    const double window_per_token =
        stats.projected_window_ms / static_cast<double>(stats.advanced_tokens);
    if (window_per_token > 0.0) stats.projected_speedup = baseline_per_token / window_per_token;
}

SimulationResult simulate_window(const std::vector<RouteEvent>& events,
                                 const SimulationConfig& config) {
    const auto& window = config.window;
    RoutePredictor predictor;
    const auto steps = build_steps(events, window.phase, predictor);

    WindowOracleStats stats;
    stats.enabled = true;
    stats.steps = steps.size();
    measure_demand_baseline(steps, config.residency, stats);

    const auto gamma = static_cast<std::size_t>(window.window_tokens);
    const std::size_t rows = gamma + 1U;
    const auto accepted = static_cast<std::size_t>(
        std::floor(std::clamp(window.acceptance_rate, 0.0, 1.0) * static_cast<double>(gamma)));
    const std::size_t advance = accepted + 1U;

    ResidencyManager residency(config.residency);
    std::uint64_t tick = 0;
    std::unordered_set<ExpertKey, ExpertKeyHash> seen;
    std::unordered_set<ExpertKey, ExpertKeyHash> cold;
    std::vector<ExpertKey> manifest;

    for (std::size_t start = 0; start < steps.size(); start += advance) {
        const std::size_t stop = std::min(start + rows, steps.size());
        seen.clear();
        cold.clear();
        manifest.clear();

        // The draft's real routers emit this manifest: every expert the verify
        // window will touch, classified against residency at window start.
        for (std::size_t step = start; step < stop; ++step) {
            for (const auto key : steps[step]) {
                ++stats.window_accesses;
                if (!seen.insert(key).second) continue;
                manifest.push_back(key);
                if (!residency.vram_resident(key)) cold.insert(key);
            }
        }
        for (std::size_t step = start; step < stop; ++step) {
            for (const auto key : steps[step]) {
                if (cold.count(key) != 0) ++stats.window_cold_accesses;
            }
        }

        ++stats.windows;
        stats.verify_rows += stop - start;
        stats.window_unique_experts += manifest.size();
        stats.window_cold_unique_experts += cold.size();
        stats.window_read_bytes += cold.size() * config.residency.expert_bytes;

        if (window.prefetch_window_union) {
            for (const auto key : manifest) {
                if (cold.count(key) == 0) continue;
                ++tick;
                (void)residency.prefetch(key, tick, 1.0);
            }
        }
        for (std::size_t step = start; step < stop; ++step) {
            for (const auto key : steps[step]) {
                ++tick;
                (void)residency.access(key, tick);
            }
        }

        stats.advanced_tokens += std::min(advance, steps.size() - start);
    }
    residency.finalize();

    if (stats.window_unique_experts > 0) {
        stats.dedup_factor = static_cast<double>(stats.window_accesses) /
                             static_cast<double>(stats.window_unique_experts);
        stats.break_even_acceptance = 1.0 / stats.dedup_factor;
    }
    if (stats.window_cold_unique_experts > 0) {
        stats.cold_dedup_factor = static_cast<double>(stats.window_cold_accesses) /
                                  static_cast<double>(stats.window_cold_unique_experts);
    }
    project_costs(window, config.residency.expert_bytes, stats);

    return SimulationResult{residency.stats(), events.size(),
                            predictor.transitions_observed(), stats};
}

}  // namespace

SimulationResult simulate(const std::vector<RouteEvent>& events,
                          const SimulationConfig& config) {
    if (config.window.window_tokens == 0) return simulate_sequential(events, config);
    return simulate_window(events, config);
}

}  // namespace strata
