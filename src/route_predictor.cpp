#include "strata/route_predictor.hpp"

#include <algorithm>
#include <unordered_map>

namespace strata {

void RoutePredictor::observe(const RouteEvent& event) {
    auto& previous = previous_by_request_[event.request];
    if (previous.valid) {
        const bool ordered_same_token =
            event.token == previous.event.token && event.layer > previous.event.layer;
        const bool ordered_next_token = event.token == previous.event.token + 1U;
        if (ordered_same_token || ordered_next_token) {
            for (const auto source_expert : previous.event.experts) {
                const ExpertKey source{previous.event.layer, source_expert};
                auto& destinations = transitions_[source];
                for (const auto destination_expert : event.experts) {
                    ++destinations[ExpertKey{event.layer, destination_expert}];
                    ++transitions_observed_;
                }
            }
        }
    }
    previous.event = event;
    previous.valid = true;
}

std::vector<RoutePrediction> RoutePredictor::predict(
    const RouteEvent& source, std::size_t limit, double minimum_confidence) const {
    if (limit == 0) return {};

    std::unordered_map<ExpertKey, std::uint64_t, ExpertKeyHash> aggregate;
    for (const auto expert : source.experts) {
        const auto found = transitions_.find(ExpertKey{source.layer, expert});
        if (found == transitions_.end()) continue;
        for (const auto& [destination, count] : found->second) {
            aggregate[destination] += count;
        }
    }
    if (aggregate.empty()) return {};

    std::uint64_t strongest = 0;
    for (const auto& [key, evidence] : aggregate) {
        (void)key;
        strongest = std::max(strongest, evidence);
    }

    std::vector<RoutePrediction> predictions;
    predictions.reserve(aggregate.size());
    for (const auto& [key, evidence] : aggregate) {
        /* Confidence is relative to the strongest candidate. Top-k destinations
         * are not mutually exclusive, so normalizing their scores to sum to one
         * would incorrectly cap a stable top-8 route near 1/8. */
        const auto confidence = static_cast<double>(evidence) /
                                static_cast<double>(strongest);
        if (confidence >= minimum_confidence) {
            predictions.push_back(RoutePrediction{key, confidence, evidence});
        }
    }
    std::sort(predictions.begin(), predictions.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.confidence != rhs.confidence) return lhs.confidence > rhs.confidence;
        if (lhs.evidence != rhs.evidence) return lhs.evidence > rhs.evidence;
        return lhs.key < rhs.key;
    });
    if (predictions.size() > limit) predictions.resize(limit);
    return predictions;
}

}  // namespace strata
