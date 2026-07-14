#include "test.hpp"

#include "strata/route_predictor.hpp"

TEST_CASE("predictor learns cross-layer transitions without future input") {
    strata::RoutePredictor predictor;
    const strata::RouteEvent source0{1, 0, 3, {1, 2}};
    const strata::RouteEvent target0{1, 0, 4, {7, 8}};
    const strata::RouteEvent source1{1, 1, 3, {1, 2}};
    predictor.observe(source0);
    REQUIRE(predictor.predict(source0, 4, 0.5).empty());
    predictor.observe(target0);
    predictor.observe(source1);
    const auto predictions = predictor.predict(source1, 4, 0.5);
    REQUIRE(predictions.size() == 2);
    REQUIRE(predictions[0].key.layer == 4);
    REQUIRE(predictions[0].confidence == 1.0);
    REQUIRE(predictor.transitions_observed() == 8);
}

TEST_CASE("predictor isolates request histories") {
    strata::RoutePredictor predictor;
    predictor.observe(strata::RouteEvent{1, 0, 3, {1}});
    predictor.observe(strata::RouteEvent{2, 0, 3, {9}});
    predictor.observe(strata::RouteEvent{1, 0, 4, {2}});
    const auto p = predictor.predict(strata::RouteEvent{1, 1, 3, {1}}, 2, 0.1);
    REQUIRE(p.size() == 1);
    const strata::ExpertKey expected{4, 2};
    REQUIRE(p[0].key == expected);
}
