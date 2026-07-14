#include "test.hpp"

#include "strata/trace.hpp"

#include <filesystem>

#ifndef STRATA_SOURCE_DIR
#define STRATA_SOURCE_DIR "."
#endif

TEST_CASE("route trace parser reads ordered expert sets") {
    const auto path = std::filesystem::path(STRATA_SOURCE_DIR) /
                      "tests/data/route_trace_v1.txt";
    const auto trace = strata::parse_route_trace(path);
    REQUIRE(trace.ok());
    REQUIRE(!trace.events.empty());
    REQUIRE(trace.events.front().request == 0);
    REQUIRE(trace.events.front().layer == 3);
    REQUIRE(trace.events.front().experts.size() == 2);
}
