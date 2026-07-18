#include "test.hpp"

#include "strata/trace.hpp"

#include <filesystem>
#include <cstdio>

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

TEST_CASE("runtime route traces round-trip into the simulator contract") {
    const auto path = std::filesystem::temp_directory_path() /
                      "strata-route-trace-roundtrip.jsonl";
    strata::RouteTraceWriter writer;
    REQUIRE(writer.open(path).ok());
    strata::RouteEvent event;
    event.request = 7U;
    event.token_position = 11U;
    event.layer = 3U;
    event.experts = {4U, 9U};
    event.coefficients = {0.75F, 0.25F};
    event.phase = strata::RoutePhase::Decode;
    REQUIRE(writer.write(event).ok());
    REQUIRE(writer.flush().ok());

    const auto parsed = strata::parse_route_trace(path);
    REQUIRE(parsed.ok());
    REQUIRE(parsed.events.size() == 1U);
    REQUIRE(parsed.events[0].request == 7U);
    REQUIRE(parsed.events[0].token_position == 11U);
    REQUIRE(parsed.events[0].layer == 3U);
    REQUIRE(parsed.events[0].experts == std::vector<std::uint32_t>({4U, 9U}));
    REQUIRE(parsed.events[0].phase == strata::RoutePhase::Decode);
    static_cast<void>(std::remove(path.c_str()));
}
