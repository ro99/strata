#include "test.hpp"

#include "strata/engine/load_report.hpp"
#include "strata/platform/process_stats.hpp"

#include <cstdint>
#include <string>

namespace {

strata::BlockDeviceStats device(std::uint64_t reads, std::uint64_t sectors,
                                std::uint64_t read_ms, std::uint64_t io_ms,
                                std::uint64_t weighted_ms) {
    strata::BlockDeviceStats stats;
    stats.reads_completed = reads;
    stats.sectors_read = sectors;
    stats.read_milliseconds = read_ms;
    stats.io_milliseconds = io_ms;
    stats.weighted_io_milliseconds = weighted_ms;
    stats.valid = true;
    return stats;
}

}  // namespace

TEST_CASE("storage interval derives queue depth from weighted in-flight time") {
    // One second of wall, during which the device accumulated 8000 ms of
    // in-flight-request time: eight requests in flight on average.
    const auto before = device(0U, 0U, 0U, 0U, 0U);
    const auto after = device(1000U, 2048U * 1000U, 4000U, 1000U, 8000U);
    strata::ProcessIoStats p_before;
    strata::ProcessIoStats p_after;
    p_after.storage_read_bytes = 1'048'576'000U;
    p_after.major_faults = 256U;

    const auto interval =
        strata::storage_interval(p_before, p_after, before, after, 1.0);

    REQUIRE(interval.device_valid);
    REQUIRE(interval.average_queue_depth > 7.99);
    REQUIRE(interval.average_queue_depth < 8.01);
    // 2,048,000 sectors of 512 bytes.
    REQUIRE(interval.device_bytes_read == 2048U * 1000U * 512U);
    REQUIRE(interval.major_faults == 256U);
    // Busy for the whole second.
    REQUIRE(interval.device_busy_fraction > 0.99);
    REQUIRE(interval.device_busy_fraction < 1.01);
    // 4000 ms spread over 1000 completed reads.
    REQUIRE(interval.average_read_latency_milliseconds > 3.99);
    REQUIRE(interval.average_read_latency_milliseconds < 4.01);
    // 1,048,576,000 device bytes over 1000 reads is 1 MiB per request.
    REQUIRE(interval.average_request_bytes > 1048575.0);
    REQUIRE(interval.average_request_bytes < 1048577.0);
}

TEST_CASE("a serialized reader reports queue depth near one") {
    // The shape experiment 0197 could not distinguish from outside the
    // process: the device is busy the whole time, but only ever with one
    // request, so more threads would not have helped.
    const auto before = device(0U, 0U, 0U, 0U, 0U);
    const auto after = device(500U, 1024U * 1000U, 1000U, 1000U, 1000U);
    const auto interval = strata::storage_interval({}, {}, before, after, 1.0);
    REQUIRE(interval.average_queue_depth > 0.99);
    REQUIRE(interval.average_queue_depth < 1.01);
    REQUIRE(interval.device_busy_fraction > 0.99);
}

TEST_CASE("storage interval survives absent or non-monotonic device counters") {
    strata::BlockDeviceStats missing;  // valid == false
    const auto interval =
        strata::storage_interval({}, {}, missing, missing, 1.0);
    REQUIRE(!interval.device_valid);
    REQUIRE(interval.average_queue_depth == 0.0);

    // A counter that went backwards means the sample is unusable. Reporting
    // zero is safer than reporting a wrapped delta of ~2^64.
    const auto high = device(1000U, 1000U, 1000U, 1000U, 1000U);
    const auto low = device(10U, 10U, 10U, 10U, 10U);
    const auto backwards = strata::storage_interval({}, {}, high, low, 1.0);
    REQUIRE(backwards.device_bytes_read == 0U);
    REQUIRE(backwards.reads_completed == 0U);
}

TEST_CASE("zero-length interval does not divide by zero") {
    const auto before = device(0U, 0U, 0U, 0U, 0U);
    const auto after = device(10U, 100U, 5U, 5U, 5U);
    const auto interval = strata::storage_interval({}, {}, before, after, 0.0);
    REQUIRE(interval.average_queue_depth == 0.0);
    REQUIRE(interval.device_read_gigabytes_per_second == 0.0);
    REQUIRE(interval.process_read_gigabytes_per_second == 0.0);
}

TEST_CASE("load report renders nothing until a phase is marked") {
    strata::LoadReport report;
    REQUIRE(report.render().empty());
    REQUIRE(!report.started());

    report.begin("");  // no block device is a valid, resolved answer
    REQUIRE(report.started());
    REQUIRE(report.render().empty());

    report.mark("load");
    REQUIRE(report.phases().size() == 1U);
    REQUIRE(report.phases().front().name == "load");
    const auto text = report.render();
    REQUIRE(text.find("load report") != std::string::npos);
    REQUIRE(text.find("major faults") != std::string::npos);
}

TEST_CASE("load report marks are ignored before begin") {
    strata::LoadReport report;
    report.mark("first-token");
    REQUIRE(report.phases().empty());
    REQUIRE(report.render().empty());
}

TEST_CASE("process stats read this process without failing") {
    const auto stats = strata::read_process_io_stats();
    // /proc is present on the hosts this runs on; the reader must at least
    // return a resident set, and must never throw.
    REQUIRE(stats.valid);
    REQUIRE(stats.resident_bytes > 0U);

    // An unresolvable device name is a normal outcome, not an error.
    const auto absent = strata::read_block_device_stats("definitely-not-a-disk");
    REQUIRE(!absent.valid);
    const auto empty = strata::read_block_device_stats("");
    REQUIRE(!empty.valid);
}
