#include "strata/platform/process_stats.hpp"

#include <array>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace strata {
namespace {

// A missing or malformed field leaves its value zero and the sample invalid,
// which callers must render as "unknown" rather than as "none". Every field
// here is optional on some kernel configuration.
bool parse_prefixed_u64(std::string_view line, std::string_view prefix,
                        std::uint64_t& out) noexcept {
    if (line.rfind(prefix, 0U) != 0U) return false;
    auto rest = line.substr(prefix.size());
    while (!rest.empty() && (rest.front() == ' ' || rest.front() == '\t')) {
        rest.remove_prefix(1U);
    }
    std::uint64_t value = 0U;
    const auto* first = rest.data();
    const auto* last = first + rest.size();
    const auto parsed = std::from_chars(first, last, value);
    if (parsed.ec != std::errc{}) return false;
    out = value;
    return true;
}

}  // namespace

ProcessIoStats read_process_io_stats() noexcept {
    ProcessIoStats stats;
    try {
        std::ifstream io("/proc/self/io");
        std::string line;
        while (std::getline(io, line)) {
            parse_prefixed_u64(line, "read_bytes:", stats.storage_read_bytes);
            parse_prefixed_u64(line, "rchar:", stats.character_read_bytes);
            parse_prefixed_u64(line, "syscr:", stats.read_syscalls);
        }

        std::ifstream status("/proc/self/status");
        std::uint64_t rss_kb = 0U;
        while (std::getline(status, line)) {
            if (parse_prefixed_u64(line, "VmRSS:", rss_kb)) {
                stats.resident_bytes = rss_kb * 1024U;
            }
        }

        // Fields 10 and 12 of /proc/self/stat are minflt and majflt. The
        // process name in field 2 may contain spaces and parentheses, so the
        // scan starts after the final ')' rather than tokenising from the
        // beginning.
        std::ifstream stat("/proc/self/stat");
        std::string contents;
        std::getline(stat, contents);
        const auto close = contents.rfind(')');
        if (close != std::string::npos) {
            std::string_view rest(contents);
            rest.remove_prefix(close + 1U);
            int field = 2;  // field 3 is the next token
            std::uint64_t value = 0U;
            while (!rest.empty()) {
                while (!rest.empty() && rest.front() == ' ') rest.remove_prefix(1U);
                if (rest.empty()) break;
                const auto end = rest.find(' ');
                const auto token = rest.substr(0U, end);
                ++field;
                const auto* first = token.data();
                if (std::from_chars(first, first + token.size(), value).ec ==
                    std::errc{}) {
                    if (field == 10) stats.minor_faults = value;
                    if (field == 12) stats.major_faults = value;
                }
                if (field >= 12) break;
                if (end == std::string_view::npos) break;
                rest.remove_prefix(end);
            }
        }
        stats.valid = stats.character_read_bytes != 0U ||
                      stats.resident_bytes != 0U;
    } catch (...) {
        stats.valid = false;
    }
    return stats;
}

BlockDeviceStats read_block_device_stats(std::string_view disk) noexcept {
    BlockDeviceStats stats;
    if (disk.empty()) return stats;
    try {
        const auto path =
            std::filesystem::path("/sys/block") / std::string(disk) / "stat";
        std::ifstream file(path);
        if (!file) return stats;
        // reads-completed reads-merged sectors-read ms-reading writes... and
        // then in-flight, ms-doing-io, weighted-ms-doing-io.
        std::array<std::uint64_t, 11> fields{};
        std::size_t count = 0U;
        while (count < fields.size() && (file >> fields[count])) ++count;
        if (count < 11U) return stats;
        stats.reads_completed = fields[0];
        stats.sectors_read = fields[2];
        stats.read_milliseconds = fields[3];
        stats.io_milliseconds = fields[9];
        stats.weighted_io_milliseconds = fields[10];
        stats.valid = true;
    } catch (...) {
        stats.valid = false;
    }
    return stats;
}

StorageInterval storage_interval(
    const ProcessIoStats& before, const ProcessIoStats& after,
    const BlockDeviceStats& device_before, const BlockDeviceStats& device_after,
    double wall_seconds) noexcept {
    StorageInterval interval;
    interval.wall_seconds = wall_seconds;
    const auto delta = [](std::uint64_t a, std::uint64_t b) {
        // Counters only advance; a decrease means the sample is unusable, and
        // reporting zero is safer than reporting a huge wrapped value.
        return b >= a ? b - a : 0U;
    };
    interval.process_storage_read_bytes =
        delta(before.storage_read_bytes, after.storage_read_bytes);
    interval.major_faults = delta(before.major_faults, after.major_faults);
    if (wall_seconds > 0.0) {
        interval.process_read_gigabytes_per_second =
            static_cast<double>(interval.process_storage_read_bytes) /
            wall_seconds / 1e9;
    }

    if (!device_before.valid || !device_after.valid) return interval;
    interval.device_valid = true;
    interval.device_bytes_read =
        delta(device_before.sectors_read, device_after.sectors_read) * 512U;
    interval.reads_completed =
        delta(device_before.reads_completed, device_after.reads_completed);
    const auto busy_ms =
        delta(device_before.io_milliseconds, device_after.io_milliseconds);
    const auto weighted_ms = delta(device_before.weighted_io_milliseconds,
                                   device_after.weighted_io_milliseconds);
    const auto read_ms =
        delta(device_before.read_milliseconds, device_after.read_milliseconds);
    if (wall_seconds > 0.0) {
        const double wall_ms = wall_seconds * 1000.0;
        interval.device_busy_fraction = static_cast<double>(busy_ms) / wall_ms;
        interval.average_queue_depth =
            static_cast<double>(weighted_ms) / wall_ms;
        interval.device_read_gigabytes_per_second =
            static_cast<double>(interval.device_bytes_read) / wall_seconds / 1e9;
    }
    if (interval.reads_completed != 0U) {
        interval.average_read_latency_milliseconds =
            static_cast<double>(read_ms) /
            static_cast<double>(interval.reads_completed);
        interval.average_request_bytes =
            static_cast<double>(interval.device_bytes_read) /
            static_cast<double>(interval.reads_completed);
    }
    return interval;
}

}  // namespace strata
