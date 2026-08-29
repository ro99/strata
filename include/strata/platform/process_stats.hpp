#pragma once

// What this process and its backing device have actually done, sampled from
// /proc and /sys.
//
// Experiment 0197 left a question these counters exist to answer. A cold GLM
// first token reads roughly 220 GB and achieves 1.14-1.53 GB/s, while a probe
// reproducing the same access pattern reaches 2.9 GB/s as soon as four reads
// are in flight -- and the runtime already faults from 28 pinned workers. Three
// candidate explanations (readahead advice, request size, mmap against pread)
// were measured and rejected. What was never measured is how many requests the
// runtime actually has in flight, because nothing in the process could see it.
//
// `weighted_io_milliseconds` is the field that answers it: the kernel
// accumulates in-flight-request-count times elapsed time, so dividing its delta
// by wall time gives average queue depth, the same figure iostat prints as
// aqu-sz. A runtime issuing one read at a time reads about 1.0 no matter how
// many threads it has.

#include <cstdint>
#include <string>
#include <string_view>

namespace strata {

// One sample of what this process has asked the storage stack for.
struct ProcessIoStats {
    // /proc/self/io read_bytes: bytes that actually reached the block layer.
    // Page-cache hits do not appear here, which is what makes it the right
    // counter for distinguishing a warm run from a cold one.
    std::uint64_t storage_read_bytes{};
    // Bytes returned by read-like syscalls, cache hits included.
    std::uint64_t character_read_bytes{};
    std::uint64_t read_syscalls{};
    // Faults that required an I/O, and those that did not.
    std::uint64_t major_faults{};
    std::uint64_t minor_faults{};
    std::uint64_t resident_bytes{};
    bool valid{};
};

[[nodiscard]] ProcessIoStats read_process_io_stats() noexcept;

// One sample of a whole block device's counters, from /sys/block/<disk>/stat.
// Device-wide rather than per-process: it includes any other reader on the
// same disk, which is why a load report quotes it beside the per-process
// figures rather than instead of them.
struct BlockDeviceStats {
    std::uint64_t reads_completed{};
    // 512-byte units by kernel convention, regardless of the device's own
    // logical block size.
    std::uint64_t sectors_read{};
    // Summed across requests, so under concurrency it exceeds wall time.
    std::uint64_t read_milliseconds{};
    // Wall time with at least one request in flight.
    std::uint64_t io_milliseconds{};
    // In-flight count integrated over time. The queue-depth numerator.
    std::uint64_t weighted_io_milliseconds{};
    bool valid{};
};

// `disk` is a whole-disk name such as `nvme0n1`, not a partition. An empty or
// unreadable name yields `valid == false` rather than an error: a load report
// on a tmpfs-backed or unresolvable checkpoint should still report everything
// else it knows.
[[nodiscard]] BlockDeviceStats read_block_device_stats(
    std::string_view disk) noexcept;

// What happened between two samples.
struct StorageInterval {
    double wall_seconds{};
    std::uint64_t process_storage_read_bytes{};
    std::uint64_t device_bytes_read{};
    std::uint64_t major_faults{};
    std::uint64_t reads_completed{};
    // Fraction of the interval with at least one request in flight.
    double device_busy_fraction{};
    // Average number of requests in flight. About 1.0 means the reader is
    // serialized however many threads it runs.
    double average_queue_depth{};
    double average_read_latency_milliseconds{};
    // Bytes per completed device read. Demand paging produces many small
    // requests; a prefetch path produces few large ones, and the two reach
    // very different fractions of a device's rated bandwidth at the same
    // queue depth.
    double average_request_bytes{};
    double process_read_gigabytes_per_second{};
    double device_read_gigabytes_per_second{};
    bool device_valid{};
};

[[nodiscard]] StorageInterval storage_interval(
    const ProcessIoStats& before, const ProcessIoStats& after,
    const BlockDeviceStats& device_before, const BlockDeviceStats& device_after,
    double wall_seconds) noexcept;

}  // namespace strata
