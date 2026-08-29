#include "strata/engine/load_report.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <sstream>
#include <utility>

namespace strata {
namespace {

double monotonic_seconds() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::string gigabytes(std::uint64_t bytes) {
    std::array<char, 32> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%.2f GB",
                  static_cast<double>(bytes) / 1e9);
    return buffer.data();
}

std::string fixed(double value, int precision) {
    std::array<char, 32> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%.*f", precision, value);
    return buffer.data();
}

}  // namespace

void LoadReport::begin(std::string disk) {
    disk_ = std::move(disk);
    last_process_ = read_process_io_stats();
    last_device_ = read_block_device_stats(disk_);
    last_seconds_ = monotonic_seconds();
    started_ = true;
}

void LoadReport::mark(std::string name) {
    if (!started_) return;
    const auto now = monotonic_seconds();
    const auto process = read_process_io_stats();
    const auto device = read_block_device_stats(disk_);

    LoadReportPhase phase;
    phase.name = std::move(name);
    phase.seconds = now - last_seconds_;
    phase.storage = storage_interval(last_process_, process, last_device_,
                                     device, phase.seconds);
    phase.resident_bytes_at_end = process.resident_bytes;
    phases_.push_back(std::move(phase));

    last_process_ = process;
    last_device_ = device;
    last_seconds_ = now;
}

void LoadReport::set_device_vram_bytes(std::vector<std::uint64_t> bytes) {
    device_vram_bytes_ = std::move(bytes);
}

void LoadReport::set_storage_tier_note(std::string note) {
    tier_note_ = std::move(note);
}

void LoadReport::set_storage_tier_bytes(std::uint64_t bytes) {
    storage_tier_bytes_ = bytes;
}

std::string LoadReport::render() const {
    if (phases_.empty()) return {};
    std::ostringstream out;
    out << "\n  load report\n";
    for (const auto& phase : phases_) {
        const auto& s = phase.storage;
        out << "    " << phase.name << "  " << fixed(phase.seconds, 2) << " s"
            << "   read " << gigabytes(s.process_storage_read_bytes) << " at "
            << fixed(s.process_read_gigabytes_per_second, 2) << " GB/s"
            << "   " << s.major_faults << " major faults"
            << "   rss " << gigabytes(phase.resident_bytes_at_end) << '\n';
        if (s.device_valid) {
            // Queue depth is the field that separates a serialized reader from
            // a saturated device: about 1.0 means one request in flight
            // however many threads are running.
            out << "      device " << gigabytes(s.device_bytes_read) << " at "
                << fixed(s.device_read_gigabytes_per_second, 2) << " GB/s"
                << ", queue depth " << fixed(s.average_queue_depth, 2)
                << ", busy " << fixed(s.device_busy_fraction * 100.0, 1) << "%"
                << ", " << s.reads_completed << " reads of "
                << fixed(s.average_request_bytes / 1024.0, 1) << " KiB at "
                << fixed(s.average_read_latency_milliseconds, 3)
                << " ms each\n";
        }
    }
    if (!device_vram_bytes_.empty()) {
        out << "    vram   ";
        for (std::size_t i = 0; i < device_vram_bytes_.size(); ++i) {
            if (i != 0U) out << ", ";
            out << gigabytes(device_vram_bytes_[i]);
        }
        out << '\n';
    }
    if (storage_tier_bytes_ != 0U) {
        out << "    storage tier " << gigabytes(storage_tier_bytes_)
            << " read on demand from the checkpoint\n";
    }
    if (!tier_note_.empty()) out << "    tier   " << tier_note_ << '\n';
    return out.str();
}

}  // namespace strata
