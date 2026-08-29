#pragma once

// Contract section 8 of the GLM-5.3 campaign requires a load report carrying
// cold load, warm load, first token, bytes read, faults, RSS, VRAM and any
// declared I/O-dependent tier. This is that report, model-neutral because
// nothing in it is GLM-specific.
//
// It also carries one field the contract does not name and experiment 0197
// showed was the missing measurement: average storage queue depth per phase.
// A cold first token reads at roughly single-stream throughput even though the
// host MoE faults from 28 pinned workers, and no probe could distinguish "the
// runtime issues one read at a time" from "the device cannot go faster" from
// outside the process. Queue depth distinguishes them directly.

#include "strata/platform/process_stats.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata {

struct LoadReportPhase {
    std::string name;
    double seconds{};
    StorageInterval storage;
    std::uint64_t resident_bytes_at_end{};
};

// Phases are whatever the caller marks; a runtime typically marks `load` when
// the model is ready to accept work and `first-token` when the first token is
// published, because on a lazily mapped checkpoint those two costs are wildly
// different and only the second contains the fault-in.
class LoadReport {
public:
    // `disk` is a whole-disk name from `resolve_backing_storage(...).disk`.
    // Empty is valid and simply omits device-wide counters.
    void begin(std::string disk);
    void mark(std::string name);

    void set_device_vram_bytes(std::vector<std::uint64_t> bytes);
    // Free-form, e.g. the placement plan's I/O-dependence verdict.
    void set_storage_tier_note(std::string note);
    void set_storage_tier_bytes(std::uint64_t bytes);

    [[nodiscard]] bool started() const noexcept { return started_; }
    [[nodiscard]] const std::vector<LoadReportPhase>& phases() const noexcept {
        return phases_;
    }
    // Human-readable block, one line per phase plus a summary. Returns an
    // empty string when nothing was marked, so a caller may print it
    // unconditionally.
    [[nodiscard]] std::string render() const;

private:
    bool started_{};
    std::string disk_;
    std::string tier_note_;
    std::uint64_t storage_tier_bytes_{};
    std::vector<std::uint64_t> device_vram_bytes_;
    std::vector<LoadReportPhase> phases_;
    ProcessIoStats last_process_;
    BlockDeviceStats last_device_;
    double last_seconds_{};
};

}  // namespace strata
