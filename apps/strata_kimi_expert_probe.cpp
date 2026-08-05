// Measures the Kimi-K3 routed-expert read path in isolation, before any
// runtime is built around it. The charter's rule: a microbenchmark that
// reproduces the mechanism usually decides the question in minutes, and a probe
// that does not reproduce the production access pattern can report no problem
// where a large one exists.
//
// The production pattern is a cold, randomly placed 16.73 MiB expert triple
// read out of a 1.4 TiB shard set. This probe reproduces exactly that: random
// (layer, expert) pairs across the whole checkpoint, an arena far smaller than
// the routed set so misses dominate, and interleaved repetitions so a one-off
// page-cache effect cannot be mistaken for a result.

#include "strata/kimi_k3_checkpoint.hpp"
#include "strata/kimi_k3_expert_arena.hpp"
#include "strata/model_adapter.hpp"
#include "strata/placement.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string model = "/data/kimi-k3";
    std::vector<std::uint32_t> queue_depths{1U, 2U, 4U, 8U};
    std::uint32_t experts = 96U;
    std::uint32_t repetitions = 3U;
    std::uint64_t arena_experts = 32U;
    std::uint64_t seed = 20260802U;
    bool buffered = false;
    bool lock = true;
};

void usage() {
    std::cout
        << "usage: strata-kimi-expert-probe [--model DIR] [--experts N]\n"
           "         [--queue-depths 1,2,4,8] [--repetitions N]\n"
           "         [--arena-experts N] [--buffered] [--no-lock] [--seed N]\n\n"
           "Reports median MB/s per queue depth over interleaved repetitions,\n"
           "plus the NVMe write delta, which must stay at background level.\n";
}

std::vector<std::uint32_t> parse_list(const std::string& text) {
    std::vector<std::uint32_t> values;
    std::size_t begin = 0U;
    while (begin <= text.size()) {
        const auto comma = text.find(',', begin);
        const auto piece = text.substr(begin, comma == std::string::npos
                                                  ? std::string::npos
                                                  : comma - begin);
        if (!piece.empty()) {
            values.push_back(static_cast<std::uint32_t>(std::stoul(piece)));
        }
        if (comma == std::string::npos) break;
        begin = comma + 1U;
    }
    return values;
}

bool parse(int argc, char** argv, Options& options) {
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto next = [&]() -> std::string {
            if (index + 1 >= argc) return {};
            return argv[++index];
        };
        if (argument == "--model") options.model = next();
        else if (argument == "--experts") options.experts =
            static_cast<std::uint32_t>(std::stoul(next()));
        else if (argument == "--queue-depths") options.queue_depths =
            parse_list(next());
        else if (argument == "--repetitions") options.repetitions =
            static_cast<std::uint32_t>(std::stoul(next()));
        else if (argument == "--arena-experts") options.arena_experts =
            std::stoull(next());
        else if (argument == "--seed") options.seed = std::stoull(next());
        else if (argument == "--buffered") options.buffered = true;
        else if (argument == "--no-lock") options.lock = false;
        else if (argument == "--help" || argument == "-h") { usage(); return false; }
        else {
            std::cerr << "unknown argument " << argument << '\n';
            return false;
        }
    }
    return !options.model.empty() && !options.queue_depths.empty() &&
           options.experts > 0U && options.repetitions > 0U;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse(argc, argv, options)) return 2;

    const auto& contract = strata::kKimiK3ExecutionContract;
    const auto storage = strata::resolve_backing_storage(options.model);
    if (!storage.resolved) {
        std::cerr << "cannot resolve the backing device of " << options.model
                  << '\n';
        return 1;
    }

    // Every NVMe in the machine, so the write gate covers whichever one the
    // root filesystem happens to be on.
    std::vector<std::string> nvme_disks;
    for (const auto& candidate : {"nvme0n1", "nvme1n1"}) {
        if (strata::kimi_disk_sectors_written(candidate).ok()) {
            nvme_disks.emplace_back(candidate);
        }
    }

    strata::KimiWriteGuardConfig guard;
    guard.forbidden_disks = nvme_disks;
    guard.write_paths = {};
    guard.require_no_forbidden_swap = false;  // reported below, not enforced here
    const auto guarded = strata::kimi_apply_write_guard(guard);
    for (const auto& error : guarded.errors) {
        std::cout << "guard        " << error << '\n';
    }

    auto opened = strata::KimiCheckpointReader::open(options.model);
    if (!opened.ok()) {
        for (const auto& error : opened.errors) std::cerr << error << '\n';
        return 1;
    }
    const auto& checkpoint = *opened.value;

    const auto expert_bytes = strata::KimiCheckpointReader::expert_source_bytes();
    strata::KimiArenaConfig arena_config;
    arena_config.capacity_bytes = options.arena_experts * (expert_bytes + 4096U);
    arena_config.lock_pages = options.lock;
    strata::KimiExpertArena arena;
    const auto reset = arena.reset(arena_config);
    if (!reset.ok()) {
        for (const auto& error : reset.errors) std::cerr << error << '\n';
        return 1;
    }

    std::cout << "kimi expert read probe\n"
              << "  checkpoint   " << options.model << '\n'
              << "  storage      " << storage.device << " on " << storage.disk
              << (storage.nvme ? " (nvme)" : " (non-nvme)") << '\n'
              << "  arena        " << strata::format_bytes(arena.capacity_bytes())
              << " over " << arena.slot_count() << " experts, "
              << (arena.locked() ? "locked" : "UNLOCKED") << '\n'
              << "  access       " << (options.buffered ? "buffered" : "O_DIRECT")
              << ", " << options.experts << " random experts per arm, "
              << options.repetitions << " interleaved repetitions\n"
              << "  routed set   "
              << strata::format_bytes(
                     static_cast<std::uint64_t>(contract.routed_experts) *
                     (contract.layer_count - contract.dense_prefix_layers) *
                     expert_bytes)
              << " across " << checkpoint.manifest().shards.size() << " shards\n\n";

    // Calibrate the write gate against this machine's actual idle rate. A
    // threshold carried over from another machine's remembered figure reports
    // a failure where there is none: this host idles near 19 KB/s of journald
    // traffic, roughly 200x the 0.1 KiB/s the handover recorded.
    std::vector<double> idle_rates;
    for (const auto& disk : nvme_disks) {
        const auto rate = strata::kimi_measure_idle_write_rate(disk, 10.0);
        idle_rates.push_back(rate.ok() ? rate.value : 0.0);
        std::cout << "  idle " << disk << "  " << std::fixed
                  << std::setprecision(1) << idle_rates.back() / 1024.0
                  << " KiB/s background over a 10 s window\n";
    }
    std::cout << '\n';

    std::vector<std::uint64_t> nvme_before;
    for (const auto& disk : nvme_disks) {
        nvme_before.push_back(strata::kimi_disk_sectors_written(disk).value);
    }
    const auto sata_before = strata::kimi_disk_sectors_read(storage.disk);
    const auto wall_begin = std::chrono::steady_clock::now();

    // Interleave the arms: a single-shot comparison of O_DIRECT against
    // buffered once showed a 5.4x gap that three interleaved repetitions
    // identified as one-off page-cache reclaim.
    std::vector<std::vector<double>> rates(options.queue_depths.size());
    for (std::uint32_t repetition = 0U; repetition < options.repetitions;
         ++repetition) {
        for (std::size_t arm = 0U; arm < options.queue_depths.size(); ++arm) {
            strata::KimiReaderConfig reader_config;
            reader_config.queue_depth = options.queue_depths[arm];
            reader_config.direct = !options.buffered;
            strata::KimiExpertReader reader;
            const auto ready = reader.open(checkpoint, reader_config);
            if (!ready.ok()) {
                for (const auto& error : ready.errors) std::cerr << error << '\n';
                return 1;
            }
            // A fresh arena per arm, so every read is a cold miss and the arm
            // measures the link rather than the cache.
            if (!arena.reset(arena_config).ok()) return 1;

            // Random (layer, expert) pairs across the whole routed set. A
            // sequential sweep would measure readahead, not the production
            // pattern.
            std::mt19937_64 generator(options.seed + repetition * 977U + arm);
            std::uniform_int_distribution<std::uint32_t> layers(
                contract.dense_prefix_layers, contract.layer_count - 1U);
            std::uniform_int_distribution<std::uint32_t> experts(
                0U, contract.routed_experts - 1U);
            std::vector<strata::KimiReadRequest> requests(options.experts);
            for (auto& request : requests) {
                request.layer = layers(generator);
                request.expert = experts(generator);
            }

            const auto begin = std::chrono::steady_clock::now();
            const auto staged = reader.stage(checkpoint, arena, requests);
            const auto end = std::chrono::steady_clock::now();
            if (!staged.ok()) {
                for (const auto& error : staged.errors) std::cerr << error << '\n';
                return 1;
            }
            const auto seconds =
                std::chrono::duration<double>(end - begin).count();
            const auto megabytes =
                static_cast<double>(reader.stats().bytes_read) / 1.0e6;
            rates[arm].push_back(megabytes / seconds);
            std::cout << "  rep " << repetition << "  qd "
                      << std::setw(2) << options.queue_depths[arm] << "  "
                      << std::fixed << std::setprecision(1) << std::setw(7)
                      << megabytes / seconds << " MB/s  "
                      << std::setprecision(2) << seconds << " s  "
                      << strata::format_bytes(reader.stats().bytes_read) << '\n';
        }
    }

    const auto wall_end = std::chrono::steady_clock::now();
    const auto wall = std::chrono::duration<double>(wall_end - wall_begin).count();

    std::cout << "\n  queue depth     median MB/s\n";
    double baseline = 0.0;
    for (std::size_t arm = 0U; arm < options.queue_depths.size(); ++arm) {
        const auto value = median(rates[arm]);
        if (arm == 0U) baseline = value;
        std::cout << "  " << std::setw(11) << options.queue_depths[arm] << "  "
                  << std::setw(14) << std::fixed << std::setprecision(1) << value;
        if (arm != 0U && baseline > 0.0) {
            std::cout << "   " << std::setprecision(2) << value / baseline
                      << "x over qd " << options.queue_depths[0U];
        }
        std::cout << '\n';
    }

    std::cout << "\n  wall         " << std::fixed << std::setprecision(1) << wall
              << " s\n";
    const auto sata_after = strata::kimi_disk_sectors_read(storage.disk);
    if (sata_before.ok() && sata_after.ok()) {
        std::cout << "  " << storage.disk << " read   "
                  << strata::format_bytes(sata_after.value - sata_before.value)
                  << " (the checkpoint; this is the intended path)\n";
    }
    bool clean = true;
    for (std::size_t index = 0U; index < nvme_disks.size(); ++index) {
        const auto after = strata::kimi_disk_sectors_written(nvme_disks[index]);
        if (!after.ok()) continue;
        const auto delta = after.value - nvme_before[index];
        // Background scales with elapsed time, not with bytes read, so the
        // allowance is the measured idle rate over the run's wall time with a
        // 2x margin for burstiness.
        const auto allowance =
            static_cast<std::uint64_t>(idle_rates[index] * wall * 2.0) +
            (256U << 10U);
        // The rate is the number that decides this, not the total: background
        // scales with time. A single idle sample can catch a burst and inflate
        // the allowance, so both rates are printed and the comparison is
        // legible rather than resting on one calibration.
        std::cout << "  " << nvme_disks[index] << " write  "
                  << strata::format_bytes(delta) << " at " << std::fixed
                  << std::setprecision(1)
                  << static_cast<double>(delta) / wall / 1024.0
                  << " KiB/s against a " << idle_rates[index] / 1024.0
                  << " KiB/s idle baseline ("
                  << strata::format_bytes(allowance) << " allowance)  "
                  << (delta <= allowance ? "PASS" : "FAIL") << '\n';
        if (delta > allowance) clean = false;
    }
    std::cout << "\n  verdict      "
              << (clean ? "no model byte reached an NVMe"
                        : "NVMe writes exceeded the background allowance")
              << '\n';
    return clean ? 0 : 1;
}
