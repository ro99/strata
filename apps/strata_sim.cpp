#include "strata/simulator.hpp"
#include "strata/trace.hpp"
#include "strata/types.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using strata::ReplacementPolicy;

struct Options {
    std::string trace;
    std::uint64_t ram_bytes = 180'000'000'000ULL;
    std::uint64_t vram_bytes = 58'000'000'000ULL;
    std::uint64_t expert_bytes = 19'000'000ULL;
    std::uint64_t cold_read_budget{};
    std::uint64_t peer_activation_bytes = 64U * 1024U;
    std::uint64_t lease_ticks = 16;
    std::uint32_t peer_basis_points{};
    std::size_t prefetch{};
    double confidence = 0.50;
    ReplacementPolicy policy{ReplacementPolicy::Lease};
    bool compare{};
    bool strict{};
    bool json{};
};

[[noreturn]] void usage(int code) {
    std::ostream& out = code == 0 ? std::cout : std::cerr;
    out << "strata-sim " << STRATA_VERSION << "\n"
        << "Usage: strata-sim --trace FILE [options]\n\n"
        << "  --policy lru|lfu|lease     replacement policy (default: lease)\n"
        << "  --compare                  run all replacement policies\n"
        << "  --ram BYTES                local RAM expert capacity (suffix K/M/G/T)\n"
        << "  --vram BYTES               VRAM expert capacity\n"
        << "  --expert-bytes BYTES       canonical int4 expert page size\n"
        << "  --prefetch N               maximum predicted experts per route event\n"
        << "  --confidence P             relative predictor threshold [0,1]\n"
        << "  --lease N                  protected residency ticks\n"
        << "  --peer-resident-pct P      deterministic remote-resident expert percent\n"
        << "  --peer-activation-bytes N  network bytes per remote expert batch\n"
        << "  --cold-read-budget BYTES   total permitted NVMe reads (0 = unlimited)\n"
        << "  --strict                   refuse reads beyond the cold budget\n"
        << "  --json                     machine-readable output\n";
    std::exit(code);
}

std::uint64_t parse_bytes(std::string text) {
    if (text.empty()) throw std::invalid_argument("empty byte value");
    if (text.front() == '-') throw std::invalid_argument("byte value cannot be negative");
    std::uint64_t multiplier = 1;
    const char suffix = static_cast<char>(std::toupper(static_cast<unsigned char>(text.back())));
    if (suffix == 'K' || suffix == 'M' || suffix == 'G' || suffix == 'T') {
        text.pop_back();
        if (suffix == 'K') multiplier = 1'000ULL;
        if (suffix == 'M') multiplier = 1'000'000ULL;
        if (suffix == 'G') multiplier = 1'000'000'000ULL;
        if (suffix == 'T') multiplier = 1'000'000'000'000ULL;
    }
    std::size_t consumed = 0;
    const auto value = std::stoull(text, &consumed);
    if (consumed != text.size() || value > std::numeric_limits<std::uint64_t>::max() / multiplier) {
        throw std::invalid_argument("invalid or overflowing byte value");
    }
    return value * multiplier;
}

ReplacementPolicy parse_policy(std::string_view value) {
    if (value == "lru") return ReplacementPolicy::Lru;
    if (value == "lfu") return ReplacementPolicy::Lfu;
    if (value == "lease") return ReplacementPolicy::Lease;
    throw std::invalid_argument("unknown policy: " + std::string(value));
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        auto next = [&]() -> std::string {
            if (++i >= argc) throw std::invalid_argument("missing value after " + std::string(argument));
            return argv[i];
        };
        if (argument == "--help" || argument == "-h") usage(0);
        else if (argument == "--trace") options.trace = next();
        else if (argument == "--policy") options.policy = parse_policy(next());
        else if (argument == "--compare") options.compare = true;
        else if (argument == "--ram") options.ram_bytes = parse_bytes(next());
        else if (argument == "--vram") options.vram_bytes = parse_bytes(next());
        else if (argument == "--expert-bytes") options.expert_bytes = parse_bytes(next());
        else if (argument == "--prefetch") options.prefetch = std::stoull(next());
        else if (argument == "--confidence") options.confidence = std::stod(next());
        else if (argument == "--lease") options.lease_ticks = std::stoull(next());
        else if (argument == "--peer-resident-pct") {
            const double percent = std::stod(next());
            if (!std::isfinite(percent) || percent < 0.0 || percent > 100.0) {
                throw std::invalid_argument("peer resident percent must be in [0,100]");
            }
            options.peer_basis_points = static_cast<std::uint32_t>(std::llround(percent * 100.0));
        } else if (argument == "--peer-activation-bytes") {
            options.peer_activation_bytes = parse_bytes(next());
        } else if (argument == "--cold-read-budget") {
            options.cold_read_budget = parse_bytes(next());
        } else if (argument == "--strict") options.strict = true;
        else if (argument == "--json") options.json = true;
        else throw std::invalid_argument("unknown option: " + std::string(argument));
    }
    if (options.trace.empty()) throw std::invalid_argument("--trace is required");
    if (options.expert_bytes == 0) throw std::invalid_argument("expert bytes must be positive");
    if (!std::isfinite(options.confidence) || options.confidence < 0.0 ||
        options.confidence > 1.0) {
        throw std::invalid_argument("confidence must be in [0,1]");
    }
    return options;
}

strata::SimulationConfig make_config(const Options& options, ReplacementPolicy policy) {
    strata::SimulationConfig config;
    config.residency.vram_capacity_bytes = options.vram_bytes;
    config.residency.ram_capacity_bytes = options.ram_bytes;
    config.residency.expert_bytes = options.expert_bytes;
    config.residency.cold_read_budget_bytes = options.cold_read_budget;
    config.residency.peer_activation_roundtrip_bytes = options.peer_activation_bytes;
    config.residency.peer_resident_basis_points = options.peer_basis_points;
    config.residency.lease_ticks = options.lease_ticks;
    config.residency.policy = policy;
    config.residency.strict_cold_read_budget = options.strict;
    config.prefetch_limit = options.prefetch;
    config.minimum_prediction_confidence = options.confidence;
    return config;
}

double hit_rate(const strata::ResidencyStats& stats) {
    if (stats.accesses == 0) return 0.0;
    return 100.0 * static_cast<double>(stats.vram_hits + stats.ram_hits + stats.peer_hits) /
           static_cast<double>(stats.accesses);
}

void print_human(ReplacementPolicy policy, const strata::SimulationResult& result) {
    const auto& s = result.residency;
    std::cout << "policy " << strata::to_string(policy) << '\n'
              << "  events/accesses:       " << result.events << " / " << s.accesses << '\n'
              << "  hit rate:              " << std::fixed << std::setprecision(2)
              << hit_rate(s) << "%\n"
              << "  VRAM/RAM/peer hits:    " << s.vram_hits << " / " << s.ram_hits
              << " / " << s.peer_hits << '\n'
              << "  NVMe demand misses:    " << s.nvme_misses << '\n'
              << "  NVMe bytes:            " << s.nvme_read_bytes << '\n'
              << "  prefetch bytes:        " << s.nvme_prefetch_bytes << '\n'
              << "  useful/wasted prefetch:" << s.useful_prefetches << " / "
              << s.wasted_prefetches << '\n'
              << "  weight H2D bytes:      " << s.weight_h2d_bytes << '\n'
              << "  peer activation bytes: " << s.peer_activation_bytes << '\n'
              << "  budget violations:     " << s.cold_budget_violations << '\n';
}

void print_json(ReplacementPolicy policy, const strata::SimulationResult& result,
                bool first, bool array) {
    const auto& s = result.residency;
    if (array && !first) std::cout << ',';
    std::cout << "{\"policy\":\"" << strata::to_string(policy)
              << "\",\"events\":" << result.events
              << ",\"accesses\":" << s.accesses
              << ",\"hit_rate_pct\":" << std::fixed << std::setprecision(6) << hit_rate(s)
              << ",\"vram_hits\":" << s.vram_hits
              << ",\"ram_hits\":" << s.ram_hits
              << ",\"peer_hits\":" << s.peer_hits
              << ",\"nvme_misses\":" << s.nvme_misses
              << ",\"nvme_read_bytes\":" << s.nvme_read_bytes
              << ",\"nvme_prefetch_bytes\":" << s.nvme_prefetch_bytes
              << ",\"weight_h2d_bytes\":" << s.weight_h2d_bytes
              << ",\"peer_activation_bytes\":" << s.peer_activation_bytes
              << ",\"prefetches\":" << s.prefetches
              << ",\"useful_prefetches\":" << s.useful_prefetches
              << ",\"wasted_prefetches\":" << s.wasted_prefetches
              << ",\"cold_budget_violations\":" << s.cold_budget_violations
              << ",\"transitions_learned\":" << result.transitions_learned << '}';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const auto trace = strata::parse_route_trace(options.trace);
        if (!trace.ok()) {
            for (const auto& error : trace.errors) std::cerr << "error: " << error << '\n';
            return 2;
        }

        std::vector<ReplacementPolicy> policies{options.policy};
        if (options.compare) {
            policies = {ReplacementPolicy::Lru, ReplacementPolicy::Lfu,
                        ReplacementPolicy::Lease};
        }
        if (options.json && options.compare) std::cout << '[';
        bool first = true;
        for (const auto policy : policies) {
            const auto result = strata::simulate(trace.events, make_config(options, policy));
            if (options.json) print_json(policy, result, first, options.compare);
            else print_human(policy, result);
            first = false;
        }
        if (options.json && options.compare) std::cout << ']';
        if (options.json) std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
