// Offline planner for the static routed-expert residency tier.
//
// DeepSeek-V4-Flash activates 3.449 GB of routed-expert weight per decode
// token -- 43 layers x top-6 x 13.37 MB -- and on this class of machine those
// bytes come from host DRAM, which is the decode bottleneck. The only store
// that is not DRAM is VRAM, and the routed set (143.7 GB) does not fit in it.
//
// It does not have to. Decode routing is measurably concentrated: over a
// 127-token decode window the model touches 120.2 distinct experts per layer
// against 243.4 under a uniform-random null, a 2.03x concentration. More
// usefully, the concentration is a property of the *model*, not of one
// conversation. Taking the hottest 10% of (layer, expert) triplets from one
// prompt's trace and evaluating them against a completely different prompt's
// trace covers 38.6% of its decode activations, against 10.4% for a random set
// of the same size -- a 3.7x lift on held-out data.
//
// So a small, permanently pinned tier pays for itself. This tool chooses it.
// It reads one or more route traces, counts decode activations per (layer,
// expert), and emits the hottest triplets that fit a byte budget, ordered by
// count. The runtime pins exactly that set and never evicts it, so there is no
// cache lookup, no eviction policy and no coherency traffic on the decode path
// -- only a residency test the device can make from a constant bitmap.
//
// Prefill activations are deliberately not counted. Prefill touches most of
// the set (82% of it at 3,565 tokens), so including it flattens the ranking
// toward uniform and describes a phase this tier does not exist to serve.
//
// This is placement, not prediction: it changes which device holds a weight,
// never which experts the router selects.

#include "cli_common.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

// DeepSeek-V4-Flash: hidden 4096, expert intermediate 2048, FP4 weights with
// one E8M0 scale per 32 values. w1 and w3 are [2048, 4096], w2 is [4096, 2048].
constexpr std::uint32_t kLayers = 43U;
constexpr std::uint32_t kExperts = 256U;
constexpr std::uint64_t kTripletBytes = 13369344ULL;

// Byte sizes with K/M/G suffixes, matching strata-deepseek-run's spelling so
// an operator writes the same thing to both tools.
bool parse_bytes(std::string_view text, std::uint64_t& value) {
    if (text.empty()) return false;
    std::uint64_t multiplier = 1U;
    const char suffix = text.back();
    if (suffix == 'K' || suffix == 'k') { multiplier = 1ULL << 10U; text.remove_suffix(1U); }
    else if (suffix == 'M' || suffix == 'm') { multiplier = 1ULL << 20U; text.remove_suffix(1U); }
    else if (suffix == 'G' || suffix == 'g') { multiplier = 1ULL << 30U; text.remove_suffix(1U); }
    std::uint64_t base = 0U;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), base);
    if (text.empty() || parsed.ec != std::errc{} ||
        parsed.ptr != text.data() + text.size() ||
        base > std::numeric_limits<std::uint64_t>::max() / multiplier) {
        return false;
    }
    value = base * multiplier;
    return true;
}

struct Options {
    std::vector<std::string> traces;
    std::string output;
    std::uint64_t budget_bytes{};
    bool quiet{};
};

void usage() {
    std::cerr
        << "usage: strata-dsv4-expert-residency --trace PATH [--trace PATH ...]\n"
        << "       --budget BYTES --output PATH [--quiet]\n\n"
        << "Chooses the static routed-expert residency tier from route traces.\n"
        << "--budget accepts suffixes (14G sizes the tier for a 16 GiB card).\n"
        << "Only decode activations are counted; see the file header for why.\n";
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--help" || argument == "-h") {
            usage();
            std::exit(0);
        }
        if (argument == "--quiet") {
            options.quiet = true;
            continue;
        }
        if (index + 1 >= argc) return false;
        const std::string_view value(argv[++index]);
        if (argument == "--trace") options.traces.emplace_back(value);
        else if (argument == "--output") options.output = value;
        else if (argument == "--budget") {
            if (!parse_bytes(value, options.budget_bytes)) return false;
        } else return false;
    }
    return !options.traces.empty() && !options.output.empty() &&
           options.budget_bytes >= kTripletBytes;
}

// The trace is one JSON object per line. Only three fields are needed and the
// format is fixed by the emitter, so a scan for each key is enough and avoids
// pulling a JSON parser into a tool that reads millions of lines.
[[nodiscard]] bool field(std::string_view line, std::string_view key,
                         std::string_view& out) {
    const auto at = line.find(key);
    if (at == std::string_view::npos) return false;
    auto rest = line.substr(at + key.size());
    const auto begin = rest.find_first_not_of(" \t:\"");
    if (begin == std::string_view::npos) return false;
    rest = rest.substr(begin);
    const auto end = rest.find_first_of(",}\"]");
    out = rest.substr(0U, end == std::string_view::npos ? rest.size() : end);
    return true;
}

[[nodiscard]] bool parse_u32(std::string_view text, std::uint32_t& out) {
    const auto result = std::from_chars(text.data(), text.data() + text.size(), out);
    return result.ec == std::errc{};
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        usage();
        return 2;
    }

    std::vector<std::uint64_t> counts(
        static_cast<std::size_t>(kLayers) * kExperts, 0U);
    std::uint64_t decode_activations = 0U;
    std::uint64_t decode_rows = 0U;
    std::uint64_t skipped_prefill = 0U;

    for (const auto& path : options.traces) {
        std::ifstream file(path);
        if (!file) {
            std::cerr << "error: cannot open trace " << path << "\n";
            return 1;
        }
        std::string line;
        while (std::getline(file, line)) {
            std::string_view view(line);
            std::string_view phase;
            if (!field(view, "\"phase\"", phase)) continue;
            if (phase != "decode") {
                ++skipped_prefill;
                continue;
            }
            std::string_view layer_text;
            if (!field(view, "\"layer\"", layer_text)) continue;
            std::uint32_t layer = 0U;
            if (!parse_u32(layer_text, layer) || layer >= kLayers) {
                std::cerr << "error: trace names layer " << layer_text
                          << ", outside the declared " << kLayers << "\n";
                return 1;
            }
            const auto experts_at = view.find("\"experts\"");
            if (experts_at == std::string_view::npos) continue;
            auto rest = view.substr(experts_at);
            const auto open = rest.find('[');
            const auto close = rest.find(']');
            if (open == std::string_view::npos || close == std::string_view::npos ||
                close < open) continue;
            auto list = rest.substr(open + 1U, close - open - 1U);
            ++decode_rows;
            while (!list.empty()) {
                const auto comma = list.find(',');
                const auto token = list.substr(0U, comma);
                std::uint32_t expert = 0U;
                const auto begin = token.find_first_not_of(" \t");
                if (begin != std::string_view::npos &&
                    parse_u32(token.substr(begin), expert)) {
                    if (expert >= kExperts) {
                        std::cerr << "error: trace names expert " << expert
                                  << ", outside the declared " << kExperts << "\n";
                        return 1;
                    }
                    ++counts[static_cast<std::size_t>(layer) * kExperts + expert];
                    ++decode_activations;
                }
                if (comma == std::string_view::npos) break;
                list = list.substr(comma + 1U);
            }
        }
    }

    if (decode_activations == 0U) {
        std::cerr << "error: the traces contain no decode activations; this tier "
                     "is chosen from decode routing only\n";
        return 1;
    }

    std::vector<std::uint32_t> order(counts.size());
    for (std::uint32_t index = 0U; index < order.size(); ++index) order[index] = index;
    // Ties broken by index so the plan is byte-identical across runs on the
    // same traces; a residency plan that changes under reordering is not one.
    std::stable_sort(order.begin(), order.end(),
                     [&counts](std::uint32_t left, std::uint32_t right) {
                         if (counts[left] != counts[right]) {
                             return counts[left] > counts[right];
                         }
                         return left < right;
                     });

    const auto capacity = static_cast<std::size_t>(
        options.budget_bytes / kTripletBytes);
    std::uint64_t covered = 0U;
    std::size_t chosen = 0U;
    for (; chosen < capacity && chosen < order.size(); ++chosen) {
        const auto count = counts[order[chosen]];
        if (count == 0U) break;  // never pin a triplet the traces never touched
        covered += count;
    }

    std::ofstream out(options.output, std::ios::trunc);
    if (!out) {
        std::cerr << "error: cannot write " << options.output << "\n";
        return 1;
    }
    out << "strata.dsv4_expert_residency 1\n"
        << "layers " << kLayers << " experts " << kExperts
        << " triplet_bytes " << kTripletBytes << "\n"
        << "decode_activations " << decode_activations
        << " decode_rows " << decode_rows << "\n"
        << "pairs " << chosen << " bytes " << chosen * kTripletBytes << "\n";
    for (std::size_t index = 0U; index < chosen; ++index) {
        const auto flat = order[index];
        out << (flat / kExperts) << ' ' << (flat % kExperts) << ' '
            << counts[flat] << '\n';
    }
    if (!out.flush()) {
        std::cerr << "error: writing " << options.output << " failed\n";
        return 1;
    }

    if (!options.quiet) {
        std::printf(
            "traces %zu | decode rows %llu | activations %llu | prefill rows skipped %llu\n",
            options.traces.size(),
            static_cast<unsigned long long>(decode_rows),
            static_cast<unsigned long long>(decode_activations),
            static_cast<unsigned long long>(skipped_prefill));
        std::printf(
            "budget %.2f GB -> %zu triplets (%.2f GB), covering %.2f%% of the "
            "traces' decode activations\n",
            static_cast<double>(options.budget_bytes) / 1.0e9, chosen,
            static_cast<double>(chosen * kTripletBytes) / 1.0e9,
            100.0 * static_cast<double>(covered) /
                static_cast<double>(decode_activations));
        std::printf(
            "note: that coverage is self-measured on these traces. The held-out "
            "figure is the one that matters; see docs/experiments/0124.\n");
    }
    return 0;
}
