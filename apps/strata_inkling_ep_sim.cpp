#include "strata/trace.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

struct Options {
    std::string trace;
    std::vector<std::uint64_t> capacities;
    std::vector<std::size_t> schedule;
    std::uint64_t expert_bytes{13'369'344ULL};
    bool decode_only{true};
};

template <typename T>
std::vector<T> parse_list(std::string_view text) {
    std::vector<T> values;
    std::size_t begin = 0U;
    while (begin <= text.size()) {
        const auto comma = text.find(',', begin);
        const auto piece = text.substr(
            begin, comma == std::string_view::npos ? text.size() - begin
                                                   : comma - begin);
        if (piece.empty()) throw std::invalid_argument("empty list member");
        values.push_back(static_cast<T>(std::stoull(std::string(piece))));
        if (comma == std::string_view::npos) break;
        begin = comma + 1U;
    }
    return values;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view flag(argv[index]);
        const auto next = [&]() -> std::string_view {
            if (++index >= argc) {
                throw std::invalid_argument("missing value after " +
                                            std::string(flag));
            }
            return argv[index];
        };
        if (flag == "--trace") options.trace = std::string(next());
        else if (flag == "--capacities") {
            options.capacities = parse_list<std::uint64_t>(next());
        } else if (flag == "--schedule") {
            options.schedule = parse_list<std::size_t>(next());
        } else if (flag == "--expert-bytes") {
            options.expert_bytes = std::stoull(std::string(next()));
        } else if (flag == "--all-phases") {
            options.decode_only = false;
        } else {
            throw std::invalid_argument("unknown option: " + std::string(flag));
        }
    }
    if (options.trace.empty() || options.capacities.empty() ||
        options.schedule.empty() || options.expert_bytes == 0U) {
        throw std::invalid_argument(
            "usage: strata-inkling-ep-sim --trace FILE --capacities B0,B1,... "
            "--schedule SLOT,SLOT,... [--expert-bytes B] [--all-phases]");
    }
    for (const auto slot : options.schedule) {
        if (slot >= options.capacities.size()) {
            throw std::invalid_argument("schedule slot exceeds capacities");
        }
    }
    return options;
}

class Lru {
public:
    explicit Lru(std::uint64_t entries) : capacity_(entries) {
        if (capacity_ == 0U) throw std::invalid_argument("zero-entry cache");
    }

    bool touch(std::uint64_t key) {
        const auto found = locations_.find(key);
        if (found != locations_.end()) {
            recency_.splice(recency_.begin(), recency_, found->second);
            found->second = recency_.begin();
            return true;
        }
        if (recency_.size() == capacity_) {
            locations_.erase(recency_.back());
            recency_.pop_back();
        }
        recency_.push_front(key);
        locations_[key] = recency_.begin();
        return false;
    }

private:
    std::size_t capacity_{};
    std::list<std::uint64_t> recency_;
    std::unordered_map<std::uint64_t, std::list<std::uint64_t>::iterator>
        locations_;
};

struct Result {
    std::uint64_t events{};
    std::uint64_t selections{};
    std::uint64_t misses{};
    std::uint64_t critical_miss_slots{};
    std::uint64_t critical_compute_slots{};
    std::vector<std::uint64_t> device_selections;
    std::vector<std::uint64_t> device_misses;
};

template <typename Placement>
Result simulate(std::span<const strata::RouteEvent> events,
                const Options& options, Placement placement) {
    std::vector<Lru> caches;
    for (const auto bytes : options.capacities) {
        caches.emplace_back(bytes / options.expert_bytes);
    }
    Result result;
    result.device_selections.assign(caches.size(), 0U);
    result.device_misses.assign(caches.size(), 0U);
    for (const auto& event : events) {
        if (options.decode_only && event.phase != strata::RoutePhase::Decode) {
            continue;
        }
        std::vector<std::uint64_t> event_selections(caches.size(), 0U);
        std::vector<std::uint64_t> event_misses(caches.size(), 0U);
        for (const auto expert : event.experts) {
            const auto slot = placement(event.layer, expert);
            const auto key = (static_cast<std::uint64_t>(event.layer) << 32U) |
                             expert;
            ++event_selections[slot];
            ++result.device_selections[slot];
            if (!caches[slot].touch(key)) {
                ++event_misses[slot];
                ++result.device_misses[slot];
                ++result.misses;
            }
        }
        ++result.events;
        result.selections += event.experts.size();
        result.critical_compute_slots +=
            *std::max_element(event_selections.begin(), event_selections.end());
        result.critical_miss_slots +=
            *std::max_element(event_misses.begin(), event_misses.end());
    }
    return result;
}

void print_result(std::string_view name, const Result& result,
                  std::uint64_t expert_bytes) {
    std::cout << name << ": events=" << result.events
              << " selections=" << result.selections
              << " misses=" << result.misses
              << " staged_bytes=" << result.misses * expert_bytes
              << " critical_miss_slots=" << result.critical_miss_slots
              << " critical_compute_slots=" << result.critical_compute_slots
              << "\n  device selections:";
    for (const auto value : result.device_selections) std::cout << ' ' << value;
    std::cout << "\n  device misses:";
    for (const auto value : result.device_misses) std::cout << ' ' << value;
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const auto trace = strata::parse_route_trace(options.trace);
        if (!trace.ok()) {
            for (const auto& error : trace.errors) {
                std::cerr << "error: " << error << '\n';
            }
            return 2;
        }
        const auto devices = options.capacities.size();
        const auto control = simulate(
            trace.events, options,
            [devices](std::uint32_t layer, std::uint32_t) {
                return static_cast<std::size_t>(layer) % devices;
            });
        const auto candidate = simulate(
            trace.events, options,
            [&options](std::uint32_t layer, std::uint32_t expert) {
                constexpr std::uint64_t kExperts = 256U;
                const auto index =
                    (static_cast<std::uint64_t>(layer) * kExperts + expert) %
                    options.schedule.size();
                return options.schedule[index];
            });
        print_result("layer-local", control, options.expert_bytes);
        print_result("expert-parallel", candidate, options.expert_bytes);
        std::cout << "projected critical H2D speedup="
                  << static_cast<double>(control.critical_miss_slots) /
                         static_cast<double>(std::max<std::uint64_t>(
                             1U, candidate.critical_miss_slots))
                  << "x\nprojected critical expert-compute speedup="
                  << static_cast<double>(control.critical_compute_slots) /
                         static_cast<double>(std::max<std::uint64_t>(
                             1U, candidate.critical_compute_slots))
                  << "x\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
