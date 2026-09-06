// One layer-page of GLM-5.3 sparse MLA attention, at any history, in seconds.
//
// The campaign's objective is prefill at 32,768 tokens, and until now nothing
// has measured that shape: an 8,192-token prefill arm is twenty minutes and a
// 32,768 one is hours, so every decision was taken from 619- and 2,591-token
// runs and two of them turned out to be regime artifacts (records 0266, 0267).
// This binary loads the real weights and runs the production attention path
// over a fabricated cache, so `--history 30720` costs one page of work instead
// of thirty. Fabricated state means the OUTPUT is noise -- see the caveat on
// `Glm53AttentionBenchRequest`; what is real here is the timing split, the
// group structure, and their scaling in history.
//
//   strata-glm53-attnbench --model <dir> --devices 1,2 --context 32768
//       --layer 19 --rows 2048 --sweep 2560,4096,8192,16384,30720
//
// Calibrate before believing it: run a history the arms have also run and
// check the group counts against that record.

#include "strata/models/glm53/glm53_runtime.hpp"

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

[[nodiscard]] bool parse_u32(std::string_view text, std::uint32_t& out) {
    const auto* first = text.data();
    const auto* last = first + text.size();
    const auto parsed = std::from_chars(first, last, out);
    return parsed.ec == std::errc{} && parsed.ptr == last;
}

[[nodiscard]] std::vector<std::string_view> split(std::string_view text,
                                                  char separator) {
    std::vector<std::string_view> parts;
    std::size_t begin = 0U;
    while (begin <= text.size()) {
        const auto found = text.find(separator, begin);
        const auto end = found == std::string_view::npos ? text.size() : found;
        if (end > begin) parts.push_back(text.substr(begin, end - begin));
        if (found == std::string_view::npos) break;
        begin = found + 1U;
    }
    return parts;
}

void usage() {
    std::cerr
        << "usage: strata-glm53-attnbench --model <dir> [--devices 1,2]\n"
           "         [--context 32768] [--layer 19] [--rows 2048]\n"
           "         [--history N | --sweep N,N,...] [--repeats 1] [--seed S]\n"
           "         [--smoothing 0.0]  # calibrate group count against a real arm\n";
}

constexpr double kNanosecondsPerSecond = 1.0e9;

double seconds(std::uint64_t nanoseconds) {
    return static_cast<double>(nanoseconds) / kNanosecondsPerSecond;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model;
    std::vector<int> devices;
    std::uint32_t context = 32'768U;
    std::uint32_t layer = 19U;
    std::uint32_t rows = 2'048U;
    std::uint32_t repeats = 1U;
    std::uint64_t seed = 0x9E3779B97F4A7C15ULL;
    std::vector<float> smoothings;
    std::vector<std::uint32_t> histories;

    for (int index = 1; index < argc; ++index) {
        const std::string_view flag{argv[index]};
        const auto value = [&]() -> std::string_view {
            return index + 1 < argc ? std::string_view{argv[++index]}
                                    : std::string_view{};
        };
        bool ok = true;
        if (flag == "--model") {
            model = std::string(value());
        } else if (flag == "--devices") {
            for (const auto part : split(value(), ',')) {
                std::uint32_t device = 0U;
                ok = ok && parse_u32(part, device);
                devices.push_back(static_cast<int>(device));
            }
        } else if (flag == "--context") {
            ok = parse_u32(value(), context);
        } else if (flag == "--layer") {
            ok = parse_u32(value(), layer);
        } else if (flag == "--rows") {
            ok = parse_u32(value(), rows);
        } else if (flag == "--repeats") {
            ok = parse_u32(value(), repeats);
        } else if (flag == "--smoothing") {
            for (const auto part : split(value(), ',')) {
                smoothings.push_back(
                    std::strtof(std::string(part).c_str(), nullptr));
            }
        } else if (flag == "--seed") {
            seed = std::strtoull(std::string(value()).c_str(), nullptr, 0);
        } else if (flag == "--history") {
            std::uint32_t history = 0U;
            ok = parse_u32(value(), history);
            histories.push_back(history);
        } else if (flag == "--sweep") {
            for (const auto part : split(value(), ',')) {
                std::uint32_t history = 0U;
                ok = ok && parse_u32(part, history);
                histories.push_back(history);
            }
        } else {
            usage();
            return 2;
        }
        if (!ok) {
            usage();
            return 2;
        }
    }
    if (model.empty()) {
        usage();
        return 2;
    }
    if (histories.empty()) histories.push_back(context - rows);
    if (smoothings.empty()) smoothings.push_back(0.0F);

    strata::Glm53RuntimeConfig config;
    config.devices = devices;
    config.maximum_context_tokens = context;
    config.load_progress = true;
    // The group and split diagnostics are gated on it, and they are the
    // point of the bench rather than a side channel.
    config.phase_profile = true;

    strata::Glm53Runtime runtime;
    const auto initialized = runtime.initialize(model, config);
    if (!initialized.ok()) {
        for (const auto& error : initialized.errors) {
            std::cerr << "error: " << error << '\n';
        }
        return 1;
    }

    std::cout << std::fixed << std::setprecision(4);
    for (const auto smoothing : smoothings)
    for (const auto history : histories) {
        strata::Glm53AttentionBenchRequest request;
        request.layer = layer;
        request.history = history;
        request.rows = rows;
        request.repeats = repeats;
        request.seed = seed;
        request.smoothing = smoothing;
        strata::Glm53AttentionBenchResult result;
        const auto ran = runtime.attention_bench(request, result);
        if (!ran.ok()) {
            for (const auto& error : ran.errors) {
                std::cerr << "error: " << error << '\n';
            }
            return 1;
        }
        std::cout << "[attnbench] layer=" << layer
                  << " smoothing=" << smoothing << " history=" << history
                  << " rows=" << rows
                  << " wall_s=" << seconds(result.wall_nanoseconds)
                  << " prelude_s=" << seconds(result.prelude_nanoseconds)
                  << " groups_s=" << seconds(result.groups_nanoseconds)
                  << " expand_s=" << seconds(result.expand_nanoseconds)
                  << " index_s=" << seconds(result.index_nanoseconds)
                  << " qk_s=" << seconds(result.qk_nanoseconds)
                  << " softmax_s=" << seconds(result.softmax_nanoseconds)
                  << " av_s=" << seconds(result.av_nanoseconds)
                  << " devicecall_s=" << seconds(result.device_nanoseconds)
                  << " ms_per_row="
                  << (rows == 0U
                          ? 0.0
                          : seconds(result.wall_nanoseconds) * 1.0e3 / rows)
                  << " checksum=0x" << std::hex << result.checksum << std::dec
                  << '\n'
                  << std::flush;
    }
    return 0;
}
