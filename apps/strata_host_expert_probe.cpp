#include "strata/checkpoint.hpp"
#include "strata/glm_int4.hpp"
#include "strata/glm_ops.hpp"
#include "strata/worker_pool.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint64_t kHidden = 6144U;
constexpr std::uint64_t kIntermediate = 2048U;

struct OwnedMatrix {
    std::span<const std::byte> packed;
    std::span<const std::byte> scales;
    strata::GlmInt4MatrixView view;
};

strata::ParseResult<OwnedMatrix> load_matrix(
    const strata::GlmCheckpointReader& checkpoint, const std::string& base,
    std::uint64_t rows, std::uint64_t columns) {
    strata::ParseResult<OwnedMatrix> result;
    const auto* packed = checkpoint.find(base + ".weight_packed");
    const auto* scales = checkpoint.find(base + ".weight_scale");
    const auto packed_columns = (columns + 7U) / 8U;
    const auto scale_columns = (columns + 127U) / 128U;
    if (packed == nullptr || scales == nullptr ||
        packed->encoding != strata::GlmTensorEncoding::Int4Group128 ||
        scales->encoding != strata::GlmTensorEncoding::Int4Group128 ||
        packed->source_shape != std::vector<std::uint64_t>{rows, packed_columns} ||
        scales->source_shape != std::vector<std::uint64_t>{rows, scale_columns}) {
        result.errors.emplace_back("probe matrix is not target INT4 group-128: " + base);
        return result;
    }
    auto packed_data = checkpoint.view(packed->name);
    auto scale_data = checkpoint.view(scales->name);
    if (!packed_data.ok()) {
        result.errors = std::move(packed_data.errors);
        return result;
    }
    if (!scale_data.ok()) {
        result.errors = std::move(scale_data.errors);
        return result;
    }
    result.value.packed = packed_data.value;
    result.value.scales = scale_data.value;
    result.value.view = {result.value.packed, result.value.scales, rows, columns,
                         packed_columns, scale_columns, 128U};
    return result;
}

bool parse_u32(std::string_view text, std::uint32_t& output) {
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), output);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

}  // namespace

int main(int argc, char** argv) {
    std::string model;
    std::uint32_t layer = 3U;
    std::uint32_t expert = 0U;
    std::uint32_t iterations = 11U;
    std::uint32_t threads = 28U;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto value = [&]() -> const char* {
            return index + 1 < argc ? argv[++index] : nullptr;
        };
        if (argument == "--model") {
            const auto* next = value();
            if (next == nullptr) return 2;
            model = next;
        } else if (argument == "--layer") {
            const auto* next = value();
            if (next == nullptr || !parse_u32(next, layer)) return 2;
        } else if (argument == "--expert") {
            const auto* next = value();
            if (next == nullptr || !parse_u32(next, expert)) return 2;
        } else if (argument == "--iterations") {
            const auto* next = value();
            if (next == nullptr || !parse_u32(next, iterations) || iterations == 0U) return 2;
        } else if (argument == "--threads") {
            const auto* next = value();
            if (next == nullptr || !parse_u32(next, threads) || threads == 0U) return 2;
        } else {
            return 2;
        }
    }
    if (model.empty() || layer < 3U || layer >= 78U || expert >= 256U) return 2;
    auto checkpoint = strata::GlmCheckpointReader::open(model, false);
    if (!checkpoint.ok()) {
        for (const auto& error : checkpoint.errors) std::cerr << error << '\n';
        return 1;
    }
    const auto prefix = "model.layers." + std::to_string(layer) +
                        ".mlp.experts." + std::to_string(expert) + ".";
    auto gate = load_matrix(*checkpoint.value, prefix + "gate_proj",
                            kIntermediate, kHidden);
    auto up = load_matrix(*checkpoint.value, prefix + "up_proj",
                          kIntermediate, kHidden);
    auto down = load_matrix(*checkpoint.value, prefix + "down_proj",
                            kHidden, kIntermediate);
    for (const auto* loaded : {&gate, &up, &down}) {
        if (!loaded->ok()) {
            for (const auto& error : loaded->errors) std::cerr << error << '\n';
            return 1;
        }
    }
    std::vector<float> input(kHidden);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = std::sin(static_cast<float>(index) * 0.013F);
    }
    std::vector<float> gate_output(kIntermediate);
    std::vector<float> up_output(kIntermediate);
    std::vector<float> output(kHidden);
    strata::HostWorkerPool workers(threads);
    const auto parallel_matvec = [&workers](
        const strata::GlmInt4MatrixView& matrix, std::span<const float> source,
        std::span<float> destination) {
        constexpr std::uint64_t rows_per_task = 64U;
        const auto tasks = static_cast<std::size_t>(
            (matrix.rows + rows_per_task - 1U) / rows_per_task);
        return workers.parallel_for(tasks, [&](std::size_t task) {
            const auto begin = static_cast<std::uint64_t>(task) * rows_per_task;
            const auto end = std::min(matrix.rows, begin + rows_per_task);
            static_cast<void>(strata::glm_int4_group128_matvec_rows(
                matrix, source, destination, begin, end));
        });
    };
    const auto execute = [&] {
        auto status = parallel_matvec(
            gate.value.view, input, gate_output);
        if (!status.ok()) return status;
        status = parallel_matvec(up.value.view, input, up_output);
        if (!status.ok()) return status;
        for (std::size_t index = 0; index < gate_output.size(); ++index) {
            gate_output[index] = strata::glm_silu_f32(gate_output[index]) * up_output[index];
        }
        return parallel_matvec(
            down.value.view, gate_output, output);
    };
    if (!execute().ok()) return 1;
    std::vector<double> milliseconds;
    milliseconds.reserve(iterations);
    for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration) {
        const auto started = std::chrono::steady_clock::now();
        if (!execute().ok()) return 1;
        milliseconds.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count());
    }
    std::sort(milliseconds.begin(), milliseconds.end());
    const double median = milliseconds[milliseconds.size() / 2U];
    const std::uint64_t weight_bytes =
        gate.value.view.packed.size() + gate.value.view.scales.size() +
        up.value.view.packed.size() + up.value.view.scales.size() +
        down.value.view.packed.size() + down.value.view.scales.size();
    std::cout << std::setprecision(10)
              << "{\"layer\":" << layer << ",\"expert\":" << expert
              << ",\"iterations\":" << iterations
              << ",\"threads\":" << threads
              << ",\"weight_bytes\":" << weight_bytes
              << ",\"service_milliseconds\":[";
    for (std::size_t index = 0; index < milliseconds.size(); ++index) {
        if (index != 0U) std::cout << ',';
        std::cout << milliseconds[index];
    }
    std::cout << "],\"median_milliseconds\":" << median
              << ",\"effective_weight_gb_s\":"
              << static_cast<double>(weight_bytes) / (median * 1.0e6)
              << "}\n";
    return 0;
}
