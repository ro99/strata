#include "strata/cuda_backend.hpp"
#include "strata/laguna_checkpoint.hpp"
#include "strata/worker_pool.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint32_t kWarmups = 5U;
constexpr std::uint32_t kSamples = 21U;

struct Shape {
    const char* label;
    const char* tensor;
    std::uint64_t rows;
    std::uint64_t columns;
};

constexpr std::array<Shape, 4> kShapes{{
    {"sliding_q", "model.layers.1.self_attn.q_proj", 9'216U, 3'072U},
    {"sliding_o", "model.layers.1.self_attn.o_proj", 3'072U, 9'216U},
    {"shared_gate", "model.layers.1.mlp.shared_expert.gate_proj", 1'024U,
     3'072U},
    {"lm_head", "lm_head", 100'352U, 3'072U},
}};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}

double milliseconds_since(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - started)
        .count();
}

void print_errors(const strata::ValidationResult& status) {
    for (const auto& error : status.errors) std::cerr << "error: " << error << '\n';
}

strata::ValidationResult upload_rows(
    strata::CudaBackend& backend, int device,
    const strata::LagunaLinear& module, std::span<const std::byte> source,
    std::uint64_t row_begin, std::uint64_t row_end,
    strata::CudaWeight& output) {
    strata::ValidationResult result;
    if (module.encoding != strata::LagunaTensorEncoding::Plain ||
        module.weight == nullptr ||
        module.weight->dtype != strata::SafetensorsDtype::Bf16 ||
        row_begin >= row_end || row_end > module.rows) {
        result.errors.emplace_back("Laguna spine TP probe needs a BF16 row slice");
        return result;
    }
    const auto row_bytes = module.columns * sizeof(std::uint16_t);
    const auto offset = row_begin * row_bytes;
    const auto bytes = (row_end - row_begin) * row_bytes;
    if (offset > source.size() || bytes > source.size() - offset) {
        result.errors.emplace_back("Laguna spine TP row slice exceeds its tensor");
        return result;
    }
    strata::CudaWeightDescriptor descriptor;
    descriptor.encoding = strata::CudaWeightEncoding::Plain;
    descriptor.dtype = strata::SafetensorsDtype::Bf16;
    descriptor.rows = row_end - row_begin;
    descriptor.columns = module.columns;
    return backend.upload(
        device, descriptor,
        source.subspan(static_cast<std::size_t>(offset),
                       static_cast<std::size_t>(bytes)),
        {}, output);
}

bool run_shape(const Shape& shape, const strata::LagunaCheckpointReader& checkpoint,
               strata::CudaBackend& backend, std::span<const int, 2> devices,
               strata::HostWorkerPool& workers) {
    auto module = checkpoint.linear(shape.tensor, shape.rows, shape.columns);
    if (!module.ok()) {
        for (const auto& error : module.errors) std::cerr << "error: " << error << '\n';
        return false;
    }
    auto source = checkpoint.view(module.value.weight->name);
    if (!source.ok()) {
        for (const auto& error : source.errors) std::cerr << "error: " << error << '\n';
        return false;
    }
    strata::CudaWeight canonical;
    std::array<strata::CudaWeight, 2> shards;
    auto status = upload_rows(backend, devices[0], module.value, source.value,
                              0U, shape.rows, canonical);
    if (!status.ok()) {
        print_errors(status);
        return false;
    }
    const auto split = shape.rows / 2U;
    status = upload_rows(backend, devices[0], module.value, source.value, 0U,
                         split, shards[0]);
    if (!status.ok()) {
        print_errors(status);
        return false;
    }
    status = upload_rows(backend, devices[1], module.value, source.value, split,
                         shape.rows, shards[1]);
    if (!status.ok()) {
        print_errors(status);
        return false;
    }
    for (const auto device : devices) {
        status = backend.synchronize_uploads(device);
        if (!status.ok()) {
            print_errors(status);
            return false;
        }
    }

    std::vector<float> input(shape.columns);
    for (std::size_t index = 0U; index < input.size(); ++index) {
        input[index] = static_cast<float>(static_cast<int>(index % 29U) - 14) /
                       32.0F;
    }
    std::vector<float> canonical_output(shape.rows);
    std::vector<float> sharded_output(shape.rows);
    std::array<strata::ValidationResult, 2> shard_status;
    const auto run_canonical = [&]() {
        return backend.matmul(canonical, input, 1U, canonical_output);
    };
    const auto run_sharded = [&]() {
        shard_status = {};
        auto dispatched = workers.parallel_for(2U, [&](std::size_t shard) {
            const auto begin = shard == 0U ? 0U : split;
            const auto end = shard == 0U ? split : shape.rows;
            shard_status[shard] = backend.matmul(
                shards[shard], input, 1U,
                std::span<float>(sharded_output).subspan(
                    static_cast<std::size_t>(begin),
                    static_cast<std::size_t>(end - begin)));
        });
        if (!dispatched.ok()) return dispatched;
        strata::ValidationResult combined;
        for (auto& item : shard_status) {
            combined.errors.insert(combined.errors.end(),
                                   item.errors.begin(), item.errors.end());
        }
        return combined;
    };

    for (std::uint32_t warmup = 0U; warmup < kWarmups; ++warmup) {
        status = run_canonical();
        if (!status.ok()) {
            print_errors(status);
            return false;
        }
        status = run_sharded();
        if (!status.ok()) {
            print_errors(status);
            return false;
        }
    }
    std::vector<double> canonical_samples;
    std::vector<double> sharded_samples;
    canonical_samples.reserve(kSamples);
    sharded_samples.reserve(kSamples);
    for (std::uint32_t sample = 0U; sample < kSamples; ++sample) {
        auto started = std::chrono::steady_clock::now();
        status = run_canonical();
        canonical_samples.push_back(milliseconds_since(started));
        if (!status.ok()) {
            print_errors(status);
            return false;
        }
        started = std::chrono::steady_clock::now();
        status = run_sharded();
        sharded_samples.push_back(milliseconds_since(started));
        if (!status.ok()) {
            print_errors(status);
            return false;
        }
    }
    std::uint64_t mismatches = 0U;
    for (std::size_t index = 0U; index < canonical_output.size(); ++index) {
        if (std::bit_cast<std::uint32_t>(canonical_output[index]) !=
            std::bit_cast<std::uint32_t>(sharded_output[index])) {
            ++mismatches;
        }
    }
    const auto canonical_ms = median(std::move(canonical_samples));
    const auto sharded_ms = median(std::move(sharded_samples));
    const auto weight_bytes = shape.rows * shape.columns * sizeof(std::uint16_t);
    const auto canonical_transfer =
        shape.columns * sizeof(float) + shape.rows * sizeof(float);
    const auto sharded_transfer =
        2U * shape.columns * sizeof(float) + shape.rows * sizeof(float);
    std::cout << std::fixed << std::setprecision(3)
              << shape.label << " rows=" << shape.rows
              << " columns=" << shape.columns
              << " weight_mib=" << static_cast<double>(weight_bytes) / 1048576.0
              << " canonical_ms=" << canonical_ms
              << " sharded_ms=" << sharded_ms
              << " speedup=" << canonical_ms / sharded_ms
              << " canonical_gbs=" << static_cast<double>(weight_bytes) /
                                            (canonical_ms * 1.0e6)
              << " sharded_gbs=" << static_cast<double>(weight_bytes) /
                                          (sharded_ms * 1.0e6)
              << " canonical_transfer_kib="
              << static_cast<double>(canonical_transfer) / 1024.0
              << " sharded_transfer_kib="
              << static_cast<double>(sharded_transfer) / 1024.0
              << " mismatches=" << mismatches << '\n';
    return mismatches == 0U;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model{"models/laguna"};
    std::array<int, 2> devices{0, 1};
    for (int index = 1; index < argc; ++index) {
        const std::string_view flag(argv[index]);
        const auto next = [&]() -> std::string_view {
            return index + 1 < argc ? std::string_view(argv[++index])
                                    : std::string_view();
        };
        if (flag == "--model") {
            model = std::string(next());
        } else if (flag == "--devices") {
            const auto text = next();
            const auto comma = text.find(',');
            if (comma == std::string_view::npos) return EXIT_FAILURE;
            devices[0] = std::stoi(std::string(text.substr(0U, comma)));
            devices[1] = std::stoi(std::string(text.substr(comma + 1U)));
        } else {
            std::cerr << "usage: strata-laguna-spine-tp-probe "
                         "[--model DIR] [--devices A,B]\n";
            return EXIT_FAILURE;
        }
    }
    if (!strata::CudaBackend::compiled() || devices[0] == devices[1]) {
        std::cerr << "error: the Laguna spine TP probe needs two CUDA devices\n";
        return EXIT_FAILURE;
    }
    auto checkpoint = strata::LagunaCheckpointReader::open(model);
    if (!checkpoint.ok()) {
        for (const auto& error : checkpoint.errors) std::cerr << "error: " << error << '\n';
        return EXIT_FAILURE;
    }
    strata::CudaBackend backend;
    auto status = backend.initialize(devices, false);
    if (!status.ok()) {
        print_errors(status);
        return EXIT_FAILURE;
    }
    strata::HostWorkerPool workers(2U);
    for (const auto& shape : kShapes) {
        if (!run_shape(shape, *checkpoint.value, backend, devices, workers)) {
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
