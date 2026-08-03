// Microbenchmark of the Laguna routed-expert staging mechanism in isolation.
//
// Decode profiling attributes 191-248 ms per device per step to "miss staging"
// and reports an effective weight-path bandwidth of 1.8-3.3 GB/s against a
// PCIe gen3 x16 link rated at 15.75 GB/s with a warm page cache. That is an
// order-of-magnitude gap, so the question is which part of the staging path
// serializes. This probe reproduces the production access pattern -- a cold,
// randomly placed slice of the 99.7 GB mapping, never the same buffer twice --
// and times the candidate mechanisms against each other.
#include "strata/cuda_backend.hpp"
#include "strata/laguna_checkpoint.hpp"
#include "strata/model_adapter.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr auto& kContract = strata::kLagunaExecutionContract;

struct Options {
    std::string model{"models/laguna-s-21"};
    int device{1};
    std::uint32_t modules{192U};
    std::uint64_t arena_bytes{6ULL << 30U};
    std::uint64_t seed{20260803U};
    // The arena must be reserved before the first upload on a device, so the
    // cudaMalloc arms and the arena arms cannot share one process.
    std::string arms{"abc"};
};

double seconds_since(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
}

void report(std::string_view label, double seconds, std::uint64_t bytes,
            std::uint32_t modules) {
    std::cout << "  " << std::left << std::setw(46) << label << std::right
              << std::fixed << std::setprecision(2) << std::setw(9)
              << (1000.0 * seconds) << " ms  " << std::setw(7)
              << (seconds > 0.0 ? static_cast<double>(bytes) /
                                      (seconds * 1.0e9)
                                : 0.0)
              << " GB/s  " << std::setw(7) << std::setprecision(3)
              << (modules == 0U ? 0.0
                                : 1000.0 * seconds /
                                      static_cast<double>(modules))
              << " ms/module\n";
}

bool parse_u32(std::string_view text, std::uint32_t& output) {
    const auto* end = text.data() + text.size();
    return std::from_chars(text.data(), end, output).ptr == end;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view flag(argv[index]);
        const auto next = [&]() -> std::string_view {
            return index + 1 < argc ? std::string_view(argv[++index])
                                    : std::string_view();
        };
        if (flag == "--model") options.model = std::string(next());
        else if (flag == "--device") { std::uint32_t v = 0U; if (!parse_u32(next(), v)) return 2; options.device = static_cast<int>(v); }
        else if (flag == "--modules") { if (!parse_u32(next(), options.modules)) return 2; }
        else if (flag == "--arms") options.arms = std::string(next());
        else {
            std::cerr << "usage: strata-laguna-stage-probe [--model DIR] "
                         "[--device N] [--modules N] [--arms abc|de]\n";
            return 2;
        }
    }

    auto checkpoint = strata::LagunaCheckpointReader::open(options.model);
    if (!checkpoint.ok()) {
        for (const auto& error : checkpoint.errors) std::cerr << "error: " << error << '\n';
        return 1;
    }
    const auto& reader = *checkpoint.value;

    // Production picks a routed expert the router chose, which is effectively a
    // random slice of the mapping. Draw distinct (layer, expert, projection)
    // triples so no arm ever re-reads a buffer another arm warmed.
    std::mt19937_64 rng(options.seed);
    struct Target { strata::LagunaLinear module; std::string name; };
    std::vector<Target> targets;
    targets.reserve(options.modules);
    std::uniform_int_distribution<std::uint32_t> layer_pick(
        1U, kContract.quantized_expert_layers - 1U);
    std::uniform_int_distribution<std::uint32_t> expert_pick(
        0U, kContract.routed_experts - 1U);
    while (targets.size() < options.modules) {
        const auto layer = layer_pick(rng);
        const auto expert = expert_pick(rng);
        const auto base = "model.layers." + std::to_string(layer) +
                          ".mlp.experts." + std::to_string(expert) + ".gate_proj";
        auto module = reader.linear(base, kContract.expert_intermediate_size,
                                    kContract.hidden_size);
        if (!module.ok()) continue;
        if (std::any_of(targets.begin(), targets.end(),
                        [&base](const Target& value) { return value.name == base; })) {
            continue;
        }
        targets.push_back({module.value, base});
    }

    std::uint64_t total_bytes = 0U;
    for (const auto& target : targets) total_bytes += target.module.source_bytes();
    std::cout << "modules " << targets.size() << "  bytes "
              << std::fixed << std::setprecision(2)
              << static_cast<double>(total_bytes) / 1048576.0 << " MiB  device "
              << options.device << "\n";

    strata::CudaBackend backend;
    const std::array<int, 1> devices{options.device};
    if (auto status = backend.initialize(devices); !status.ok()) {
        for (const auto& error : status.errors) std::cerr << "error: " << error << '\n';
        return 1;
    }

    const bool run_abc = options.arms.find('a') != std::string::npos;
    const bool run_de = options.arms.find('d') != std::string::npos;

    if (run_abc) {
    // Arm A: what the runtime does today -- copy the tensor out of the mapping
    // into a fresh heap vector, cudaMalloc, then a pageable host-to-device copy.
    {
        const auto started = std::chrono::steady_clock::now();
        std::vector<strata::CudaWeight> held;
        held.reserve(targets.size());
        for (const auto& target : targets) {
            strata::CudaWeight weight;
            auto status = load_laguna_cuda_linear(reader, target.module,
                                                  options.device, backend, weight);
            if (!status.ok()) {
                for (const auto& error : status.errors) std::cerr << "error: " << error << '\n';
                return 1;
            }
            held.push_back(std::move(weight));
        }
        report("A read()+cudaMalloc+pageable  [current]",
               seconds_since(started), total_bytes,
               static_cast<std::uint32_t>(targets.size()));
    }
    reader.release_mapped_views();

    // Arm B: isolate the host-side copy. Time only read() into a heap vector,
    // with no device work at all.
    {
        const auto started = std::chrono::steady_clock::now();
        std::uint64_t sink = 0U;
        for (const auto& target : targets) {
            auto packed = reader.read(target.module.packed->name,
                                      target.module.packed->bytes);
            if (!packed.ok()) return 1;
            sink += packed.value.size();
        }
        report("B read() host copy only  [no device]",
               seconds_since(started), sink,
               static_cast<std::uint32_t>(targets.size()));
    }
    reader.release_mapped_views();

    // Arm C: isolate the mapping itself. Touch every page of the same tensors
    // through the zero-copy view, with no copy and no device work.
    {
        const auto started = std::chrono::steady_clock::now();
        std::uint64_t sink = 0U;
        std::uint64_t bytes = 0U;
        for (const auto& target : targets) {
            auto view = reader.view(target.module.packed->name);
            if (!view.ok()) return 1;
            bytes += view.value.size();
            for (std::size_t offset = 0U; offset < view.value.size();
                 offset += 4096U) {
                sink += static_cast<std::uint64_t>(view.value[offset]);
            }
        }
        report("C view() page touch only  [no copy]",
               seconds_since(started), bytes,
               static_cast<std::uint32_t>(targets.size()));
        if (sink == 0xFFFF'FFFF'FFFF'FFFFULL) std::cout << "";
    }
    }

    if (!run_de) return 0;
    // The remaining arms need an arena so no arm pays cudaMalloc per weight.
    if (auto status = backend.reserve_weight_arena(options.device,
                                                   options.arena_bytes);
        !status.ok()) {
        for (const auto& error : status.errors) std::cerr << "error: " << error << '\n';
        return 1;
    }

    const auto upload_view = [&](const Target& target,
                                 std::span<const std::byte> packed,
                                 std::span<const std::byte> scale,
                                 strata::CudaWeight& weight) {
        strata::CudaWeightDescriptor descriptor;
        descriptor.encoding = strata::CudaWeightEncoding::Nvfp4Group16;
        descriptor.dtype = strata::SafetensorsDtype::U8;
        descriptor.rows = target.module.rows;
        descriptor.columns = target.module.columns;
        descriptor.packed_columns = target.module.columns / 2U;
        descriptor.scale_columns =
            target.module.columns / kContract.nvfp4_group_size;
        descriptor.group_size = kContract.nvfp4_group_size;
        descriptor.global_scale = 1.0F;  // value is irrelevant to transfer cost
        return backend.upload(options.device, descriptor, packed, scale, weight);
    };

    // Arm D: zero-copy source straight out of the mapping, arena allocation.
    // Removes the heap copy and the per-weight cudaMalloc, but the source is
    // still pageable, so the driver must stage it internally.
    reader.release_mapped_views();
    {
        std::vector<strata::CudaWeight> held;
        held.reserve(targets.size());
        const auto started = std::chrono::steady_clock::now();
        for (const auto& target : targets) {
            auto packed = reader.view(target.module.packed->name);
            auto scale = reader.view(target.module.scale->name);
            if (!packed.ok() || !scale.ok()) return 1;
            strata::CudaWeight weight;
            if (auto status = upload_view(target, packed.value, scale.value, weight);
                !status.ok()) {
                for (const auto& error : status.errors) std::cerr << "error: " << error << '\n';
                return 1;
            }
            held.push_back(std::move(weight));
        }
        report("D view()+arena+pageable", seconds_since(started), total_bytes,
               static_cast<std::uint32_t>(targets.size()));
    }

    // Arm E: stage through a page-locked bounce buffer so the host-to-device
    // copy is a real DMA instead of a driver staging copy. Costs one extra
    // host memcpy, which arm C priced at DRAM speed.
    reader.release_mapped_views();
    {
        constexpr std::uint64_t kBounceBytes = 64ULL << 20U;
        std::vector<std::byte> bounce(kBounceBytes);
        if (auto status = backend.register_host_memory(bounce.data(), kBounceBytes);
            !status.ok()) {
            for (const auto& error : status.errors) std::cerr << "error: " << error << '\n';
            return 1;
        }
        std::vector<strata::CudaWeight> held;
        held.reserve(targets.size());
        const auto started = std::chrono::steady_clock::now();
        for (const auto& target : targets) {
            auto packed = reader.view(target.module.packed->name);
            auto scale = reader.view(target.module.scale->name);
            if (!packed.ok() || !scale.ok()) return 1;
            const auto packed_bytes = packed.value.size();
            const auto scale_bytes = scale.value.size();
            if (packed_bytes + scale_bytes > kBounceBytes) return 1;
            std::copy(packed.value.begin(), packed.value.end(), bounce.begin());
            std::copy(scale.value.begin(), scale.value.end(),
                      bounce.begin() + static_cast<std::ptrdiff_t>(packed_bytes));
            strata::CudaWeight weight;
            if (auto status = upload_view(
                    target,
                    std::span<const std::byte>(bounce).subspan(0U, packed_bytes),
                    std::span<const std::byte>(bounce).subspan(packed_bytes,
                                                               scale_bytes),
                    weight);
                !status.ok()) {
                for (const auto& error : status.errors) std::cerr << "error: " << error << '\n';
                return 1;
            }
            held.push_back(std::move(weight));
        }
        report("E view()+arena+pinned bounce", seconds_since(started),
               total_bytes, static_cast<std::uint32_t>(targets.size()));
        backend.unregister_host_memory(bounce.data());
    }

    std::cout << "\nB and C bound the host side; A minus (B or C) is the device"
                 " side.\nIf C is far cheaper than B, the heap copy in read() is"
                 " the removable term.\n";
    return 0;
}
