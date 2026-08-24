// Prices the ordering defect that caps the routed-expert tier.
//
// The tier works and is exact, but `backend.cu` enqueues its kernels *behind*
// the `cudaLaunchHostFunc` callback that runs the host MoE inline, so the two
// are serial by construction. Under the cost model that turns a `max` into a
// sum: at the measured operating point the host share is ~1.04 ms/layer and
// the tier share ~0.33 ms/layer, so the difference between the two orderings
// is the difference between a 1.32x and a 1.13x decode win -- larger than the
// entire mechanism's serial value.
//
// The proposed repair splits the callback in two -- route+selection, then the
// host MoE -- records an event between them, and runs the tier kernels on
// their own stream gated by that event. Whether that actually overlaps is not
// obvious: experiment 0124 found that blocking a driver callback thread
// stalled the whole CUDA context, so the open question is whether a host
// function occupying stream A also blocks kernels on stream B. This probe
// answers exactly that, with a spin kernel calibrated to the tier's real
// duration, in seconds and with no model stage.
//
// It is deliberately not the tier kernel. The question here is scheduling
// semantics, and `strata-dsv4-static-tier-probe` already prices the kernel
// itself (0.143-0.179 ms per expert on the RTX 5060 Ti).

#include "cli_common.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Options {
    int rank_device{0};
    int tier_device{0};
    double host_microseconds{1040.0};
    double tier_microseconds{330.0};
    std::uint32_t iterations{200U};
};

void usage() {
    std::cerr
        << "usage: strata-dsv4-tier-overlap-probe [--rank-device N] "
           "[--tier-device N] [--host-us F] [--tier-us F] [--iterations N]\n";
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--help" || argument == "-h") { usage(); std::exit(0); }
        if (index + 1 >= argc) return false;
        const std::string value(argv[++index]);
        std::uint32_t parsed = 0U;
        if (argument == "--rank-device") {
            if (!strata::cli::parse_u32(value, parsed)) return false;
            options.rank_device = static_cast<int>(parsed);
        } else if (argument == "--tier-device") {
            if (!strata::cli::parse_u32(value, parsed)) return false;
            options.tier_device = static_cast<int>(parsed);
        } else if (argument == "--host-us") {
            options.host_microseconds = std::stod(value);
        } else if (argument == "--tier-us") {
            options.tier_microseconds = std::stod(value);
        } else if (argument == "--iterations") {
            if (!strata::cli::parse_positive_u32(value, options.iterations)) return false;
        } else return false;
    }
    return true;
}

// Occupies the device for a measured wall duration. One block, so it leaves the
// rest of the device free -- the tier's own kernels are equally narrow at batch
// one, and a probe that saturated the card would be answering a different
// question.
__global__ void spin_kernel(std::int64_t cycles) {
    const auto started = clock64();
    while (clock64() - started < cycles) { __threadfence_block(); }
}

struct HostSpin {
    double microseconds{};
};

void CUDART_CB host_spin(void* raw) {
    const auto* spin = static_cast<const HostSpin*>(raw);
    const auto started = std::chrono::steady_clock::now();
    while (std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now() - started).count() <
           spin->microseconds) {
    }
}

bool cuda_ok(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return true;
    std::cerr << "error: " << what << ": " << cudaGetErrorString(status) << "\n";
    return false;
}

double median_of(std::vector<double>& samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) { usage(); return 2; }

    cudaDeviceProp tier_properties{};
    if (!cuda_ok(cudaGetDeviceProperties(&tier_properties, options.tier_device),
                 "query tier device")) {
        return 1;
    }
    // clockRate is in kHz, so cycles = us * kHz / 1000.
    const auto tier_cycles = static_cast<std::int64_t>(
        options.tier_microseconds *
        static_cast<double>(tier_properties.clockRate) / 1000.0);

    cudaStream_t rank_stream{};
    cudaStream_t tier_stream{};
    cudaEvent_t route_done{};
    cudaEvent_t tier_done{};
    if (!cuda_ok(cudaSetDevice(options.rank_device), "select rank device") ||
        !cuda_ok(cudaStreamCreate(&rank_stream), "create rank stream") ||
        !cuda_ok(cudaEventCreateWithFlags(&route_done, cudaEventDisableTiming),
                 "create route event")) {
        return 1;
    }
    if (!cuda_ok(cudaSetDevice(options.tier_device), "select tier device") ||
        !cuda_ok(cudaStreamCreate(&tier_stream), "create tier stream") ||
        !cuda_ok(cudaEventCreateWithFlags(&tier_done, cudaEventDisableTiming),
                 "create tier event")) {
        return 1;
    }

    HostSpin spin{options.host_microseconds};

    // Today's ordering: the tier kernel is enqueued behind the callback on the
    // one stream, so it cannot start until the host MoE inside it has returned.
    const auto serial_arm = [&]() -> bool {
        if (!cuda_ok(cudaSetDevice(options.rank_device), "serial: select device")) return false;
        if (!cuda_ok(cudaLaunchHostFunc(rank_stream, host_spin, &spin),
                     "serial: enqueue host func")) return false;
        spin_kernel<<<1U, 32U, 0U, rank_stream>>>(tier_cycles);
        if (!cuda_ok(cudaGetLastError(), "serial: launch spin")) return false;
        return cuda_ok(cudaStreamSynchronize(rank_stream), "serial: sync");
    };

    // The repair: an event recorded between the route callback and the host-MoE
    // callback releases the tier stream, which then runs concurrently with the
    // host share. The rank stream rejoins it before the partial join.
    const auto overlap_arm = [&]() -> bool {
        if (!cuda_ok(cudaSetDevice(options.rank_device), "overlap: select rank")) return false;
        if (!cuda_ok(cudaEventRecord(route_done, rank_stream),
                     "overlap: record route event")) return false;
        if (!cuda_ok(cudaLaunchHostFunc(rank_stream, host_spin, &spin),
                     "overlap: enqueue host func")) return false;
        if (!cuda_ok(cudaSetDevice(options.tier_device), "overlap: select tier")) return false;
        if (!cuda_ok(cudaStreamWaitEvent(tier_stream, route_done, 0U),
                     "overlap: gate tier stream")) return false;
        spin_kernel<<<1U, 32U, 0U, tier_stream>>>(tier_cycles);
        if (!cuda_ok(cudaGetLastError(), "overlap: launch spin")) return false;
        if (!cuda_ok(cudaEventRecord(tier_done, tier_stream),
                     "overlap: record tier event")) return false;
        if (!cuda_ok(cudaSetDevice(options.rank_device), "overlap: reselect rank")) return false;
        if (!cuda_ok(cudaStreamWaitEvent(rank_stream, tier_done, 0U),
                     "overlap: rejoin tier stream")) return false;
        return cuda_ok(cudaStreamSynchronize(rank_stream), "overlap: sync");
    };

    // Both halves alone, so the two arms above can be read against what the
    // scheduler could in principle deliver rather than only against each other.
    const auto host_only_arm = [&]() -> bool {
        if (!cuda_ok(cudaSetDevice(options.rank_device), "host: select device")) return false;
        if (!cuda_ok(cudaLaunchHostFunc(rank_stream, host_spin, &spin),
                     "host: enqueue host func")) return false;
        return cuda_ok(cudaStreamSynchronize(rank_stream), "host: sync");
    };
    const auto tier_only_arm = [&]() -> bool {
        if (!cuda_ok(cudaSetDevice(options.tier_device), "tier: select device")) return false;
        spin_kernel<<<1U, 32U, 0U, tier_stream>>>(tier_cycles);
        if (!cuda_ok(cudaGetLastError(), "tier: launch spin")) return false;
        return cuda_ok(cudaStreamSynchronize(tier_stream), "tier: sync");
    };

    struct Arm {
        const char* name;
        bool (*unused)();
    };
    const auto run = [&](const char* name, auto&& arm) -> double {
        for (std::uint32_t warm = 0U; warm < 10U; ++warm) {
            if (!arm()) std::exit(1);
        }
        std::vector<double> samples;
        samples.reserve(options.iterations);
        for (std::uint32_t index = 0U; index < options.iterations; ++index) {
            const auto started = std::chrono::steady_clock::now();
            if (!arm()) std::exit(1);
            samples.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count());
        }
        const auto median = median_of(samples);
        std::printf("  %-14s median %.4f ms\n", name, median);
        return median;
    };

    std::printf(
        "rank device %d | tier device %d | host %.0f us | tier %.0f us | %u "
        "iterations\n",
        options.rank_device, options.tier_device, options.host_microseconds,
        options.tier_microseconds, options.iterations);
    const auto host_only = run("host only", host_only_arm);
    const auto tier_only = run("tier only", tier_only_arm);
    const auto serial = run("serial", serial_arm);
    const auto overlap = run("overlap", overlap_arm);

    const auto sum = host_only + tier_only;
    const auto maximum = std::max(host_only, tier_only);
    std::printf("\n  sum %.4f ms | max %.4f ms\n", sum, maximum);
    std::printf(
        "  serial  is %.2fx of max (%.4f ms over)\n",
        serial / maximum, serial - maximum);
    std::printf(
        "  overlap is %.2fx of max (%.4f ms over)\n",
        overlap / maximum, overlap - maximum);
    std::printf(
        "  43 layers/token -> serial %.2f ms/token, overlap %.2f ms/token, "
        "saving %.2f ms/token\n",
        serial * 43.0, overlap * 43.0, (serial - overlap) * 43.0);
    return 0;
}
