// F8-1 phase-B cheapest falsifier: launch/wave/read-only upper bounds for the
// protected 2.10 MiB wkv and 4.19 MiB wq_a FP8 matrices, plus their valid
// same-input fused geometry. No decode, MMA, activation, output, or reduction
// work is present; passing authorizes a real kernel but proves no F8-2 gate.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kMiB = 1ULL << 20U;
constexpr std::size_t kScrubBytes = 256ULL * kMiB;
constexpr std::size_t kRulerBytes = 128ULL * kMiB;
constexpr std::size_t kArenaBudget = 96ULL * kMiB;
constexpr std::size_t kMemoryCeiling = 512ULL * kMiB;
constexpr std::uint32_t kWarmups = 3U;
constexpr std::uint32_t kSamples = 11U;
constexpr double kRequiredEfficiency = 0.82;

constexpr std::size_t fp8_bytes(std::uint32_t n, std::uint32_t k) {
    return static_cast<std::size_t>(n) * k +
           static_cast<std::size_t>(n / 128U) * (k / 128U);
}

constexpr std::size_t kWkvBytes = fp8_bytes(512U, 4096U);
constexpr std::size_t kWqaBytes = fp8_bytes(1024U, 4096U);
constexpr std::size_t kFusedBytes = kWkvBytes + kWqaBytes;

void check(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        check(cudaMalloc(&value_, bytes), "allocate FP8 launch-ceiling buffer");
    }
    ~DeviceBuffer() {
        if (value_ != nullptr) static_cast<void>(cudaFree(value_));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    [[nodiscard]] void* get() const noexcept { return value_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

  private:
    void* value_{nullptr};
    std::size_t bytes_{0U};
};

__global__ void scrub_kernel(std::uint8_t* data, std::size_t bytes) {
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) *
                                 blockDim.x + threadIdx.x;
         index < bytes; index += stride) {
        data[index] = static_cast<std::uint8_t>(index);
    }
}

__global__ void read_ilp4_kernel(const uint4* data, std::uint64_t vectors,
                                 unsigned int* sink) {
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    unsigned int a = 0U, b = 0U, c = 0U, d = 0U;
    for (; index + 3U * stride < vectors; index += 4U * stride) {
        const uint4 va = data[index];
        const uint4 vb = data[index + stride];
        const uint4 vc = data[index + 2U * stride];
        const uint4 vd = data[index + 3U * stride];
        a ^= va.x ^ va.y ^ va.z ^ va.w;
        b ^= vb.x ^ vb.y ^ vb.z ^ vb.w;
        c ^= vc.x ^ vc.y ^ vc.z ^ vc.w;
        d ^= vd.x ^ vd.y ^ vd.z ^ vd.w;
    }
    for (; index < vectors; index += stride) {
        const uint4 value = data[index];
        a ^= value.x ^ value.y ^ value.z ^ value.w;
    }
    if ((a ^ b ^ c ^ d) == 0xDEAD'BEEFU) sink[threadIdx.x] = a;
}

__global__ void empty_kernel(unsigned int* sink) {
    if (blockIdx.x == 0U && threadIdx.x == 0U && *sink == 0xDEAD'BEEFU) {
        *sink = 0U;
    }
}

float median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}

template <typename Function>
float time_once(cudaEvent_t begin, cudaEvent_t end, cudaStream_t stream,
                Function&& function) {
    check(cudaEventRecord(begin, stream), "record ceiling start");
    function();
    check(cudaEventRecord(end, stream), "record ceiling end");
    check(cudaEventSynchronize(end), "finish ceiling timing");
    float milliseconds = 0.0F;
    check(cudaEventElapsedTime(&milliseconds, begin, end),
          "read ceiling timing");
    return milliseconds * 1000.0F;
}

struct GeometryResult {
    std::uint32_t threads{0U};
    std::uint32_t blocks{0U};
    double waves{0.0};
    float empty_us{0.0F};
    float wkv_us{0.0F};
    float wqa_us{0.0F};
    float fused_us{0.0F};
    double wkv_efficiency{0.0};
    double wqa_efficiency{0.0};
    double fused_efficiency{0.0};
};

double gbps(std::size_t bytes, float microseconds) {
    return static_cast<double>(bytes) / static_cast<double>(microseconds) /
           1000.0;
}

void print_geometry(std::ostream& out, const GeometryResult& value, bool last) {
    out << "    {\"threads\": " << value.threads
        << ", \"blocks\": " << value.blocks << ", \"waves\": "
        << value.waves << ", \"empty_us\": " << value.empty_us
        << ", \"wkv_us\": " << value.wkv_us
        << ", \"wqa_us\": " << value.wqa_us
        << ", \"fused_us\": " << value.fused_us
        << ", \"wkv_efficiency\": " << value.wkv_efficiency
        << ", \"wqa_efficiency\": " << value.wqa_efficiency
        << ", \"fused_efficiency\": " << value.fused_efficiency << "}"
        << (last ? "\n" : ",\n");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        int device_index = 0;
        std::string output_path;
        for (int i = 1; i < argc; ++i) {
            const std::string_view flag = argv[i];
            if (flag == "--device" && i + 1 < argc) {
                device_index = std::stoi(argv[++i]);
            } else if (flag == "--output" && i + 1 < argc) {
                output_path = argv[++i];
            } else {
                std::cerr << "usage: " << argv[0]
                          << " [--device INDEX] [--output PATH]\n";
                return EXIT_FAILURE;
            }
        }

        check(cudaSetDevice(device_index), "select ceiling device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device_index),
              "query ceiling device");
        if (properties.major != 8 || properties.minor != 6) {
            throw std::runtime_error("FP8 launch ceiling requires SM86");
        }

        const std::uint32_t replicas =
            static_cast<std::uint32_t>(kArenaBudget / kFusedBytes);
        const std::size_t arena_bytes =
            static_cast<std::size_t>(replicas) * kFusedBytes;
        DeviceBuffer arena(arena_bytes);
        DeviceBuffer scrub(kScrubBytes);
        DeviceBuffer ruler(kRulerBytes);
        DeviceBuffer sink(512U * sizeof(unsigned int));
        const std::size_t peak_bytes =
            arena.bytes() + scrub.bytes() + ruler.bytes() + sink.bytes();
        if (peak_bytes > kMemoryCeiling) {
            throw std::runtime_error("launch ceiling exceeds 512 MiB");
        }

        cudaStream_t stream = nullptr;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "create ceiling stream");
        check(cudaMemsetAsync(arena.get(), 0xA5, arena.bytes(), stream),
              "initialize ceiling arena");
        check(cudaMemsetAsync(ruler.get(), 0x5A, ruler.bytes(), stream),
              "initialize ceiling ruler");
        check(cudaMemsetAsync(sink.get(), 0, sink.bytes(), stream),
              "initialize ceiling sink");
        check(cudaStreamSynchronize(stream), "finish ceiling initialization");

        cudaEvent_t begin = nullptr, end = nullptr;
        check(cudaEventCreate(&begin), "create ceiling start event");
        check(cudaEventCreate(&end), "create ceiling end event");
        const auto scrub_l2 = [&] {
            scrub_kernel<<<static_cast<unsigned int>(kScrubBytes / 4096U),
                           256U, 0U, stream>>>(
                static_cast<std::uint8_t*>(scrub.get()), scrub.bytes());
        };
        const auto read = [&](const void* pointer, std::size_t bytes,
                              std::uint32_t blocks,
                              std::uint32_t threads) {
            read_ilp4_kernel<<<blocks, threads, 0U, stream>>>(
                static_cast<const uint4*>(pointer), bytes / sizeof(uint4),
                static_cast<unsigned int*>(sink.get()));
        };

        const std::uint32_t ruler_blocks =
            static_cast<std::uint32_t>(properties.multiProcessorCount) * 8U;
        for (std::uint32_t i = 0U; i < kWarmups; ++i) {
            read(ruler.get(), ruler.bytes(), ruler_blocks, 256U);
        }
        check(cudaStreamSynchronize(stream), "finish ruler warmup");
        std::vector<float> ruler_samples;
        for (std::uint32_t i = 0U; i < kSamples; ++i) {
            scrub_l2();
            ruler_samples.push_back(time_once(begin, end, stream, [&] {
                read(ruler.get(), ruler.bytes(), ruler_blocks, 256U);
            }));
        }
        const float ruler_us = median(ruler_samples);
        const double ruler_gbps = gbps(kRulerBytes, ruler_us);

        const std::uint32_t block_multipliers[] = {1U, 2U, 4U, 8U};
        const std::uint32_t thread_counts[] = {128U, 256U};
        std::vector<GeometryResult> results;
        for (const std::uint32_t threads : thread_counts) {
            for (const std::uint32_t multiplier : block_multipliers) {
                const std::uint32_t blocks =
                    static_cast<std::uint32_t>(properties.multiProcessorCount) *
                    multiplier;
                for (std::uint32_t i = 0U; i < kWarmups; ++i) {
                    read(arena.get(), kWkvBytes, blocks, threads);
                    read(arena.get(), kWqaBytes, blocks, threads);
                    read(arena.get(), kFusedBytes, blocks, threads);
                }
                check(cudaStreamSynchronize(stream), "finish geometry warmup");
                std::vector<float> empty_samples, wkv_samples, wqa_samples,
                    fused_samples;
                for (std::uint32_t i = 0U; i < kSamples; ++i) {
                    const std::uint32_t replica = i % replicas;
                    auto* base = static_cast<std::uint8_t*>(arena.get()) +
                                 static_cast<std::size_t>(replica) * kFusedBytes;
                    scrub_l2();
                    empty_samples.push_back(time_once(begin, end, stream, [&] {
                        empty_kernel<<<blocks, threads, 0U, stream>>>(
                            static_cast<unsigned int*>(sink.get()));
                    }));
                    scrub_l2();
                    wkv_samples.push_back(time_once(begin, end, stream, [&] {
                        read(base, kWkvBytes, blocks, threads);
                    }));
                    scrub_l2();
                    wqa_samples.push_back(time_once(begin, end, stream, [&] {
                        read(base + kWkvBytes, kWqaBytes, blocks, threads);
                    }));
                    scrub_l2();
                    fused_samples.push_back(time_once(begin, end, stream, [&] {
                        read(base, kFusedBytes, blocks, threads);
                    }));
                }
                GeometryResult result{};
                result.threads = threads;
                result.blocks = blocks;
                result.waves = static_cast<double>(blocks) /
                               properties.multiProcessorCount;
                result.empty_us = median(empty_samples);
                result.wkv_us = median(wkv_samples);
                result.wqa_us = median(wqa_samples);
                result.fused_us = median(fused_samples);
                result.wkv_efficiency = gbps(kWkvBytes, result.wkv_us) /
                                        ruler_gbps;
                result.wqa_efficiency = gbps(kWqaBytes, result.wqa_us) /
                                        ruler_gbps;
                result.fused_efficiency = gbps(kFusedBytes, result.fused_us) /
                                          ruler_gbps;
                results.push_back(result);
            }
        }

        const auto best_wkv = std::max_element(
            results.begin(), results.end(), [](const auto& a, const auto& b) {
                return a.wkv_efficiency < b.wkv_efficiency;
            });
        const auto best_wqa = std::max_element(
            results.begin(), results.end(), [](const auto& a, const auto& b) {
                return a.wqa_efficiency < b.wqa_efficiency;
            });
        const auto best_fused = std::max_element(
            results.begin(), results.end(), [](const auto& a, const auto& b) {
                return a.fused_efficiency < b.fused_efficiency;
            });

        std::ofstream file;
        std::ostream* out = &std::cout;
        if (!output_path.empty()) {
            file.open(output_path);
            if (!file) throw std::runtime_error("cannot open output path");
            out = &file;
        }
        *out << std::fixed << std::setprecision(9)
             << "{\n  \"milestone\": \"F8-1 phase-B launch ceiling\",\n"
             << "  \"device_index\": " << device_index << ",\n"
             << "  \"device_name\": \"" << properties.name << "\",\n"
             << "  \"device_capability\": \"" << properties.major << "."
             << properties.minor << "\",\n"
             << "  \"sm_count\": " << properties.multiProcessorCount << ",\n"
             << "  \"replicas\": " << replicas << ",\n"
             << "  \"peak_device_bytes\": " << peak_bytes << ",\n"
             << "  \"ruler_bytes\": " << kRulerBytes << ",\n"
             << "  \"ruler_us\": " << ruler_us << ",\n"
             << "  \"ruler_gbps\": " << ruler_gbps << ",\n"
             << "  \"required_efficiency\": " << kRequiredEfficiency
             << ",\n"
             << "  \"wkv_bytes\": " << kWkvBytes << ",\n"
             << "  \"wqa_bytes\": " << kWqaBytes << ",\n"
             << "  \"fused_bytes\": " << kFusedBytes << ",\n"
             << "  \"wkv_budget_us\": "
             << static_cast<double>(kWkvBytes) / ruler_gbps / 1000.0 /
                    kRequiredEfficiency
             << ",\n"
             << "  \"wqa_budget_us\": "
             << static_cast<double>(kWqaBytes) / ruler_gbps / 1000.0 /
                    kRequiredEfficiency
             << ",\n"
             << "  \"fused_budget_us\": "
             << static_cast<double>(kFusedBytes) / ruler_gbps / 1000.0 /
                    kRequiredEfficiency
             << ",\n"
             << "  \"best_wkv_efficiency\": " << best_wkv->wkv_efficiency
             << ",\n"
             << "  \"best_wqa_efficiency\": " << best_wqa->wqa_efficiency
             << ",\n"
             << "  \"best_fused_efficiency\": "
             << best_fused->fused_efficiency << ",\n"
             << "  \"standalone_upper_bound_pass\": "
             << (best_wkv->wkv_efficiency >= kRequiredEfficiency &&
                         best_wqa->wqa_efficiency >= kRequiredEfficiency
                     ? "true"
                     : "false")
             << ",\n"
             << "  \"fused_upper_bound_pass\": "
             << (best_fused->fused_efficiency >= kRequiredEfficiency ? "true"
                                                                       : "false")
             << ",\n  \"geometries\": [\n";
        for (std::size_t i = 0U; i < results.size(); ++i) {
            print_geometry(*out, results[i], i + 1U == results.size());
        }
        *out << "  ]\n}\n";

        check(cudaEventDestroy(begin), "destroy ceiling start event");
        check(cudaEventDestroy(end), "destroy ceiling end event");
        check(cudaStreamDestroy(stream), "destroy ceiling stream");
        return best_fused->fused_efficiency >= kRequiredEfficiency
                   ? EXIT_SUCCESS
                   : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
