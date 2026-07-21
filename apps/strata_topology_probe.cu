#include "cli_common.hpp"

#include "strata/cuda_backend.hpp"
#include "strata/model.hpp"

#include <cuda_runtime.h>
#include <numa.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::vector<int> devices{0, 1, 2};
    std::vector<int> numa_nodes{0, 1};
    std::vector<unsigned int> queue_depths{1, 4, 8};
    std::string checkpoint_file;
    unsigned int repetitions{3};
    std::size_t transfer_bytes{64U * 1024U * 1024U};
    std::size_t activation_bytes{8U * 1024U * 1024U};
    std::size_t io_bytes{256U * 1024U * 1024U};
    std::size_t io_block_bytes{4U * 1024U * 1024U};
    std::uint32_t prefill_rows{30};
};

struct TimedSample {
    double seconds{};
    std::uint64_t bytes{};
    bool verified{};
    std::uint64_t checksum{};
    std::uint64_t physical_read_bytes{};
};

std::map<int, std::uint64_t> observed_vram_used;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

void cuda_check(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        fail(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void observe_vram(int device) {
    cuda_check(cudaSetDevice(device), "select CUDA device for VRAM observation");
    std::size_t free_bytes = 0U;
    std::size_t total_bytes = 0U;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "observe CUDA memory use");
    observed_vram_used[device] = std::max(observed_vram_used[device],
                                          static_cast<std::uint64_t>(total_bytes - free_bytes));
}

std::string read_text(const std::string& path) {
    std::ifstream input(path);
    std::string value;
    std::getline(input, value);
    return value;
}

std::vector<int> parse_int_list(std::string_view text) {
    std::vector<int> values;
    std::size_t begin = 0;
    while (begin < text.size()) {
        const auto end = text.find(',', begin);
        const auto token = text.substr(begin, end == std::string_view::npos
                                                  ? text.size() - begin
                                                  : end - begin);
        if (token.empty()) fail("empty integer-list item");
        values.push_back(std::stoi(std::string(token)));
        if (end == std::string_view::npos) break;
        begin = end + 1U;
    }
    if (values.empty()) fail("integer list cannot be empty");
    return values;
}

std::vector<unsigned int> parse_uint_list(std::string_view text) {
    std::vector<unsigned int> values;
    for (const int value : parse_int_list(text)) {
        if (value <= 0) fail("queue depths must be positive");
        values.push_back(static_cast<unsigned int>(value));
    }
    return values;
}

std::size_t parse_size(std::string value) {
    std::uint64_t multiplier = 1U;
    if (!value.empty()) {
        const char suffix = value.back();
        if (suffix == 'K' || suffix == 'k') {
            multiplier = 1024U;
            value.pop_back();
        } else if (suffix == 'M' || suffix == 'm') {
            multiplier = 1024U * 1024U;
            value.pop_back();
        } else if (suffix == 'G' || suffix == 'g') {
            multiplier = 1024U * 1024U * 1024U;
            value.pop_back();
        }
    }
    const auto base = std::stoull(value);
    if (base > std::numeric_limits<std::size_t>::max() / multiplier) {
        fail("byte size is too large");
    }
    return static_cast<std::size_t>(base * multiplier);
}

void print_help() {
    std::cout
        << "usage: strata-topology-probe --checkpoint-file FILE [options]\n"
        << "  --devices 0,1,2          CUDA indexes in configured order\n"
        << "  --numa-nodes 0,1        first-touch source nodes\n"
        << "  --repetitions 3         repetitions per measured path (minimum 3)\n"
        << "  --transfer-bytes 64M    H2D/D2H test extent\n"
        << "  --activation-bytes 8M   inter-GPU activation extent\n"
        << "  --io-bytes 256M         bytes per checkpoint read repetition\n"
        << "  --io-block-bytes 4M     aligned checkpoint range size\n"
        << "  --queue-depths 1,4,8    concurrent synchronous read workers\n"
        << "  --prefill-rows 30       representative prefill row count\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            print_help();
            std::exit(0);
        }
        if (index + 1 >= argc) fail("missing value for " + argument);
        const std::string value = argv[++index];
        if (argument == "--checkpoint-file") options.checkpoint_file = value;
        else if (argument == "--devices") options.devices = parse_int_list(value);
        else if (argument == "--numa-nodes") options.numa_nodes = parse_int_list(value);
        else if (argument == "--queue-depths") options.queue_depths = parse_uint_list(value);
        else if (argument == "--repetitions") options.repetitions = std::stoul(value);
        else if (argument == "--transfer-bytes") options.transfer_bytes = parse_size(value);
        else if (argument == "--activation-bytes") options.activation_bytes = parse_size(value);
        else if (argument == "--io-bytes") options.io_bytes = parse_size(value);
        else if (argument == "--io-block-bytes") options.io_block_bytes = parse_size(value);
        else if (argument == "--prefill-rows") options.prefill_rows = std::stoul(value);
        else fail("unknown argument: " + argument);
    }
    if (options.checkpoint_file.empty()) fail("--checkpoint-file is required");
    if (options.repetitions < 3U) fail("T1 requires at least three repetitions");
    if (options.transfer_bytes == 0U || options.activation_bytes == 0U ||
        options.io_bytes == 0U || options.io_block_bytes == 0U ||
        options.prefill_rows == 0U) {
        fail("byte extents and prefill rows must be positive");
    }
    if (options.io_block_bytes % 4096U != 0U ||
        options.io_bytes % options.io_block_bytes != 0U) {
        fail("I/O sizes must be 4096-aligned and io-bytes must divide into blocks");
    }
    return options;
}

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2U;
    return values.size() % 2U == 1U ? values[middle]
                                    : (values[middle - 1U] + values[middle]) / 2.0;
}

std::uint64_t checksum(std::span<const std::byte> bytes) {
    std::uint64_t value = 1469598103934665603ULL;
    for (const auto byte : bytes) {
        value ^= std::to_integer<unsigned char>(byte);
        value *= 1099511628211ULL;
    }
    return value;
}

void fill_pattern(std::span<std::byte> bytes, unsigned int seed) {
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::byte>((index * 131U + seed * 17U) & 0xffU);
    }
}

class NumaBuffer {
public:
    NumaBuffer(std::size_t bytes, int node) : bytes_(bytes), node_(node) {
        data_ = numa_alloc_onnode(bytes_, node_);
        if (data_ == nullptr) fail("numa_alloc_onnode failed");
        std::memset(data_, 0, bytes_);
        verify_placement();
    }
    ~NumaBuffer() {
        if (registered_) static_cast<void>(cudaHostUnregister(data_));
        if (data_ != nullptr) numa_free(data_, bytes_);
    }
    NumaBuffer(const NumaBuffer&) = delete;
    NumaBuffer& operator=(const NumaBuffer&) = delete;
    void pin() {
        if (!registered_) {
            cuda_check(cudaHostRegister(data_, bytes_, cudaHostRegisterPortable),
                       "register NUMA host buffer");
            registered_ = true;
        }
    }
    std::span<std::byte> span() {
        return {static_cast<std::byte*>(data_), bytes_};
    }
    void* data() noexcept { return data_; }

private:
    void verify_placement() {
        constexpr std::size_t maximum_samples = 64U;
        const long page_size = sysconf(_SC_PAGESIZE);
        if (page_size <= 0) fail("query page size for NUMA verification");
        const auto pages = (bytes_ + static_cast<std::size_t>(page_size) - 1U) /
                           static_cast<std::size_t>(page_size);
        const auto stride = std::max<std::size_t>(1U, pages / maximum_samples);
        std::vector<void*> addresses;
        for (std::size_t page = 0; page < pages && addresses.size() < maximum_samples;
             page += stride) {
            addresses.push_back(static_cast<std::byte*>(data_) +
                                static_cast<std::ptrdiff_t>(page *
                                                            static_cast<std::size_t>(page_size)));
        }
        std::vector<int> status(addresses.size(), -1);
        if (numa_move_pages(0, static_cast<unsigned long>(addresses.size()),
                            addresses.data(), nullptr, status.data(), 0) != 0) {
            fail("query NUMA page placement: " + std::string(std::strerror(errno)));
        }
        if (!std::all_of(status.begin(), status.end(),
                         [this](int placed) { return placed == node_; })) {
            fail("NUMA allocation was not first-touched on the requested node");
        }
    }

    void* data_{};
    std::size_t bytes_{};
    int node_{};
    bool registered_{};
};

struct DeviceIdentity {
    int index{};
    std::string name;
    std::string pci_bdf;
    std::string uuid;
    int compute_major{};
    int compute_minor{};
    std::uint64_t free_bytes{};
    std::uint64_t total_bytes{};
    int numa_node{-1};
    std::string link_speed;
    std::string link_width;
};

std::string device_uuid(const cudaDeviceProp& properties) {
    std::ostringstream output;
    for (std::size_t index = 0; index < sizeof(properties.uuid.bytes); ++index) {
        if (index == 4U || index == 6U || index == 8U || index == 10U) output << '-';
        output << std::hex << std::setw(2) << std::setfill('0')
               << static_cast<unsigned int>(
                      static_cast<unsigned char>(properties.uuid.bytes[index]));
    }
    return output.str();
}

DeviceIdentity inspect_device(int device) {
    cuda_check(cudaSetDevice(device), "select CUDA device for identity");
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device), "query CUDA device properties");
    char pci[32]{};
    cuda_check(cudaDeviceGetPCIBusId(pci, sizeof(pci), device), "query PCI bus id");
    std::size_t free_bytes = 0U;
    std::size_t total_bytes = 0U;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "query device memory");
    const std::string sysfs = "/sys/bus/pci/devices/" + std::string(pci);
    DeviceIdentity identity;
    identity.index = device;
    identity.name = properties.name;
    identity.pci_bdf = pci;
    identity.uuid = device_uuid(properties);
    identity.compute_major = properties.major;
    identity.compute_minor = properties.minor;
    identity.free_bytes = free_bytes;
    identity.total_bytes = total_bytes;
    const auto numa_text = read_text(sysfs + "/numa_node");
    if (!numa_text.empty()) identity.numa_node = std::stoi(numa_text);
    identity.link_speed = read_text(sysfs + "/current_link_speed");
    identity.link_width = read_text(sysfs + "/current_link_width");
    return identity;
}

std::vector<TimedSample> transfer_samples(int device, int numa_node,
                                          std::size_t bytes, unsigned int repetitions,
                                          bool pinned, bool h2d,
                                          double& registration_seconds) {
    cuda_check(cudaSetDevice(device), "select CUDA device for transfer");
    NumaBuffer source(bytes, numa_node);
    NumaBuffer destination(bytes, numa_node);
    fill_pattern(source.span(), static_cast<unsigned int>(device * 7 + numa_node));
    if (pinned) {
        const auto started = Clock::now();
        source.pin();
        destination.pin();
        registration_seconds = seconds_since(started);
    }
    void* device_buffer = nullptr;
    cuda_check(cudaMalloc(&device_buffer, bytes), "allocate transfer device buffer");
    observe_vram(device);
    cuda_check(cudaMemcpy(device_buffer, source.data(), bytes, cudaMemcpyHostToDevice),
               "initialize transfer device buffer");
    cudaEvent_t begin{};
    cudaEvent_t end{};
    cuda_check(cudaEventCreate(&begin), "create transfer begin event");
    cuda_check(cudaEventCreate(&end), "create transfer end event");
    std::vector<TimedSample> samples;
    for (unsigned int repetition = 0; repetition < repetitions; ++repetition) {
        if (!h2d) std::memset(destination.data(), 0, bytes);
        double seconds = 0.0;
        if (pinned) {
            cuda_check(cudaEventRecord(begin), "record transfer begin");
            cuda_check(cudaMemcpyAsync(h2d ? device_buffer : destination.data(),
                                       h2d ? source.data() : device_buffer, bytes,
                                       h2d ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost),
                       "enqueue pinned transfer");
            cuda_check(cudaEventRecord(end), "record transfer end");
            cuda_check(cudaEventSynchronize(end), "synchronize pinned transfer");
            float milliseconds = 0.0F;
            cuda_check(cudaEventElapsedTime(&milliseconds, begin, end),
                       "measure pinned transfer");
            seconds = static_cast<double>(milliseconds) / 1000.0;
        } else {
            const auto started = Clock::now();
            cuda_check(cudaMemcpy(h2d ? device_buffer : destination.data(),
                                  h2d ? source.data() : device_buffer, bytes,
                                  h2d ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost),
                       "execute pageable transfer");
            seconds = seconds_since(started);
        }
        if (h2d) {
            cuda_check(cudaMemcpy(destination.data(), device_buffer, bytes,
                                  cudaMemcpyDeviceToHost),
                       "validate H2D transfer");
        }
        const bool verified = std::memcmp(source.data(), destination.data(), bytes) == 0;
        samples.push_back({seconds, bytes, verified, checksum(destination.span()), 0U});
    }
    static_cast<void>(cudaEventDestroy(begin));
    static_cast<void>(cudaEventDestroy(end));
    static_cast<void>(cudaFree(device_buffer));
    return samples;
}

std::vector<double> allocation_samples(int device, std::size_t bytes,
                                       unsigned int repetitions) {
    cuda_check(cudaSetDevice(device), "select CUDA device for allocation");
    std::vector<double> samples;
    for (unsigned int repetition = 0; repetition < repetitions; ++repetition) {
        void* pointer = nullptr;
        const auto started = Clock::now();
        cuda_check(cudaMalloc(&pointer, bytes), "benchmark cudaMalloc");
        observe_vram(device);
        cuda_check(cudaFree(pointer), "benchmark cudaFree");
        samples.push_back(seconds_since(started));
    }
    return samples;
}

std::vector<double> synchronization_samples(int device, unsigned int repetitions) {
    cuda_check(cudaSetDevice(device), "select CUDA device for synchronization");
    cudaStream_t stream{};
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create synchronization stream");
    std::vector<double> samples;
    for (unsigned int repetition = 0; repetition < repetitions; ++repetition) {
        const auto started = Clock::now();
        cuda_check(cudaStreamSynchronize(stream), "benchmark empty synchronization");
        samples.push_back(seconds_since(started));
    }
    static_cast<void>(cudaStreamDestroy(stream));
    return samples;
}

struct PeerResult {
    int source{};
    int destination{};
    bool peer_capable{};
    std::string path;
    std::vector<TimedSample> samples;
};

PeerResult peer_samples(int source_device, int destination_device, std::size_t bytes,
                        unsigned int repetitions) {
    PeerResult result;
    result.source = source_device;
    result.destination = destination_device;
    int capable = 0;
    cuda_check(cudaDeviceCanAccessPeer(&capable, source_device, destination_device),
               "query CUDA peer capability");
    result.peer_capable = capable != 0;
    result.path = result.peer_capable ? "cuda_peer" : "pinned_host_staging";
    void* source = nullptr;
    void* destination = nullptr;
    cuda_check(cudaSetDevice(source_device), "select peer source device");
    cuda_check(cudaMalloc(&source, bytes), "allocate peer source");
    observe_vram(source_device);
    if (result.peer_capable) {
        const auto status = cudaDeviceEnablePeerAccess(destination_device, 0);
        if (status != cudaSuccess && status != cudaErrorPeerAccessAlreadyEnabled) {
            cuda_check(status, "enable peer access from source");
        }
        static_cast<void>(cudaGetLastError());
    }
    cuda_check(cudaSetDevice(destination_device), "select peer destination device");
    cuda_check(cudaMalloc(&destination, bytes), "allocate peer destination");
    observe_vram(destination_device);
    if (result.peer_capable) {
        const auto status = cudaDeviceEnablePeerAccess(source_device, 0);
        if (status != cudaSuccess && status != cudaErrorPeerAccessAlreadyEnabled) {
            cuda_check(status, "enable peer access from destination");
        }
        static_cast<void>(cudaGetLastError());
    }
    void* staging = nullptr;
    void* validation = nullptr;
    cuda_check(cudaHostAlloc(&staging, bytes, cudaHostAllocPortable),
               "allocate peer staging buffer");
    cuda_check(cudaHostAlloc(&validation, bytes, cudaHostAllocPortable),
               "allocate peer validation buffer");
    fill_pattern({static_cast<std::byte*>(staging), bytes},
                 static_cast<unsigned int>(source_device * 11 + destination_device));
    cuda_check(cudaSetDevice(source_device), "select peer source for initialization");
    cuda_check(cudaMemcpy(source, staging, bytes, cudaMemcpyHostToDevice),
               "initialize peer source");
    for (unsigned int repetition = 0; repetition < repetitions; ++repetition) {
        const auto started = Clock::now();
        if (result.peer_capable) {
            cuda_check(cudaMemcpyPeer(destination, destination_device, source,
                                      source_device, bytes),
                       "execute peer transfer");
        } else {
            cuda_check(cudaSetDevice(source_device), "select staged source");
            cuda_check(cudaMemcpy(staging, source, bytes, cudaMemcpyDeviceToHost),
                       "stage activation to host");
            cuda_check(cudaSetDevice(destination_device), "select staged destination");
            cuda_check(cudaMemcpy(destination, staging, bytes, cudaMemcpyHostToDevice),
                       "stage activation to destination");
        }
        const double seconds = seconds_since(started);
        cuda_check(cudaSetDevice(destination_device), "select peer destination for validation");
        cuda_check(cudaMemcpy(validation, destination, bytes, cudaMemcpyDeviceToHost),
                   "validate peer transfer");
        const bool verified = std::memcmp(staging, validation, bytes) == 0;
        result.samples.push_back(
            {seconds, bytes, verified,
             checksum({static_cast<const std::byte*>(validation), bytes}), 0U});
    }
    static_cast<void>(cudaFree(destination));
    cuda_check(cudaSetDevice(source_device), "select peer source for cleanup");
    static_cast<void>(cudaFree(source));
    static_cast<void>(cudaFreeHost(staging));
    static_cast<void>(cudaFreeHost(validation));
    return result;
}

struct KernelResult {
    int device{};
    std::string encoding;
    std::uint32_t rows{};
    std::uint64_t output_rows{};
    std::uint64_t columns{};
    std::uint64_t weight_bytes{};
    std::vector<double> kernel_seconds;
    std::vector<double> service_seconds;
    bool verified{};
};

std::array<std::byte, 2> bf16_one() {
    return {std::byte{0x80}, std::byte{0x3f}};
}

void fill_quantized(NumaBuffer& weights, NumaBuffer& scales, std::uint32_t bits) {
    const std::uint32_t word = bits == 4U ? 0x9999'9999U : 0x8181'8181U;
    for (std::size_t offset = 0; offset < weights.span().size(); offset += 4U) {
        std::memcpy(weights.span().data() + static_cast<std::ptrdiff_t>(offset),
                    &word, sizeof(word));
    }
    const auto one = bf16_one();
    for (std::size_t offset = 0; offset < scales.span().size(); offset += 2U) {
        std::copy(one.begin(), one.end(),
                  scales.span().begin() + static_cast<std::ptrdiff_t>(offset));
    }
}

KernelResult kernel_samples(int device, int numa_node, std::uint32_t bits,
                            std::uint32_t batch_rows, unsigned int repetitions) {
    constexpr std::uint64_t hidden = 6144U;
    const std::uint64_t output_rows = bits == 4U ? 2048U : 6144U;
    const std::uint64_t lanes = 32U / bits;
    const std::uint64_t packed_columns = (hidden + lanes - 1U) / lanes;
    const std::uint64_t scale_columns = (hidden + 127U) / 128U;
    NumaBuffer weights(output_rows * packed_columns * 4U, numa_node);
    NumaBuffer scales(output_rows * scale_columns * 2U, numa_node);
    fill_quantized(weights, scales, bits);
    weights.pin();
    scales.pin();

    strata::CudaWeightDescriptor descriptor;
    descriptor.encoding = bits == 4U ? strata::CudaWeightEncoding::OffsetPackedInt4
                                     : strata::CudaWeightEncoding::OffsetPackedInt8;
    descriptor.dtype = strata::SafetensorsDtype::I32;
    descriptor.rows = output_rows;
    descriptor.columns = hidden;
    descriptor.packed_columns = packed_columns;
    descriptor.scale_columns = scale_columns;
    descriptor.group_size = 128U;
    strata::CudaBackend backend;
    const std::array<int, 1> selected{device};
    const auto initialized = backend.initialize(selected, true);
    if (!initialized.ok()) fail(initialized.errors.front());
    strata::CudaWeight weight;
    const auto uploaded = backend.upload(device, descriptor, weights.span(), scales.span(), weight);
    if (!uploaded.ok()) fail(uploaded.errors.front());
    observe_vram(device);
    std::vector<float> input(static_cast<std::size_t>(hidden) * batch_rows, 1.0F);
    std::vector<float> output(static_cast<std::size_t>(output_rows) * batch_rows);
    auto warm = backend.matmul(weight, input, batch_rows, output);
    if (!warm.ok()) fail(warm.errors.front());
    observe_vram(device);

    KernelResult result;
    result.device = device;
    result.encoding = bits == 4U ? "int4_group128" : "int8_group128";
    result.rows = batch_rows;
    result.output_rows = output_rows;
    result.columns = hidden;
    result.weight_bytes = weight.device_bytes();
    result.verified = true;
    for (unsigned int repetition = 0; repetition < repetitions; ++repetition) {
        const auto before = backend.stats();
        const auto started = Clock::now();
        const auto status = backend.matmul(weight, input, batch_rows, output);
        const double service = seconds_since(started);
        if (!status.ok()) fail(status.errors.front());
        const auto after = backend.stats();
        result.kernel_seconds.push_back(
            static_cast<double>(after.kernel_nanoseconds - before.kernel_nanoseconds) / 1.0e9);
        result.service_seconds.push_back(service);
        for (const float value : output) {
            if (std::abs(value - static_cast<float>(hidden)) > 0.5F) {
                result.verified = false;
                break;
            }
        }
    }
    return result;
}

struct ExpertServiceResult {
    int device{};
    int numa_node{};
    std::uint64_t weight_bytes{};
    std::vector<double> seconds;
    std::vector<double> upload_wait_seconds;
    std::vector<double> kernel_seconds;
    bool verified{true};
};

ExpertServiceResult expert_service_samples(int device, int numa_node,
                                           unsigned int repetitions) {
    constexpr std::uint64_t hidden = 6144U;
    constexpr std::uint64_t intermediate = 2048U;
    ExpertServiceResult result;
    result.device = device;
    result.numa_node = numa_node;
    for (unsigned int repetition = 0; repetition < repetitions; ++repetition) {
        strata::CudaBackend backend;
        const std::array<int, 1> selected{device};
        auto initialized = backend.initialize(selected, true);
        if (!initialized.ok()) fail(initialized.errors.front());
        std::vector<strata::CudaWeight> device_weights(3);
        const std::array<std::pair<std::uint64_t, std::uint64_t>, 3> shapes{{
            {intermediate, hidden}, {intermediate, hidden}, {hidden, intermediate}}};
        std::vector<std::unique_ptr<NumaBuffer>> host_weights;
        std::vector<std::unique_ptr<NumaBuffer>> host_scales;
        std::vector<strata::CudaWeightDescriptor> descriptors;
        for (const auto& [rows, columns] : shapes) {
            const auto packed_columns = (columns + 7U) / 8U;
            const auto scale_columns = (columns + 127U) / 128U;
            host_weights.push_back(std::make_unique<NumaBuffer>(
                rows * packed_columns * 4U, numa_node));
            host_scales.push_back(std::make_unique<NumaBuffer>(
                rows * scale_columns * 2U, numa_node));
            fill_quantized(*host_weights.back(), *host_scales.back(), 4U);
            host_weights.back()->pin();
            host_scales.back()->pin();
            strata::CudaWeightDescriptor descriptor;
            descriptor.encoding = strata::CudaWeightEncoding::OffsetPackedInt4;
            descriptor.dtype = strata::SafetensorsDtype::I32;
            descriptor.rows = rows;
            descriptor.columns = columns;
            descriptor.packed_columns = packed_columns;
            descriptor.scale_columns = scale_columns;
            descriptor.group_size = 128U;
            descriptors.push_back(descriptor);
        }
        std::vector<float> hidden_input(hidden, 1.0F);
        std::vector<float> intermediate_input(intermediate, 1.0F);
        std::vector<float> intermediate_output(intermediate);
        std::vector<float> hidden_output(hidden);
        const auto started = Clock::now();
        for (std::size_t index = 0; index < descriptors.size(); ++index) {
            const auto status = backend.upload(device, descriptors[index],
                                               host_weights[index]->span(),
                                               host_scales[index]->span(),
                                               device_weights[index]);
            if (!status.ok()) fail(status.errors.front());
        }
        observe_vram(device);
        for (std::size_t index = 0; index < 2U; ++index) {
            const auto status = backend.matmul(device_weights[index], hidden_input, 1U,
                                               intermediate_output);
            if (!status.ok()) fail(status.errors.front());
            for (const float value : intermediate_output) {
                if (std::abs(value - static_cast<float>(hidden)) > 0.5F) {
                    result.verified = false;
                    break;
                }
            }
        }
        const auto down_status = backend.matmul(device_weights[2], intermediate_input, 1U,
                                                hidden_output);
        if (!down_status.ok()) fail(down_status.errors.front());
        observe_vram(device);
        for (const float value : hidden_output) {
            if (std::abs(value - static_cast<float>(intermediate)) > 0.5F) {
                result.verified = false;
                break;
            }
        }
        result.seconds.push_back(seconds_since(started));
        const auto stats = backend.stats();
        result.upload_wait_seconds.push_back(
            static_cast<double>(stats.upload_wait_nanoseconds) / 1.0e9);
        result.kernel_seconds.push_back(static_cast<double>(stats.kernel_nanoseconds) / 1.0e9);
        result.weight_bytes = stats.weight_upload_bytes;
    }
    return result;
}

struct IoResult {
    std::string mode;
    unsigned int queue_depth{};
    std::vector<TimedSample> samples;
};

std::uint64_t process_physical_read_bytes() {
    std::ifstream input("/proc/self/io");
    std::string key;
    std::uint64_t value = 0U;
    while (input >> key >> value) {
        if (key == "read_bytes:") return value;
    }
    fail("cannot read process physical I/O counters");
}

TimedSample read_ranges(const std::string& path, bool direct, unsigned int queue_depth,
                        std::size_t total_bytes, std::size_t block_bytes,
                        off_t base_offset) {
    const int flags = O_RDONLY | (direct ? O_DIRECT : 0);
    const int descriptor = open(path.c_str(), flags);
    if (descriptor < 0) fail("open checkpoint range: " + std::string(std::strerror(errno)));
    const std::size_t blocks = total_bytes / block_bytes;
    void* raw = nullptr;
    if (posix_memalign(&raw, 4096U, total_bytes) != 0) {
        close(descriptor);
        fail("allocate aligned checkpoint-read buffer");
    }
    auto* buffer = static_cast<std::byte*>(raw);
    std::atomic<std::size_t> next{0U};
    std::atomic<bool> complete{true};
    const auto physical_before = process_physical_read_bytes();
    const auto started = Clock::now();
    std::vector<std::thread> workers;
    for (unsigned int worker = 0; worker < queue_depth; ++worker) {
        workers.emplace_back([&]() {
            while (true) {
                const auto block = next.fetch_add(1U);
                if (block >= blocks) break;
                const auto offset = base_offset + static_cast<off_t>(block * block_bytes);
                auto* block_buffer = buffer + block * block_bytes;
                std::size_t done = 0U;
                while (done < block_bytes) {
                    const auto count = pread(descriptor, block_buffer + done,
                                             block_bytes - done,
                                             offset + static_cast<off_t>(done));
                    if (count <= 0) {
                        complete.store(false);
                        break;
                    }
                    done += static_cast<std::size_t>(count);
                }
                if (done != block_bytes) break;
            }
        });
    }
    for (auto& worker : workers) worker.join();
    const double seconds = seconds_since(started);
    const auto physical_after = process_physical_read_bytes();
    const auto result_checksum = checksum({buffer, total_bytes});
    std::free(raw);
    close(descriptor);
    return {seconds, total_bytes, complete.load(), result_checksum,
            physical_after - physical_before};
}

std::vector<IoResult> io_samples(const Options& options) {
    struct stat status{};
    if (stat(options.checkpoint_file.c_str(), &status) != 0) {
        fail("stat checkpoint file: " + std::string(std::strerror(errno)));
    }
    const auto file_bytes = static_cast<std::uint64_t>(status.st_size);
    const auto needed = static_cast<std::uint64_t>(options.io_bytes) *
                        options.repetitions * options.queue_depths.size();
    if (file_bytes < options.io_bytes + options.io_block_bytes) {
        fail("checkpoint file is too small for requested I/O extent");
    }
    std::vector<IoResult> results;
    std::uint64_t direct_cursor = 0U;
    for (const auto depth : options.queue_depths) {
        IoResult hot{"page_cache_hot", depth, {}};
        IoResult physical{"direct_physical", depth, {}};
        for (unsigned int repetition = 0; repetition < options.repetitions; ++repetition) {
            const auto hot_offset = static_cast<off_t>(
                (static_cast<std::uint64_t>(repetition) * options.io_bytes) %
                (file_bytes - options.io_bytes));
            const auto aligned_hot = hot_offset - hot_offset % 4096;
            const auto warm = read_ranges(options.checkpoint_file, false, depth,
                                          options.io_bytes, options.io_block_bytes,
                                          aligned_hot);
            if (!warm.verified) fail("page-cache warm-up read was incomplete");
            hot.samples.push_back(read_ranges(options.checkpoint_file, false, depth,
                                              options.io_bytes, options.io_block_bytes,
                                              aligned_hot));
            const auto usable = file_bytes - options.io_bytes;
            const auto direct_offset = static_cast<off_t>((direct_cursor % usable) / 4096U * 4096U);
            physical.samples.push_back(read_ranges(options.checkpoint_file, true, depth,
                                                   options.io_bytes, options.io_block_bytes,
                                                   direct_offset));
            direct_cursor += options.io_bytes;
        }
        results.push_back(std::move(hot));
        results.push_back(std::move(physical));
    }
    static_cast<void>(needed);
    return results;
}

void write_double_samples(std::ostream& output, const std::vector<double>& samples) {
    output << '[';
    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (index != 0U) output << ',';
        output << samples[index];
    }
    output << ']';
}

void write_timed_samples(std::ostream& output, const std::vector<TimedSample>& samples) {
    output << '[';
    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (index != 0U) output << ',';
        const auto& sample = samples[index];
        output << "{\"seconds\":" << sample.seconds
               << ",\"bytes\":" << sample.bytes
               << ",\"gb_s\":" << (static_cast<double>(sample.bytes) / sample.seconds / 1.0e9)
               << ",\"verified\":" << (sample.verified ? "true" : "false")
               << ",\"checksum\":" << sample.checksum
               << ",\"physical_read_bytes\":" << sample.physical_read_bytes << '}';
    }
    output << ']';
}

double timed_median_gbps(const std::vector<TimedSample>& samples) {
    std::vector<double> values;
    for (const auto& sample : samples) {
        values.push_back(static_cast<double>(sample.bytes) / sample.seconds / 1.0e9);
    }
    return median(std::move(values));
}

bool all_verified(const std::vector<TimedSample>& samples) {
    return std::all_of(samples.begin(), samples.end(),
                       [](const auto& sample) { return sample.verified; });
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        if (numa_available() < 0) fail("NUMA is unavailable");
        int device_count = 0;
        cuda_check(cudaGetDeviceCount(&device_count), "enumerate CUDA devices");
        for (const int device : options.devices) {
            if (device < 0 || device >= device_count) fail("requested CUDA device is unavailable");
        }
        for (const int node : options.numa_nodes) {
            if (node < 0 || node > numa_max_node()) fail("requested NUMA node is unavailable");
        }

        int driver_version = 0;
        int runtime_version = 0;
        cuda_check(cudaDriverGetVersion(&driver_version), "query CUDA driver version");
        cuda_check(cudaRuntimeGetVersion(&runtime_version), "query CUDA runtime version");
        struct utsname system_identity {};
        if (uname(&system_identity) != 0) fail("uname failed");
        char hostname[256]{};
        if (gethostname(hostname, sizeof(hostname)) != 0) fail("gethostname failed");

        std::vector<DeviceIdentity> identities;
        for (const int device : options.devices) identities.push_back(inspect_device(device));

        struct TransferResult {
            int device{};
            int node{};
            bool pinned{};
            bool h2d{};
            double registration_seconds{};
            std::vector<TimedSample> samples;
        };
        std::vector<TransferResult> transfers;
        for (const int device : options.devices) {
            std::cerr << "[T1] transfers device " << device << '\n';
            for (const int node : options.numa_nodes) {
                for (const bool pinned : {false, true}) {
                    for (const bool h2d : {true, false}) {
                        double registration_seconds = 0.0;
                        auto samples = transfer_samples(device, node, options.transfer_bytes,
                                                        options.repetitions, pinned, h2d,
                                                        registration_seconds);
                        if (!all_verified(samples)) fail("transfer validation failed");
                        transfers.push_back({device, node, pinned, h2d,
                                             registration_seconds, std::move(samples)});
                    }
                }
            }
        }

        std::vector<std::pair<int, std::vector<double>>> allocations;
        std::vector<std::pair<int, std::vector<double>>> synchronizations;
        for (const int device : options.devices) {
            allocations.emplace_back(device, allocation_samples(
                device, options.transfer_bytes, options.repetitions));
            synchronizations.emplace_back(device, synchronization_samples(
                device, options.repetitions));
        }

        std::vector<PeerResult> peers;
        for (const int source : options.devices) {
            for (const int destination : options.devices) {
                if (source == destination) continue;
                peers.push_back(peer_samples(source, destination, options.activation_bytes,
                                             options.repetitions));
                if (!all_verified(peers.back().samples)) fail("inter-GPU transfer validation failed");
            }
        }

        std::vector<KernelResult> kernels;
        for (const int device : options.devices) {
            std::cerr << "[T1] kernels device " << device << '\n';
            const int source_node = inspect_device(device).numa_node >= 0
                                        ? inspect_device(device).numa_node
                                        : options.numa_nodes.front();
            for (const std::uint32_t bits : {4U, 8U}) {
                kernels.push_back(kernel_samples(device, source_node, bits, 1U,
                                                 options.repetitions));
                kernels.push_back(kernel_samples(device, source_node, bits,
                                                 options.prefill_rows,
                                                 options.repetitions));
                if (!kernels[kernels.size() - 1U].verified ||
                    !kernels[kernels.size() - 2U].verified) {
                    fail("kernel reference validation failed");
                }
            }
        }

        std::vector<ExpertServiceResult> experts;
        for (const int device : options.devices) {
            std::cerr << "[T1] cold expert service device " << device << '\n';
            for (const int node : options.numa_nodes) {
                experts.push_back(expert_service_samples(device, node,
                                                         options.repetitions));
                if (!experts.back().verified) fail("expert service reference validation failed");
            }
        }

        std::cerr << "[T1] checkpoint ranges\n";
        const auto io = io_samples(options);
        for (const auto& result : io) {
            if (!all_verified(result.samples)) fail("checkpoint range read was incomplete");
        }

        std::cout << std::setprecision(10);
        std::cout << "{\"schema\":\"strata.topology_probe\",\"version\":1";
        std::cout << ",\"identity\":{\"hostname\":\""
                  << strata::cli::json_escape(hostname)
                  << "\",\"kernel\":\""
                  << strata::cli::json_escape(system_identity.release)
                  << "\",\"cuda_driver_version\":" << driver_version
                  << ",\"cuda_runtime_version\":" << runtime_version << '}';
        std::cout << ",\"configuration\":{\"repetitions\":" << options.repetitions
                  << ",\"transfer_bytes\":" << options.transfer_bytes
                  << ",\"activation_bytes\":" << options.activation_bytes
                  << ",\"io_bytes\":" << options.io_bytes
                  << ",\"io_block_bytes\":" << options.io_block_bytes
                  << ",\"prefill_rows\":" << options.prefill_rows
                  << ",\"checkpoint_file\":\""
                  << strata::cli::json_escape(options.checkpoint_file)
                  << "\"}";
        std::cout << ",\"devices\":[";
        for (std::size_t index = 0; index < identities.size(); ++index) {
            if (index != 0U) std::cout << ',';
            const auto& device = identities[index];
            std::cout << "{\"cuda_index\":" << device.index
                      << ",\"name\":\"" << strata::cli::json_escape(device.name)
                      << "\",\"pci_bdf\":\"" << strata::cli::json_escape(device.pci_bdf)
                      << "\",\"uuid\":\"" << device.uuid
                      << "\",\"compute_capability\":\"" << device.compute_major << '.'
                      << device.compute_minor << "\",\"free_vram_bytes\":" << device.free_bytes
                      << ",\"total_vram_bytes\":" << device.total_bytes
                      << ",\"numa_node\":" << device.numa_node
                      << ",\"pcie_current_speed\":\""
                      << strata::cli::json_escape(device.link_speed)
                      << "\",\"pcie_current_width\":\""
                      << strata::cli::json_escape(device.link_width)
                      << "\"}";
        }
        std::cout << ']';
        std::cout << ",\"vram_peak\":[";
        for (std::size_t index = 0; index < options.devices.size(); ++index) {
            if (index != 0U) std::cout << ',';
            const int device = options.devices[index];
            std::cout << "{\"device\":" << device
                      << ",\"observed_used_bytes\":" << observed_vram_used[device] << '}';
        }
        std::cout << ']';
        std::cout << ",\"transfers\":[";
        for (std::size_t index = 0; index < transfers.size(); ++index) {
            if (index != 0U) std::cout << ',';
            const auto& transfer = transfers[index];
            std::cout << "{\"device\":" << transfer.device
                      << ",\"numa_node\":" << transfer.node
                      << ",\"memory\":\"" << (transfer.pinned ? "pinned" : "pageable")
                      << "\",\"direction\":\"" << (transfer.h2d ? "h2d" : "d2h")
                      << "\",\"registration_seconds\":" << transfer.registration_seconds
                      << ",\"median_gb_s\":" << timed_median_gbps(transfer.samples)
                      << ",\"samples\":";
            write_timed_samples(std::cout, transfer.samples);
            std::cout << '}';
        }
        std::cout << ']';
        std::cout << ",\"allocation\":[";
        for (std::size_t index = 0; index < allocations.size(); ++index) {
            if (index != 0U) std::cout << ',';
            std::cout << "{\"device\":" << allocations[index].first
                      << ",\"bytes\":" << options.transfer_bytes
                      << ",\"median_seconds\":" << median(allocations[index].second)
                      << ",\"samples_seconds\":";
            write_double_samples(std::cout, allocations[index].second);
            std::cout << '}';
        }
        std::cout << ']';
        std::cout << ",\"synchronization\":[";
        for (std::size_t index = 0; index < synchronizations.size(); ++index) {
            if (index != 0U) std::cout << ',';
            std::cout << "{\"device\":" << synchronizations[index].first
                      << ",\"median_seconds\":" << median(synchronizations[index].second)
                      << ",\"samples_seconds\":";
            write_double_samples(std::cout, synchronizations[index].second);
            std::cout << '}';
        }
        std::cout << ']';
        std::cout << ",\"device_transfers\":[";
        for (std::size_t index = 0; index < peers.size(); ++index) {
            if (index != 0U) std::cout << ',';
            const auto& peer = peers[index];
            std::cout << "{\"source\":" << peer.source
                      << ",\"destination\":" << peer.destination
                      << ",\"peer_capable\":" << (peer.peer_capable ? "true" : "false")
                      << ",\"path\":\"" << peer.path
                      << "\",\"median_gb_s\":" << timed_median_gbps(peer.samples)
                      << ",\"samples\":";
            write_timed_samples(std::cout, peer.samples);
            std::cout << '}';
        }
        std::cout << ']';
        std::cout << ",\"kernels\":[";
        for (std::size_t index = 0; index < kernels.size(); ++index) {
            if (index != 0U) std::cout << ',';
            const auto& kernel = kernels[index];
            std::cout << "{\"device\":" << kernel.device
                      << ",\"encoding\":\"" << kernel.encoding
                      << "\",\"batch_rows\":" << kernel.rows
                      << ",\"output_rows\":" << kernel.output_rows
                      << ",\"columns\":" << kernel.columns
                      << ",\"weight_bytes\":" << kernel.weight_bytes
                      << ",\"verified\":" << (kernel.verified ? "true" : "false")
                      << ",\"kernel_seconds_median\":" << median(kernel.kernel_seconds)
                      << ",\"service_seconds_median\":" << median(kernel.service_seconds)
                      << ",\"kernel_seconds\":";
            write_double_samples(std::cout, kernel.kernel_seconds);
            std::cout << ",\"service_seconds\":";
            write_double_samples(std::cout, kernel.service_seconds);
            std::cout << '}';
        }
        std::cout << ']';
        std::cout << ",\"row1_expert_service\":[";
        for (std::size_t index = 0; index < experts.size(); ++index) {
            if (index != 0U) std::cout << ',';
            const auto& expert = experts[index];
            std::cout << "{\"device\":" << expert.device
                      << ",\"numa_node\":" << expert.numa_node
                      << ",\"encoding\":\"int4_group128\""
                      << ",\"weight_bytes\":" << expert.weight_bytes
                      << ",\"verified\":" << (expert.verified ? "true" : "false")
                      << ",\"median_seconds\":" << median(expert.seconds)
                      << ",\"median_upload_wait_seconds\":" << median(expert.upload_wait_seconds)
                      << ",\"median_kernel_seconds\":" << median(expert.kernel_seconds)
                      << ",\"samples_seconds\":";
            write_double_samples(std::cout, expert.seconds);
            std::cout << ",\"upload_wait_seconds\":";
            write_double_samples(std::cout, expert.upload_wait_seconds);
            std::cout << ",\"kernel_seconds\":";
            write_double_samples(std::cout, expert.kernel_seconds);
            std::cout << '}';
        }
        std::cout << ']';
        std::cout << ",\"checkpoint_reads\":[";
        for (std::size_t index = 0; index < io.size(); ++index) {
            if (index != 0U) std::cout << ',';
            const auto& read = io[index];
            std::cout << "{\"mode\":\"" << read.mode
                      << "\",\"queue_depth\":" << read.queue_depth
                      << ",\"median_gb_s\":" << timed_median_gbps(read.samples)
                      << ",\"samples\":";
            write_timed_samples(std::cout, read.samples);
            std::cout << '}';
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
