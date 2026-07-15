#pragma once

#include "strata/model.hpp"
#include "strata/safetensors.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace strata {

enum class CudaWeightEncoding : std::uint8_t {
    Plain,
    OffsetPackedInt4,
    OffsetPackedInt8,
};

struct CudaWeightDescriptor {
    CudaWeightEncoding encoding{CudaWeightEncoding::Plain};
    SafetensorsDtype dtype{SafetensorsDtype::Other};
    std::uint64_t rows{};
    std::uint64_t columns{};
    std::uint64_t packed_columns{};
    std::uint64_t scale_columns{};
    std::uint32_t group_size{};
};

struct CudaBackendStats {
    // Byte and event totals sum devices. Aggregate durations are the maximum
    // per-device service duration so concurrent device work is not double-counted.
    struct Device {
        int device{-1};
        std::uint64_t weight_upload_bytes{};
        std::uint64_t activation_h2d_bytes{};
        std::uint64_t activation_d2h_bytes{};
        std::uint64_t matmul_calls{};
        std::uint64_t weight_allocation_calls{};
        std::uint64_t weight_allocation_bytes{};
        std::uint64_t workspace_allocation_calls{};
        std::uint64_t workspace_allocation_bytes{};
        std::uint64_t synchronization_calls{};
        std::uint64_t synchronization_nanoseconds{};
        std::uint64_t upload_wait_nanoseconds{};
        std::uint64_t activation_h2d_nanoseconds{};
        std::uint64_t kernel_nanoseconds{};
        std::uint64_t activation_d2h_nanoseconds{};
    };

    std::uint64_t weight_upload_bytes{};
    std::uint64_t activation_h2d_bytes{};
    std::uint64_t activation_d2h_bytes{};
    std::uint64_t matmul_calls{};
    std::uint64_t weight_allocation_calls{};
    std::uint64_t weight_allocation_bytes{};
    std::uint64_t workspace_allocation_calls{};
    std::uint64_t workspace_allocation_bytes{};
    std::uint64_t synchronization_calls{};
    std::uint64_t synchronization_nanoseconds{};
    std::uint64_t upload_wait_nanoseconds{};
    std::uint64_t activation_h2d_nanoseconds{};
    std::uint64_t kernel_nanoseconds{};
    std::uint64_t activation_d2h_nanoseconds{};
    std::vector<Device> devices;
};

struct CudaDeviceMemory {
    std::uint64_t free_bytes{};
    std::uint64_t total_bytes{};
};

class CudaWeight {
public:
    CudaWeight();
    ~CudaWeight();
    CudaWeight(CudaWeight&&) noexcept;
    CudaWeight& operator=(CudaWeight&&) noexcept;
    CudaWeight(const CudaWeight&) = delete;
    CudaWeight& operator=(const CudaWeight&) = delete;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint64_t device_bytes() const noexcept;
    [[nodiscard]] int device() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class CudaBackend;
};

class CudaBackend {
public:
    CudaBackend();
    ~CudaBackend();
    CudaBackend(CudaBackend&&) noexcept;
    CudaBackend& operator=(CudaBackend&&) noexcept;
    CudaBackend(const CudaBackend&) = delete;
    CudaBackend& operator=(const CudaBackend&) = delete;

    [[nodiscard]] static bool compiled() noexcept;
    [[nodiscard]] static std::vector<int> available_devices();
    [[nodiscard]] static ParseResult<CudaDeviceMemory> device_memory(int device);

    [[nodiscard]] ValidationResult initialize(std::span<const int> devices,
                                              bool detailed_timing = false);
    [[nodiscard]] ValidationResult upload(
        int device, const CudaWeightDescriptor& descriptor,
        std::span<const std::byte> weights, std::span<const std::byte> scales,
        CudaWeight& output);
    [[nodiscard]] ValidationResult matmul(
        const CudaWeight& weight, std::span<const float> input,
        std::uint32_t rows, std::span<float> output);
    [[nodiscard]] ValidationResult synchronize(int device);

    [[nodiscard]] CudaBackendStats stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata
