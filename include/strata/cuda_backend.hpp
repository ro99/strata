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
    Fp8E4m3Block128,
    Fp4E2m1Group32,
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
        std::uint64_t deepseek_moe_calls{};
        std::uint64_t deepseek_moe_kernel_launches{};
        std::uint64_t deepseek_moe_h2d_transfers{};
        std::uint64_t deepseek_moe_d2h_transfers{};
        std::uint64_t deepseek_moe_h2d_bytes{};
        std::uint64_t deepseek_moe_d2h_bytes{};
        std::uint64_t deepseek_moe_h2d_nanoseconds{};
        std::uint64_t deepseek_moe_kernel_nanoseconds{};
        std::uint64_t deepseek_moe_d2h_nanoseconds{};
        std::uint64_t deepseek_moe_nanoseconds{};
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
    std::uint64_t deepseek_moe_calls{};
    std::uint64_t deepseek_moe_kernel_launches{};
    std::uint64_t deepseek_moe_h2d_transfers{};
    std::uint64_t deepseek_moe_d2h_transfers{};
    std::uint64_t deepseek_moe_h2d_bytes{};
    std::uint64_t deepseek_moe_d2h_bytes{};
    std::uint64_t deepseek_moe_h2d_nanoseconds{};
    std::uint64_t deepseek_moe_kernel_nanoseconds{};
    std::uint64_t deepseek_moe_d2h_nanoseconds{};
    std::uint64_t deepseek_moe_nanoseconds{};
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
    std::shared_ptr<Impl> impl_;
    friend class CudaBackend;
};

// One exact DeepSeek expert projection triplet. The weight objects must remain
// alive until the matching collect call completes. Routed coefficients are
// applied twice; the optional shared expert must use coefficient 1.0.
struct CudaDeepSeekMoeExpert {
    const CudaWeight* w1{};
    const CudaWeight* w3{};
    const CudaWeight* w2{};
    float coefficient{1.0F};
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
    [[nodiscard]] static std::uint64_t weight_storage_bytes(
        std::uint64_t weight_bytes, std::uint64_t scale_bytes) noexcept;

    [[nodiscard]] ValidationResult initialize(std::span<const int> devices,
                                              bool detailed_timing = false);
    // Reserve one capacity-bounded allocation per device. Subsequent uploads
    // suballocate from it and fail explicitly when it is exhausted; there is
    // no per-weight cudaMalloc fallback once the arena is enabled.
    [[nodiscard]] ValidationResult reserve_weight_arena(int device,
                                                        std::uint64_t bytes);
    [[nodiscard]] ValidationResult upload(
        int device, const CudaWeightDescriptor& descriptor,
        std::span<const std::byte> weights, std::span<const std::byte> scales,
        CudaWeight& output);
    [[nodiscard]] ValidationResult matmul(
        const CudaWeight& weight, std::span<const float> input,
        std::uint32_t rows, std::span<float> output);
    [[nodiscard]] ValidationResult matmul_grouped(
        const CudaWeight& weight, std::span<const float> input,
        std::uint32_t groups, std::uint64_t rows_per_group,
        std::span<float> output);
    // Enqueue first on every active device, then collect each device. Routed
    // results are flattened in the same order as `routed`; shared output is
    // returned separately so the caller retains the global accumulation order.
    // An enqueue error drains submitted work before returning. Every collect
    // return consumes or drains its pending command and outputs are invalid on
    // error. Downloads target backend-owned pinned staging and touch caller
    // spans only after confirmed completion. If CUDA cannot confirm a drain
    // after a fatal device error, the backend retains the weights and poisons
    // that workspace instead of reusing it or freeing device storage early.
    [[nodiscard]] ValidationResult enqueue_deepseek_moe(
        int device, std::span<const float> hidden,
        std::span<const CudaDeepSeekMoeExpert> routed,
        const CudaDeepSeekMoeExpert* shared, float swiglu_limit);
    [[nodiscard]] ValidationResult collect_deepseek_moe(
        int device, std::span<float> routed_output,
        std::span<float> shared_output);
    [[nodiscard]] ValidationResult synchronize(int device);

    [[nodiscard]] CudaBackendStats stats() const noexcept;

private:
    [[nodiscard]] ValidationResult matmul_impl(
        const CudaWeight& weight, std::span<const float> input,
        std::uint32_t rows, std::uint32_t groups,
        std::uint64_t rows_per_group, std::span<float> output);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata
