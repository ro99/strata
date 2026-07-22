#include "strata/cuda_backend.hpp"

#include <limits>

namespace strata {

struct CudaWeight::Impl {};
struct CudaBuffer::Impl {};
struct CudaBackend::Impl {};

CudaWeight::CudaWeight() = default;
CudaWeight::~CudaWeight() = default;
CudaWeight::CudaWeight(CudaWeight&&) noexcept = default;
CudaWeight& CudaWeight::operator=(CudaWeight&&) noexcept = default;
bool CudaWeight::valid() const noexcept { return false; }
std::uint64_t CudaWeight::device_bytes() const noexcept { return 0U; }
int CudaWeight::device() const noexcept { return -1; }

CudaBuffer::CudaBuffer() = default;
CudaBuffer::~CudaBuffer() = default;
CudaBuffer::CudaBuffer(CudaBuffer&&) noexcept = default;
CudaBuffer& CudaBuffer::operator=(CudaBuffer&&) noexcept = default;
bool CudaBuffer::valid() const noexcept { return false; }
std::uint64_t CudaBuffer::device_bytes() const noexcept { return 0U; }
int CudaBuffer::device() const noexcept { return -1; }

CudaBackend::CudaBackend() : impl_(std::make_unique<Impl>()) {}
CudaBackend::~CudaBackend() = default;
CudaBackend::CudaBackend(CudaBackend&&) noexcept = default;
CudaBackend& CudaBackend::operator=(CudaBackend&&) noexcept = default;
bool CudaBackend::compiled() noexcept { return false; }
std::vector<int> CudaBackend::available_devices() { return {}; }
ParseResult<CudaDeviceMemory> CudaBackend::device_memory(int) {
    return {{}, {"CUDA support was not compiled into this build"}};
}

std::uint64_t CudaBackend::weight_storage_bytes(
    std::uint64_t weight_bytes, std::uint64_t scale_bytes) noexcept {
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    if (weight_bytes == 0U || weight_bytes > maximum - 15U) return 0U;
    const auto scale_offset = (weight_bytes + 15U) & ~std::uint64_t{15U};
    if (scale_bytes > maximum - scale_offset) return 0U;
    const auto total = scale_offset + scale_bytes;
    if (total > maximum - 255U) return 0U;
    return (total + 255U) & ~std::uint64_t{255U};
}

ValidationResult CudaBackend::initialize(std::span<const int>, bool) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::reserve_weight_arena(int, std::uint64_t) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::upload(int, const CudaWeightDescriptor&,
                                     std::span<const std::byte>,
                                     std::span<const std::byte>, CudaWeight&) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::upload_buffer(
    int, std::span<const std::byte>, CudaBuffer&) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::matmul(const CudaWeight&, std::span<const float>,
                                     std::uint32_t, std::span<float>) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::matmul_grouped(
    const CudaWeight&, std::span<const float>, std::uint32_t,
    std::uint64_t, std::span<float>) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::validate_flash_attention_device(int) const {
    return {{"FlashAttention requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::flash_attention(
    int, const FlashAttentionRequest&, std::span<float>) {
    return {{"FlashAttention requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::enqueue_deepseek_moe(
    int, std::span<const float>,
    std::span<const CudaDeepSeekMoeExpert>,
    const CudaDeepSeekMoeExpert*, float) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::collect_deepseek_moe(
    int, std::span<float>, std::span<float>) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::matmul_impl(
    const CudaWeight&, std::span<const float>, std::uint32_t,
    std::uint32_t, std::uint64_t, std::span<float>) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::synchronize(int) {
    return {{"CUDA support was not compiled into this build"}};
}

CudaBackendStats CudaBackend::stats() const noexcept { return {}; }

}  // namespace strata
