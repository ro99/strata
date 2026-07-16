#include "strata/cuda_backend.hpp"

namespace strata {

struct CudaWeight::Impl {};
struct CudaBackend::Impl {};

CudaWeight::CudaWeight() = default;
CudaWeight::~CudaWeight() = default;
CudaWeight::CudaWeight(CudaWeight&&) noexcept = default;
CudaWeight& CudaWeight::operator=(CudaWeight&&) noexcept = default;
bool CudaWeight::valid() const noexcept { return false; }
std::uint64_t CudaWeight::device_bytes() const noexcept { return 0U; }
int CudaWeight::device() const noexcept { return -1; }

CudaBackend::CudaBackend() : impl_(std::make_unique<Impl>()) {}
CudaBackend::~CudaBackend() = default;
CudaBackend::CudaBackend(CudaBackend&&) noexcept = default;
CudaBackend& CudaBackend::operator=(CudaBackend&&) noexcept = default;
bool CudaBackend::compiled() noexcept { return false; }
std::vector<int> CudaBackend::available_devices() { return {}; }
ParseResult<CudaDeviceMemory> CudaBackend::device_memory(int) {
    return {{}, {"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::initialize(std::span<const int>, bool) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::upload(int, const CudaWeightDescriptor&,
                                     std::span<const std::byte>,
                                     std::span<const std::byte>, CudaWeight&) {
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
