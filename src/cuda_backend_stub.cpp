#include "strata/cuda_backend.hpp"

#include <limits>

namespace strata {

struct CudaWeight::Impl {};
struct CudaBuffer::Impl {};
struct CudaDsv4MhcWeights::Impl {};
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

CudaDsv4MhcWeights::CudaDsv4MhcWeights() = default;
CudaDsv4MhcWeights::~CudaDsv4MhcWeights() = default;
CudaDsv4MhcWeights::CudaDsv4MhcWeights(CudaDsv4MhcWeights&&) noexcept = default;
CudaDsv4MhcWeights& CudaDsv4MhcWeights::operator=(
    CudaDsv4MhcWeights&&) noexcept = default;
bool CudaDsv4MhcWeights::valid() const noexcept { return false; }
std::uint64_t CudaDsv4MhcWeights::device_bytes() const noexcept { return 0U; }
int CudaDsv4MhcWeights::device() const noexcept { return -1; }

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

ValidationResult CudaBackend::register_host_memory(const void*, std::uint64_t) {
    return {{"CUDA support was not compiled into this build"}};
}

void CudaBackend::unregister_host_memory(const void*) noexcept {}

ValidationResult CudaBackend::upload(int, const CudaWeightDescriptor&,
                                     std::span<const std::byte>,
                                     std::span<const std::byte>, CudaWeight&,
                                     UploadCompletion) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::synchronize_uploads(int) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::upload_buffer(
    int, std::span<const std::byte>, CudaBuffer&) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::update_buffer(
    const CudaBuffer&, std::span<const CudaBufferPatch>) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::download_buffer(
    const CudaBuffer&, std::uint64_t, std::span<std::byte>) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::matmul(const CudaWeight&, std::span<const float>,
                                     std::uint32_t, std::span<float>, bool,
                                     CudaMatmulProfile*, bool) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::matmul_softcap(
    const CudaWeight&, std::span<const float>, float, std::span<float>) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::matmul_grouped(
    const CudaWeight&, std::span<const float>, std::uint32_t,
    std::uint64_t, std::span<float>) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::matmul_grouped_rows(
    const CudaWeight&, std::span<const float>, std::uint32_t,
    std::uint32_t, std::uint64_t, std::span<float>) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::validate_flash_attention_device(int) const {
    return {{"FlashAttention requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::validate_lightning_index_device(int) const {
    return {{"Lightning Indexer requires a CUDA-enabled build"}};
}

bool CudaBackend::dsv4_fp8_tensor_page_supported(int) const noexcept {
    return false;
}

ValidationResult CudaBackend::validate_dsv4_mhc_device(int) const {
    return {{"DeepSeek device mHC requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::flash_attention(
    int, const FlashAttentionRequest&, std::span<float>) {
    return {{"FlashAttention requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::dsv4_paged_attention(
    int, const CudaDsv4PagedAttentionRequest&, std::span<float>) {
    return {{"DeepSeek paged attention requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::dsv4_paged_attention_to_mhc(
    int, const CudaDsv4PagedAttentionMhcRequest&, std::span<float>) {
    return {{"DeepSeek paged attention requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::dsv4_prepare_attention(
    int, const CudaDsv4AttentionPrepareRequest&, std::span<float>,
    std::span<float>, std::span<float>, std::span<float>, std::span<float>,
    std::span<float>) {
    return {{"DeepSeek attention preparation requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::dsv4_copy_prepared_queries(
    int, std::span<float>) {
    return {{"DeepSeek prepared query capture requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::upload_dsv4_mhc_weights(
    int, std::span<const float>, std::span<const float>,
    std::span<const float>, std::span<const float>, CudaDsv4MhcWeights&) {
    return {{"DeepSeek device mHC requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::dsv4_mhc_select_slot(int, std::uint32_t) {
    return {{"DeepSeek device mHC requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::dsv4_mhc_reserve_slots(int, std::uint32_t) {
    return {{"DeepSeek device mHC requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::dsv4_mhc_begin(
    int, const CudaDsv4MhcWeights&, std::span<const float>,
    std::span<float>, std::span<float>) {
    return {{"DeepSeek device mHC requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::dsv4_mhc_transition(
    int, const CudaDsv4MhcWeights&, std::span<const float>,
    std::span<float>, std::span<float>, std::span<float>) {
    return {{"DeepSeek device mHC requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::dsv4_mhc_transition_device(
    int, const CudaDsv4MhcWeights&, std::span<float>, std::span<float>,
    std::span<float>) {
    return {{"DeepSeek device mHC requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::dsv4_mhc_finish(
    int, std::span<const float>, std::span<float>) {
    return {{"DeepSeek device mHC requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::dsv4_mhc_finish_device(
    int, std::span<float>) {
    return {{"DeepSeek device mHC requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::dsv4_mhc_device_view(
    int, CudaDsv4MhcDeviceView&) {
    return {{"DeepSeek device mHC requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::dsv4_mhc_branch_to_fp32(int, float*) {
    return {{"DeepSeek device mHC requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::dsv4_mhc_commit_reduced_branch(
    int, const float*) {
    return {{"DeepSeek device mHC requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::dsv4_mhc_abort_branch(int) {
    return {{"DeepSeek device mHC requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::dsv4_mhc_transition_router_device(
    int, const CudaDsv4MhcWeights&, const CudaWeight&) {
    return {{"DeepSeek device mHC requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::dsv4_mhc_transition_next_device(
    int, const CudaDsv4MhcWeights&) {
    return {{"DeepSeek device mHC requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::glm_absorbed_attention(
    const CudaWeight&, const CudaGlmAbsorbedAttentionRequest&,
    std::span<float>) {
    return {{"GLM absorbed attention requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::lightning_index(
    int, const CudaLightningIndexRequest&, std::span<std::uint32_t>) {
    return {{"Lightning Indexer requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::dsv4_index_query_rope_quantize(
    int, std::span<float>, std::span<const float>, std::span<const float>,
    std::uint32_t, std::uint32_t, std::uint32_t, bool) {
    return {{"index-query preparation requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::dsv4_index_projections(
    int, const CudaDsv4IndexProjectionRequest&, std::span<float>,
    std::span<float>) {
    return {{"index projections require a CUDA-enabled build"}};
}

ValidationResult CudaBackend::dsv4_physical_lightning_index(
    int, const CudaDsv4PhysicalIndexRequest&, std::span<std::uint32_t>,
    CudaDsv4DeviceIndexSelection*) {
    return {{"physical Lightning Indexer requires a CUDA-enabled build"}};
}

ValidationResult CudaBackend::enqueue_deepseek_moe(
    int, std::span<const float>,
    std::span<const CudaDeepSeekMoeExpert>,
    const CudaDeepSeekMoeExpert*, float) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::enqueue_dsv4_host_moe(
    int, std::span<const float>, const CudaDeepSeekMoeExpert&, float,
    CudaDsv4HostMoeCallback, void*) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::enqueue_dsv4_host_moe_device_view(
    int, std::span<const float>, const CudaDeepSeekMoeExpert&, float,
    CudaDsv4HostMoeCallback, void*, CudaDsv4HostMoeDeviceView&) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::enqueue_dsv4_host_moe_from_mhc(
    int, const CudaDeepSeekMoeExpert&, float,
    CudaDsv4HostMoeCallback, void*) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::enqueue_dsv4_host_moe_from_device_input(
    int, const CudaDeepSeekMoeExpert&, float,
    CudaDsv4DeviceInputHostMoeCallback, void*) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::enqueue_dsv4_host_moe_from_device_input_device_view(
    int, const CudaDeepSeekMoeExpert&, float,
    CudaDsv4DeviceInputHostMoeCallback, void*, CudaDsv4HostMoeDeviceView&) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::collect_deepseek_moe(
    int, std::span<float>, std::span<float>) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::finish_deepseek_moe_chain(int) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::enqueue_deepseek_moe_rows(
    int, std::span<const float>, std::uint32_t,
    std::span<const CudaDeepSeekMoeRowGroup>, const CudaDeepSeekMoeExpert*,
    std::span<const std::uint32_t>, float) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::collect_deepseek_moe_rows(
    int, std::span<float>, std::span<float>) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::enqueue_moe(
    int, std::span<const float>, std::uint32_t,
    std::span<const CudaMoeExpert>, const CudaMoeExpert*) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::collect_moe(
    int, std::span<float>, std::span<float>) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::matmul_impl(
    const CudaWeight&, std::span<const float>, std::uint32_t,
    std::uint32_t, std::uint64_t, std::span<float>, float, bool,
    CudaMatmulProfile*) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::synchronize(int) {
    return {{"CUDA support was not compiled into this build"}};
}

CudaBackendStats CudaBackend::stats() const noexcept { return {}; }

}  // namespace strata
