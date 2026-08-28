#include "strata/device/cuda_backend.hpp"

#include <limits>

namespace strata {

namespace {

ValidationResult cuda_unavailable() {
    return {{"CUDA support was not compiled into this build"}};
}

}  // namespace

bool register_fed_matmul_enabled() noexcept { return false; }
void set_register_fed_matmul(bool) noexcept {}
CudaMatmulRouteCensus cuda_matmul_route_census() noexcept { return {}; }
void reset_cuda_matmul_route_census() noexcept {}
void record_cuda_matmul_route(CudaMatmulRoute) noexcept {}
const char* cuda_matmul_route_name(CudaMatmulRoute route) noexcept {
    switch (route) {
        case CudaMatmulRoute::PlainBf16Matvec: return "plain_bf16_matvec";
        case CudaMatmulRoute::PlainGeneric: return "plain_generic";
        case CudaMatmulRoute::PackedInt8Group32: return "packed_int8_group32";
        case CudaMatmulRoute::PackedOffsetInt: return "packed_offset_int";
        case CudaMatmulRoute::Nvfp4Group16: return "nvfp4_group16";
        case CudaMatmulRoute::Fp8TensorPage: return "fp8_tensor_page";
        case CudaMatmulRoute::Fp8F32TensorPage:
            return "fp8_f32_tensor_page";
        case CudaMatmulRoute::Fp8E4m3Block128: return "fp8_e4m3_block128";
        case CudaMatmulRoute::Fp8E4m3Block128F32:
            return "fp8_e4m3_block128_f32";
        case CudaMatmulRoute::Fp4E2m1Group32: return "fp4_e2m1_group32";
        case CudaMatmulRoute::Fp8RegisterFed: return "fp8_register_fed";
        case CudaMatmulRoute::Fp8F32RegisterFed:
            return "fp8_f32_register_fed";
        case CudaMatmulRoute::Fp4RegisterFed: return "fp4_register_fed";
        case CudaMatmulRoute::GemmaMarlin: return "gemma_marlin";
        case CudaMatmulRoute::MoePlainBf16: return "moe_plain_bf16";
        case CudaMatmulRoute::MoeFp8E4m3Block128F32:
            return "moe_fp8_e4m3_block128_f32";
        case CudaMatmulRoute::MoeFp8F32RegisterFed:
            return "moe_fp8_f32_register_fed";
        case CudaMatmulRoute::MoeNvfp4Group16: return "moe_nvfp4_group16";
        case CudaMatmulRoute::MoeFp4E2m1Group32: return "moe_fp4_e2m1_group32";
        case CudaMatmulRoute::MoePackedInt4: return "moe_packed_int4";
        case CudaMatmulRoute::MoeFp4RegisterFed: return "moe_fp4_register_fed";
        case CudaMatmulRoute::Dsv4MoeRoutedFp4: return "dsv4_moe_routed_fp4";
        case CudaMatmulRoute::Dsv4MoeSharedFp8: return "dsv4_moe_shared_fp8";
        case CudaMatmulRoute::Dsv4MoeSharedFp8RegisterFed:
            return "dsv4_moe_shared_fp8_register_fed";
        case CudaMatmulRoute::Dsv4MoeTierFp4: return "dsv4_moe_tier_fp4";
        case CudaMatmulRoute::Unsupported: return "unsupported";
        default: return "invalid";
    }
}

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
ValidationResult CudaBackend::prepack_fragment(int, const CudaWeight&) {
    return cuda_unavailable();
}
bool CudaBackend::fragment_prepacked(const CudaWeight&) noexcept { return false; }
ValidationResult CudaBackend::prepack_marlin(int, const CudaWeight&) {
    return cuda_unavailable();
}
bool CudaBackend::marlin_prepacked(const CudaWeight&) noexcept { return false; }
CudaMatmulRouteCensus CudaBackend::matmul_route_census() const noexcept {
    return {};
}
void CudaBackend::reset_matmul_route_census() noexcept {}
bool CudaBackend::compiled() noexcept { return false; }
std::vector<int> CudaBackend::available_devices() { return {}; }
ParseResult<CudaDeviceMemory> CudaBackend::device_memory(int) {
    return {{}, {"CUDA support was not compiled into this build"}};
}
int CudaBackend::device_numa_node(int) noexcept { return -1; }
bool CudaBackend::high_speed_peer_access_supported(int, int) noexcept {
    return false;
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
                                     UploadCompletion, FragmentLayout) {
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

ValidationResult CudaBackend::allocate_buffer(int, std::uint64_t, CudaBuffer&) {
    return cuda_unavailable();
}

ValidationResult CudaBackend::glm53_kda_decode(
    const CudaGlm53KdaRequest&, std::span<float>) {
    return cuda_unavailable();
}

ValidationResult CudaBackend::upload_gemma4_kv(
    const CudaBuffer&, std::span<const std::uint16_t>,
    std::span<const std::uint16_t>, std::uint32_t, std::uint32_t,
    std::uint32_t) {
    return cuda_unavailable();
}

ValidationResult CudaBackend::bf16_kv_attention(
    int, const CudaBf16KvAttentionRequest&, std::span<float>) {
    return cuda_unavailable();
}

ValidationResult CudaBackend::gemma4_decode_layers(
    int, std::span<const CudaGemma4DecodeLayer>, std::span<const float>,
    std::uint32_t, std::span<float>) {
    return cuda_unavailable();
}

ValidationResult CudaBackend::reserve_gemma4_workspace(
    int, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) {
    return cuda_unavailable();
}

ValidationResult CudaBackend::gemma4_prefill_layers(
    int, std::span<const CudaGemma4DecodeLayer>, std::span<const float>,
    std::uint32_t, std::uint32_t, std::span<float>) {
    return cuda_unavailable();
}

ValidationResult CudaBackend::matmul(const CudaWeight&, std::span<const float>,
                                     std::uint32_t, std::span<float>, bool,
                                     CudaMatmulProfile*, bool) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::matmul_batch(
    std::span<const CudaMatmulBatchItem>) {
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

bool CudaBackend::fp8_f32_tensor_page_supported(int) const noexcept {
    return false;
}

bool CudaBackend::fp8_f32_register_fed_supported(int) const noexcept {
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

ParseResult<std::uint64_t>
CudaBackend::dsv4_paged_attention_to_mhc_page_workspace_bytes(
    std::span<const CudaDsv4PhysicalPage>, std::uint32_t,
    std::uint32_t, bool) const {
    return {0U, {"DeepSeek paged attention requires a CUDA-enabled build"}};
}

ParseResult<std::uint32_t>
CudaBackend::dsv4_paged_attention_to_mhc_page_maximum_rows(
    std::span<const CudaDsv4PhysicalPage>, std::uint32_t, std::uint32_t,
    std::uint64_t, bool) const {
    return {0U, {"DeepSeek paged attention requires a CUDA-enabled build"}};
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

ValidationResult CudaBackend::dsv4_mhc_begin_device(
    int, const CudaDsv4MhcWeights&, std::span<const float>) {
    return cuda_unavailable();
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

ValidationResult CudaBackend::dsv4_mhc_download_layer_input(
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

ValidationResult CudaBackend::reserve_dsv4_mhc_head(int, std::uint64_t) {
    return cuda_unavailable();
}

ValidationResult CudaBackend::enqueue_dsv4_mhc_finish_head_device(
    int, const CudaWeight&, CudaDsv4MhcHeadCallback, void*,
    CudaDsv4MhcHeadDeviceView*) {
    return cuda_unavailable();
}

ValidationResult CudaBackend::complete_dsv4_mhc_head_device(
    int, std::span<float>) {
    return cuda_unavailable();
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
    CudaDsv4DeviceInputHostMoeCallback, void*, CudaDsv4HostMoeDeviceView&,
    CudaDsv4DeviceInputHostMoeRouteCallback) {
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
    std::span<const CudaMoeExpert>, const CudaMoeExpert*, float) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::dsv4_tier_reserve(
    int, std::uint32_t, std::uint32_t) {
    return cuda_unavailable();
}

ValidationResult CudaBackend::dsv4_tier_add(
    int, std::uint32_t, std::uint32_t, const CudaWeight&,
    const CudaWeight&, const CudaWeight&) {
    return cuda_unavailable();
}

ValidationResult CudaBackend::dsv4_tier_commit(int) {
    return cuda_unavailable();
}

bool CudaBackend::dsv4_tier_active(int) const noexcept { return false; }
CudaDsv4TierSelection* CudaBackend::dsv4_tier_selection(int) noexcept {
    return nullptr;
}

ValidationResult CudaBackend::collect_moe(
    int, std::span<float>, std::span<float>) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::matmul_impl(
    const CudaWeight&, std::span<const float>, std::uint32_t,
    std::uint32_t, std::uint64_t, std::span<float>, float, bool,
    CudaMatmulProfile*, bool, const std::byte*, std::byte*, bool) {
    return {{"CUDA support was not compiled into this build"}};
}

ValidationResult CudaBackend::synchronize(int) {
    return {{"CUDA support was not compiled into this build"}};
}

CudaBackendStats CudaBackend::stats() const noexcept { return {}; }

}  // namespace strata
