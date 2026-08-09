#pragma once

#include "strata/attention.hpp"
#include "strata/model.hpp"
#include "strata/safetensors.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace strata {

class CudaBuffer;

enum class CudaWeightEncoding : std::uint8_t {
    Plain,
    OffsetPackedInt4,
    OffsetPackedInt8,
    Fp8E4m3Block128,
    Fp4E2m1Group32,
    // compressed-tensors "nvfp4-pack-quantized": E2M1 nibble pairs with FP8
    // E4M3 group scales and one FP32 per-tensor global scale. Weights
    // dequantize to w = e2m1 * (e4m3_scale / global_scale) and the activation
    // stays FP32; this is a W4A16 path, not the W4A4 form the checkpoint's
    // input_global_scale tensors would allow.
    Nvfp4Group16,
};

struct CudaWeightDescriptor {
    CudaWeightEncoding encoding{CudaWeightEncoding::Plain};
    SafetensorsDtype dtype{SafetensorsDtype::Other};
    std::uint64_t rows{};
    std::uint64_t columns{};
    std::uint64_t packed_columns{};
    std::uint64_t scale_columns{};
    std::uint32_t group_size{};
    // Per-tensor divisor for Nvfp4Group16 group scales; unused otherwise.
    float global_scale{1.0F};
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
        // matmul_impl split on the host clock: everything before the stream
        // synchronize, and everything after it. Separates driver submission
        // cost from time genuinely spent waiting on the device.
        std::uint64_t matmul_issue_nanoseconds{};
        std::uint64_t matmul_finish_nanoseconds{};
        std::uint64_t upload_wait_nanoseconds{};
        // upload() partitioned. A pageable source makes cudaMemcpyAsync
        // synchronous inside the call, so its cost lands in
        // weight_copy_nanoseconds and not in upload_wait_nanoseconds; reading
        // only the wait counter reports an impossible H2D rate.
        std::uint64_t weight_allocation_nanoseconds{};
        std::uint64_t weight_copy_nanoseconds{};
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
        std::uint64_t flash_attention_calls{};
        std::uint64_t flash_attention_kernel_launches{};
        std::uint64_t flash_attention_h2d_transfers{};
        std::uint64_t flash_attention_d2h_transfers{};
        std::uint64_t flash_attention_h2d_bytes{};
        std::uint64_t flash_attention_d2h_bytes{};
        std::uint64_t flash_attention_useful_staging_bytes{};
        std::uint64_t flash_attention_wasted_staging_bytes{};
        std::uint64_t flash_attention_h2d_nanoseconds{};
        std::uint64_t flash_attention_kernel_nanoseconds{};
        std::uint64_t flash_attention_d2h_nanoseconds{};
        std::uint64_t flash_attention_nanoseconds{};
        std::uint64_t dsv4_paged_attention_calls{};
        std::uint64_t dsv4_paged_attention_kernel_launches{};
        std::uint64_t dsv4_paged_attention_h2d_bytes{};
        std::uint64_t dsv4_paged_attention_d2h_bytes{};
        std::uint64_t dsv4_paged_attention_page_bytes{};
        std::uint64_t dsv4_paged_attention_h2d_nanoseconds{};
        std::uint64_t dsv4_paged_attention_kernel_nanoseconds{};
        std::uint64_t dsv4_paged_attention_d2h_nanoseconds{};
        std::uint64_t dsv4_paged_attention_nanoseconds{};
        std::uint64_t dsv4_mhc_calls{};
        std::uint64_t dsv4_mhc_standalone_calls{};
        std::uint64_t dsv4_mhc_transition_calls{};
        std::uint64_t dsv4_mhc_final_calls{};
        std::uint64_t dsv4_mhc_kernel_launches{};
        std::uint64_t dsv4_mhc_resident_weight_bytes{};
        std::uint64_t dsv4_mhc_h2d_bytes{};
        std::uint64_t dsv4_mhc_d2h_bytes{};
        std::uint64_t dsv4_mhc_h2d_nanoseconds{};
        std::uint64_t dsv4_mhc_kernel_nanoseconds{};
        std::uint64_t dsv4_mhc_d2h_nanoseconds{};
        std::uint64_t dsv4_mhc_nanoseconds{};
        std::uint64_t lightning_index_calls{};
        std::uint64_t lightning_index_kernel_launches{};
        std::uint64_t lightning_index_candidates{};
        std::uint64_t lightning_index_selected{};
        std::uint64_t lightning_index_h2d_transfers{};
        std::uint64_t lightning_index_d2h_transfers{};
        std::uint64_t lightning_index_h2d_bytes{};
        std::uint64_t lightning_index_d2h_bytes{};
        std::uint64_t lightning_index_useful_selection_bytes{};
        std::uint64_t lightning_index_h2d_nanoseconds{};
        std::uint64_t lightning_index_kernel_nanoseconds{};
        std::uint64_t lightning_index_d2h_nanoseconds{};
        std::uint64_t lightning_index_nanoseconds{};
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
    std::uint64_t matmul_issue_nanoseconds{};
    std::uint64_t matmul_finish_nanoseconds{};
    std::uint64_t upload_wait_nanoseconds{};
    std::uint64_t weight_allocation_nanoseconds{};
    std::uint64_t weight_copy_nanoseconds{};
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
    std::uint64_t flash_attention_calls{};
    std::uint64_t flash_attention_kernel_launches{};
    std::uint64_t flash_attention_h2d_transfers{};
    std::uint64_t flash_attention_d2h_transfers{};
    std::uint64_t flash_attention_h2d_bytes{};
    std::uint64_t flash_attention_d2h_bytes{};
    std::uint64_t flash_attention_useful_staging_bytes{};
    std::uint64_t flash_attention_wasted_staging_bytes{};
    std::uint64_t flash_attention_h2d_nanoseconds{};
    std::uint64_t flash_attention_kernel_nanoseconds{};
    std::uint64_t flash_attention_d2h_nanoseconds{};
    std::uint64_t flash_attention_nanoseconds{};
    std::uint64_t dsv4_paged_attention_calls{};
    std::uint64_t dsv4_paged_attention_kernel_launches{};
    std::uint64_t dsv4_paged_attention_h2d_bytes{};
    std::uint64_t dsv4_paged_attention_d2h_bytes{};
    std::uint64_t dsv4_paged_attention_page_bytes{};
    std::uint64_t dsv4_paged_attention_h2d_nanoseconds{};
    std::uint64_t dsv4_paged_attention_kernel_nanoseconds{};
    std::uint64_t dsv4_paged_attention_d2h_nanoseconds{};
    std::uint64_t dsv4_paged_attention_nanoseconds{};
    std::uint64_t dsv4_mhc_calls{};
    std::uint64_t dsv4_mhc_standalone_calls{};
    std::uint64_t dsv4_mhc_transition_calls{};
    std::uint64_t dsv4_mhc_final_calls{};
    std::uint64_t dsv4_mhc_kernel_launches{};
    std::uint64_t dsv4_mhc_resident_weight_bytes{};
    std::uint64_t dsv4_mhc_h2d_bytes{};
    std::uint64_t dsv4_mhc_d2h_bytes{};
    std::uint64_t dsv4_mhc_h2d_nanoseconds{};
    std::uint64_t dsv4_mhc_kernel_nanoseconds{};
    std::uint64_t dsv4_mhc_d2h_nanoseconds{};
    std::uint64_t dsv4_mhc_nanoseconds{};
    std::uint64_t lightning_index_calls{};
    std::uint64_t lightning_index_kernel_launches{};
    std::uint64_t lightning_index_candidates{};
    std::uint64_t lightning_index_selected{};
    std::uint64_t lightning_index_h2d_transfers{};
    std::uint64_t lightning_index_d2h_transfers{};
    std::uint64_t lightning_index_h2d_bytes{};
    std::uint64_t lightning_index_d2h_bytes{};
    std::uint64_t lightning_index_useful_selection_bytes{};
    std::uint64_t lightning_index_h2d_nanoseconds{};
    std::uint64_t lightning_index_kernel_nanoseconds{};
    std::uint64_t lightning_index_d2h_nanoseconds{};
    std::uint64_t lightning_index_nanoseconds{};
    std::vector<Device> devices;
};

struct CudaDeviceMemory {
    std::uint64_t free_bytes{};
    std::uint64_t total_bytes{};
};

// Each segment contains contiguous packed FP4 E2M1 keys followed by their
// per-32 E8M0 scales. Exactly one source must be present. Device buffers let
// the indexer consume cache blocks without restaging the compressed history.
struct CudaLightningIndexSegment {
    const CudaBuffer* device_buffer{};
    std::span<const std::byte> host_bytes;
    std::uint64_t byte_offset{};
    std::uint32_t rows{};
};

struct CudaLightningIndexRequest {
    // Queries are post-projection/RoPE BF16 values. CUDA applies normalized
    // Hadamard rotation and FP4 E2M1/per-32 E8M0 simulation before scoring.
    std::span<const float> queries;
    std::span<const float> weights;
    std::span<const CudaLightningIndexSegment> segments;
    std::uint32_t heads{};
    std::uint32_t head_dim{};
    std::uint32_t top_k{};
    std::uint64_t maximum_workspace_bytes{32ULL << 20U};
};

// One persistent physical-format physical KV page. The buffer contains only
// the block-major payload: 576 data plus eight scale bytes per row.
struct CudaDsv4PhysicalPage {
    const CudaBuffer* buffer{};
    std::uint32_t rows{};
};

// Candidate order is semantically significant. Invalid padding candidates
// may use page/row zero; the backend masks them before dereferencing.
struct CudaDsv4AttentionCandidate {
    std::uint32_t page{};
    std::uint32_t row{};
    bool valid{};
};

struct CudaDsv4PagedAttentionRequest {
    // Queries have crossed the production BF16 boundary. The backend converts
    // them to BF16, reads page payloads in place, and returns the exact BF16
    // division/store output as floats for the current host-owned continuation.
    std::span<const float> queries;
    std::span<const float> head_sinks;
    std::span<const CudaDsv4PhysicalPage> pages;
    std::span<const CudaDsv4AttentionCandidate> candidates;
    float scale{};
    std::uint64_t maximum_workspace_bytes{4ULL << 20U};
};

struct CudaGlmAbsorbedAttentionRequest {
    std::span<const float> queries;
    std::span<const float> latent;
    std::span<const float> rope;
    std::span<const std::uint32_t> causal_key_counts;
    float scale{};
    std::uint64_t maximum_workspace_bytes{768ULL << 20U};
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

class CudaDsv4MhcWeights;

struct CudaDsv4AttentionPageWrite {
    const CudaBuffer* buffer{};
    std::uint64_t data_offset{};
    std::uint64_t scale_offset{};
    std::uint32_t data_bytes{};
    std::uint32_t scale_bytes{};
};

struct CudaDsv4AttentionPrepareHostView {
    std::span<const std::uint16_t> query_rank;
    std::span<const std::uint16_t> key_value;
    std::span<const float> compressor_values;
    std::span<const float> compressor_scores;
    std::span<const float> index_compressor_values;
    std::span<const float> index_compressor_scores;
    // Concatenated data/scale bytes for `page_writes`, in request order.
    std::span<std::byte> page_patches;
};

using CudaDsv4AttentionPrepareHostCallback = bool (*)(
    void* context, const CudaDsv4AttentionPrepareHostView& view);

// Completes the accepted 64-head DeepSeek attention path without returning
// the attended activation to the host. The two rank-local 32-head score and
// finish groups retain their accepted arithmetic; inverse RoPE and both FP8
// output projections follow in stream order. The final BF16 branch is placed
// directly in the active persistent mHC workspace. A diagnostic branch output
// may be requested by the caller without changing the device computation.
struct CudaDsv4PagedAttentionMhcRequest {
    CudaDsv4PagedAttentionRequest attention;
    std::span<const float> inverse_rope_cosines;
    std::span<const float> inverse_rope_sines;
    const CudaWeight* output_a{};
    const CudaWeight* output_b{};
    int mhc_device{-1};
    // Optional same-device transition into the following FFN branch. Its
    // normalized layer input is the only transition state returned to host.
    const CudaDsv4MhcWeights* mhc_transition{};
    std::span<float> mhc_layer_input;
    // Optional resident router consumed from the transition's exact BF16
    // layer input. Logits remain raw F32 for the host selection contract.
    const CudaWeight* router{};
    std::span<float> router_logits;
    // Keep the normalized BF16 input and raw router logits device-resident for
    // the immediately following stream-ordered CPU-MoE command. Both host
    // output spans above must be empty in this mode.
    bool defer_host_moe_input{};
};

// Produces the accepted BF16 Q/KV boundary from the persistent mHC layer
// input. The 64x512 query remains device-resident for the immediately
// following paged-attention command. Query-rank, KV, and optional raw-F32
// compressor projections return only for the still-host-owned index,
// compression-state, and page-allocation arithmetic.
struct CudaDsv4AttentionPrepareRequest {
    const CudaWeight* query_a{};
    const CudaWeight* query_b{};
    const CudaWeight* key_value{};
    const CudaWeight* compressor_value{};
    const CudaWeight* compressor_gate{};
    const CudaWeight* index_compressor_value{};
    const CudaWeight* index_compressor_gate{};
    // Optional accepted mHC transition that produces this same-GPU
    // attention input immediately before QKV preparation. The backend owns
    // the one combined completion wait.
    const CudaDsv4MhcWeights* mhc_transition{};
    std::span<const float> query_norm;
    std::span<const float> key_value_norm;
    std::span<const float> rope_cosines;
    std::span<const float> rope_sines;
    // Optional host bridge when the layer device differs from `mhc_device`.
    // When empty, the backend forwards the persistent mHC layer input through
    // its fixed pinned cross-device slot using stream events, without a host
    // wait.
    std::span<const float> cross_device_input;
    // Optional stream host node that consumes the exact prepared outputs and
    // fills page-patch staging without forcing a host continuation.  Every
    // target is patched after the callback on this same stream.
    CudaDsv4AttentionPrepareHostCallback host_callback{};
    void* host_callback_context{};
    std::span<const CudaDsv4AttentionPageWrite> page_writes;
    int mhc_device{-1};
    std::uint64_t maximum_workspace_bytes{1ULL << 20U};
};

class CudaBuffer {
public:
    CudaBuffer();
    ~CudaBuffer();
    CudaBuffer(CudaBuffer&&) noexcept;
    CudaBuffer& operator=(CudaBuffer&&) noexcept;
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint64_t device_bytes() const noexcept;
    [[nodiscard]] int device() const noexcept;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class CudaBackend;
};

struct CudaBufferPatch {
    std::uint64_t offset{};
    std::span<const std::byte> bytes;
};

// Persistent target-format inputs for one DeepSeek mHC pre boundary. The
// projection remains F32 as in the accepted SM86 contract; scale/base and the
// BF16 norm weight are packed into one immutable auxiliary allocation.
class CudaDsv4MhcWeights {
public:
    CudaDsv4MhcWeights();
    ~CudaDsv4MhcWeights();
    CudaDsv4MhcWeights(CudaDsv4MhcWeights&&) noexcept;
    CudaDsv4MhcWeights& operator=(CudaDsv4MhcWeights&&) noexcept;
    CudaDsv4MhcWeights(const CudaDsv4MhcWeights&) = delete;
    CudaDsv4MhcWeights& operator=(const CudaDsv4MhcWeights&) = delete;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint64_t device_bytes() const noexcept;
    [[nodiscard]] int device() const noexcept;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class CudaBackend;
};

// One exact DeepSeek expert projection triplet. The weight objects must remain
// alive until the matching collect call completes. Each routed coefficient is
// applied once before w2; the optional shared expert must use coefficient 1.0.
struct CudaDeepSeekMoeExpert {
    const CudaWeight* w1{};
    const CudaWeight* w3{};
    const CudaWeight* w2{};
    float coefficient{1.0F};
};

// One stream-ordered CPU-MoE callback. The backend supplies two contiguous
// rank-local FP32 partial outputs in pinned storage. The callback may perform
// host arithmetic only; CUDA APIs are forbidden by cudaLaunchHostFunc. A false
// return fails the command after the stream is safely drained.
using CudaDsv4HostMoeCallback = bool (*)(
    void* context, std::span<float> rank_partials);

// Device-owned form of the same callback boundary. The backend first stages
// the persistent mHC BF16 layer input and raw FP32 router logits in stream
// order. The callback performs host arithmetic only and fills the same two
// rank-local FP32 partial outputs.
using CudaDsv4DeviceInputHostMoeCallback = bool (*)(
    void* context, std::span<const std::uint16_t> encoded_hidden,
    std::span<const float> router_logits, std::span<float> rank_partials);

// Stream-ordered exact output-head reduction. The callback consumes the final
// four-copy BF16 mHC residual and writes the normalized 4096-wide FP32 input
// for the resident vocabulary projection. It may perform host arithmetic only;
// the matching MoE collection is the sole completion wait for the whole token.
using CudaDsv4MhcHeadCallback = bool (*)(
    void* context, std::span<const std::uint16_t> encoded_hidden,
    std::span<float> reduced);

// One routed expert of a prefill page together with every page row that
// selected it. The page path exists because the single-row command above reads
// a 13.37 MB expert triplet from HBM to serve one row: at a 512-row page a hot
// expert is read once per row that chose it, roughly twelve times over. Here
// the weight read is hoisted out of the row loop, so one read serves the whole
// group. `rows` indexes the page's hidden rows and `coefficients` carries that
// row's router coefficient in the same order; both spans must have equal size
// and stay alive until collection completes.
struct CudaDeepSeekMoeRowGroup {
    const CudaWeight* w1{};
    const CudaWeight* w3{};
    const CudaWeight* w2{};
    std::span<const std::uint32_t> rows;
    std::span<const float> coefficients;
};

// Model-neutral gate/up/down expert descriptor for the common device MoE
// workspace. Weight objects must remain alive until collection completes.
struct CudaMoeExpert {
    const CudaWeight* gate{};
    const CudaWeight* up{};
    const CudaWeight* down{};
    float coefficient{1.0F};
};

// One resident Gemma 4 decode layer. The backend queues the complete layer on
// one stream, so hidden state and KV data stay on the device between kernels.
struct CudaGemma4DecodeLayer {
    const CudaWeight* query{};
    const CudaWeight* key{};
    const CudaWeight* value{};
    const CudaWeight* output{};
    const CudaWeight* gate{};
    const CudaWeight* up{};
    const CudaWeight* down{};
    const CudaBuffer* input_norm{};
    const CudaBuffer* post_attention_norm{};
    const CudaBuffer* pre_feedforward_norm{};
    const CudaBuffer* post_feedforward_norm{};
    const CudaBuffer* query_norm{};
    const CudaBuffer* key_norm{};
    const CudaBuffer* kv_cache{};
    std::span<std::uint16_t> next_keys;
    std::span<std::uint16_t> next_values;
    std::uint32_t cache_capacity_rows{};
    std::uint32_t cache_start{};
    std::uint32_t cached_rows{};
    float scalar{1.0F};
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
    // Page-locks a host region so H2D transfers DMA straight out of it instead
    // of going through the driver's staging copy. Measured on the resident
    // weight arena, a cold 4.46 MB slice costs 1.32 ms pageable and 0.37 ms
    // pinned: the staging copy dominates because every expert is a different
    // cold slice of a 147 GB mapping, not a reused warm buffer.
    // Registration itself runs at about 2.7 GB/s, so it is a load-time cost.
    [[nodiscard]] ValidationResult register_host_memory(const void* base,
                                                       std::uint64_t bytes);
    void unregister_host_memory(const void* base) noexcept;
    // Deferred completion leaves the upload's copies in flight on the device
    // stream instead of waiting for them. Every consumer of a weight is issued
    // on that same stream, so the device side is already ordered; what the wait
    // actually protects is the host source buffer, which must outlive the copy.
    // Deferring is therefore legal only when the payload spans point at
    // storage that outlives the batch -- in practice the pinned resident
    // arena, never a temporary decoded into by the caller.
    //
    // The caller must call synchronize_uploads() for every device it touched
    // before the batch's source buffers can go away, and to collect the wait.
    // Until then the transfer is unaccounted: upload_wait_nanoseconds is
    // attributed at the synchronize, not here.
    enum class UploadCompletion : std::uint8_t {
        Synchronous,
        Deferred,
    };
    [[nodiscard]] ValidationResult upload(
        int device, const CudaWeightDescriptor& descriptor,
        std::span<const std::byte> weights, std::span<const std::byte> scales,
        CudaWeight& output,
        UploadCompletion completion = UploadCompletion::Synchronous);
    // Waits out any deferred uploads issued on this device and attributes the
    // wait. Idempotent, and free when nothing was deferred.
    [[nodiscard]] ValidationResult synchronize_uploads(int device);
    [[nodiscard]] ValidationResult upload_buffer(
        int device, std::span<const std::byte> bytes, CudaBuffer& output);
    // Enqueues patches from stable host storage on the buffer's command stream.
    // The source spans must remain valid until the stream next completes.
    [[nodiscard]] ValidationResult update_buffer(
        const CudaBuffer& buffer, std::span<const CudaBufferPatch> patches);
    [[nodiscard]] ValidationResult allocate_buffer(
        int device, std::uint64_t bytes, CudaBuffer& output);
    [[nodiscard]] ValidationResult upload_gemma4_kv(
        const CudaBuffer& cache, std::span<const std::uint16_t> keys,
        std::span<const std::uint16_t> values, std::uint32_t start,
        std::uint32_t capacity_rows, std::uint32_t columns);
    [[nodiscard]] ValidationResult gemma4_decode_layers(
        int device, std::span<const CudaGemma4DecodeLayer> layers,
        std::span<const float> input, std::uint32_t position,
        std::span<float> output);
    [[nodiscard]] ValidationResult matmul(
        const CudaWeight& weight, std::span<const float> input,
        std::uint32_t rows, std::span<float> output);
    [[nodiscard]] ValidationResult matmul_softcap(
        const CudaWeight& weight, std::span<const float> input,
        float softcap, std::span<float> output);
    [[nodiscard]] ValidationResult matmul_grouped(
        const CudaWeight& weight, std::span<const float> input,
        std::uint32_t groups, std::uint64_t rows_per_group,
        std::span<float> output);
    [[nodiscard]] ValidationResult matmul_grouped_rows(
        const CudaWeight& weight, std::span<const float> input,
        std::uint32_t rows, std::uint32_t groups,
        std::uint64_t rows_per_group, std::span<float> output);
    // Validate an explicitly requested FlashAttention device before model
    // admission. Shape-aware dispatch must not hide an unsupported
    // architecture until a later, longer request reaches the CUDA branch.
    [[nodiscard]] ValidationResult validate_flash_attention_device(
        int device) const;
    [[nodiscard]] ValidationResult validate_lightning_index_device(
        int device) const;
    [[nodiscard]] ValidationResult validate_dsv4_mhc_device(
        int device) const;
    // Executes the model-neutral forward attention primitive under the
    // request's explicit numerical contract. Host segments are packed into
    // bounded reusable device workspaces; indexed rows are gathered exactly in
    // descriptor order. Unsupported devices or shapes return an error and
    // never select another numerical path silently.
    [[nodiscard]] ValidationResult flash_attention(
        int device, const FlashAttentionRequest& request,
        std::span<float> output);
    // Exact SM86 DeepSeek physical-page attention. Historical KV remains in
    // the supplied device pages; only query/sink/descriptor metadata crosses
    // H2D. Unsupported shapes and devices fail without a fallback.
    [[nodiscard]] ValidationResult dsv4_paged_attention(
        int device, const CudaDsv4PagedAttentionRequest& request,
        std::span<float> output);
    [[nodiscard]] ValidationResult dsv4_paged_attention_to_mhc(
        int device, const CudaDsv4PagedAttentionMhcRequest& request,
        std::span<float> diagnostic_branch = {});
    [[nodiscard]] ValidationResult dsv4_prepare_attention(
        int device, const CudaDsv4AttentionPrepareRequest& request,
        std::span<float> query_rank, std::span<float> key_value,
        std::span<float> compressor_values = {},
        std::span<float> compressor_scores = {},
        std::span<float> index_compressor_values = {},
        std::span<float> index_compressor_scores = {});
    // Accepted dsv4-sm86-v1 mHC sequence. `begin` uploads one initial
    // four-copy residual and retains it on the device. Each transition fuses
    // the preceding post mix with the next projection/Sinkhorn/pre/RMSNorm;
    // only the 4096-wide branch output crosses H2D and the next layer input
    // crosses D2H. The weighted state is an optional diagnostic output.
    // `finish` applies the final post and downloads the four-copy state.
    // Command order is strict and has no host-arithmetic fallback.
    [[nodiscard]] ValidationResult upload_dsv4_mhc_weights(
        int device, std::span<const float> projection,
        std::span<const float> scale, std::span<const float> base,
        std::span<const float> norm_weight, CudaDsv4MhcWeights& output);
    [[nodiscard]] ValidationResult dsv4_mhc_begin(
        int device, const CudaDsv4MhcWeights& weights,
        std::span<const float> hidden, std::span<float> weighted,
        std::span<float> layer_input);
    [[nodiscard]] ValidationResult dsv4_mhc_begin_device(
        int device, const CudaDsv4MhcWeights& weights,
        std::span<const float> hidden);
    [[nodiscard]] ValidationResult dsv4_mhc_transition(
        int device, const CudaDsv4MhcWeights& next_weights,
        std::span<const float> branch_output, std::span<float> weighted,
        std::span<float> layer_input,
        std::span<float> post_residual = {});
    // Consumes the exact BF16 branch result left in the persistent mHC
    // workspace by enqueue_dsv4_host_moe_from_mhc. No branch activation is
    // uploaded; diagnostic weighted/layer/residual outputs remain optional.
    [[nodiscard]] ValidationResult dsv4_mhc_transition_device(
        int device, const CudaDsv4MhcWeights& next_weights,
        std::span<float> weighted, std::span<float> layer_input,
        std::span<float> post_residual = {});
    [[nodiscard]] ValidationResult dsv4_mhc_finish(
        int device, std::span<const float> branch_output,
        std::span<float> hidden);
    [[nodiscard]] ValidationResult dsv4_mhc_finish_device(
        int device, std::span<float> hidden);
    [[nodiscard]] ValidationResult reserve_dsv4_mhc_head(
        int device, std::uint64_t logits);
    [[nodiscard]] ValidationResult enqueue_dsv4_mhc_finish_head_device(
        int device, const CudaWeight& head,
        CudaDsv4MhcHeadCallback callback, void* callback_context);
    // Called only after collect_deepseek_moe has completed the same stream.
    // This copies the already-staged logits and validates the host node without
    // issuing or waiting on CUDA work.
    [[nodiscard]] ValidationResult complete_dsv4_mhc_head_device(
        int device, std::span<float> logits);
    // Exact GLM-5.2 MLA weight absorption for the dense causal window. It
    // avoids materializing per-head K/V from the compact 512+64 cache.
    [[nodiscard]] ValidationResult glm_absorbed_attention(
        const CudaWeight& key_value_projection,
        const CudaGlmAbsorbedAttentionRequest& request,
        std::span<float> output);
    // Exact bounded-workspace DeepSeek Lightning Indexer. Output contains
    // min(top_k, candidates) positions in descending score order with lower
    // positions winning ties. Unsupported shapes fail; no scalar fallback is
    // selected inside the backend.
    [[nodiscard]] ValidationResult lightning_index(
        int device, const CudaLightningIndexRequest& request,
        std::span<std::uint32_t> output);
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
    // Runs the shared expert on an auxiliary stream while `callback` produces
    // two CPU rank partials. The main stream uploads those fixed partials,
    // applies the exact BF16 routed/shared join on device, and exposes the
    // combined result through collect_deepseek_moe's `shared_output` span.
    [[nodiscard]] ValidationResult enqueue_dsv4_host_moe(
        int device, std::span<const float> hidden,
        const CudaDeepSeekMoeExpert& shared, float swiglu_limit,
        CudaDsv4HostMoeCallback callback, void* callback_context);
    // Uses the current persistent mHC layer_input as the GPU shared-expert
    // source and writes the exact routed/shared BF16 result back to the mHC
    // branch buffer. The CPU callback still consumes the already-decoded host
    // layer input, so its arithmetic and worker placement are unchanged.
    [[nodiscard]] ValidationResult enqueue_dsv4_host_moe_from_mhc(
        int device, const CudaDeepSeekMoeExpert& shared, float swiglu_limit,
        CudaDsv4HostMoeCallback callback, void* callback_context);
    [[nodiscard]] ValidationResult enqueue_dsv4_host_moe_from_device_input(
        int device, const CudaDeepSeekMoeExpert& shared, float swiglu_limit,
        CudaDsv4DeviceInputHostMoeCallback callback,
        void* callback_context);
    [[nodiscard]] ValidationResult collect_deepseek_moe(
        int device, std::span<float> routed_output,
        std::span<float> shared_output);
    // Row-grouped form of the command above. `hidden` holds `hidden_rows`
    // contiguous activation rows; each group names one expert and the rows it
    // serves. Routed results are flattened group-major and, within a group, in
    // the group's own row order, so the caller reconstructs a row's rank order
    // from the offsets it built the groups with. When `shared` is present it
    // runs over `shared_rows` and its results are flattened in that order.
    // Per-row arithmetic is identical to issuing one single-row command per
    // (expert, row): only the weight read is shared.
    [[nodiscard]] ValidationResult enqueue_deepseek_moe_rows(
        int device, std::span<const float> hidden, std::uint32_t hidden_rows,
        std::span<const CudaDeepSeekMoeRowGroup> groups,
        const CudaDeepSeekMoeExpert* shared,
        std::span<const std::uint32_t> shared_rows, float swiglu_limit);
    [[nodiscard]] ValidationResult collect_deepseek_moe_rows(
        int device, std::span<float> routed_output,
        std::span<float> shared_output);
    [[nodiscard]] ValidationResult enqueue_moe(
        int device, std::span<const float> hidden, std::uint32_t rows,
        std::span<const CudaMoeExpert> routed,
        const CudaMoeExpert* shared = nullptr);
    [[nodiscard]] ValidationResult collect_moe(
        int device, std::span<float> routed_output,
        std::span<float> shared_output = {});
    [[nodiscard]] ValidationResult synchronize(int device);

    [[nodiscard]] CudaBackendStats stats() const noexcept;

private:
    [[nodiscard]] ValidationResult dsv4_mhc_begin_impl(
        int device, const CudaDsv4MhcWeights& weights,
        std::span<const float> hidden, std::span<float> weighted,
        std::span<float> layer_input, bool device_only);
    [[nodiscard]] ValidationResult dsv4_mhc_transition_impl(
        int device, const CudaDsv4MhcWeights& next_weights,
        std::span<const float> branch_output, std::span<float> weighted,
        std::span<float> layer_input, std::span<float> post_residual,
        bool device_branch);
    [[nodiscard]] ValidationResult dsv4_mhc_finish_impl(
        int device, std::span<const float> branch_output,
        std::span<float> hidden, bool device_branch);
    [[nodiscard]] ValidationResult enqueue_dsv4_host_moe_impl(
        int device, std::span<const float> hidden,
        const CudaDeepSeekMoeExpert& shared, float swiglu_limit,
        CudaDsv4HostMoeCallback callback, void* callback_context,
        CudaDsv4DeviceInputHostMoeCallback device_input_callback,
        bool mhc_source_and_destination);
    [[nodiscard]] ValidationResult matmul_impl(
        const CudaWeight& weight, std::span<const float> input,
        std::uint32_t rows, std::uint32_t groups,
        std::uint64_t rows_per_group, std::span<float> output,
        float softcap);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata
