#pragma once

#include "strata/cuda_backend.hpp"
#include "strata/result.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace strata {

enum class Dsv4KvBlockKind : std::uint8_t {
    Sliding,
    Csa,
    Hca,
    LearnedIndex,
};

enum class Dsv4KvFormat : std::uint8_t {
    F32,
    Fp8E4m3Group64Bf16Rope,
    Fp4E2m1Group32,
    PhysicalFp8E4m3Group64Bf16Rope,
    PhysicalFp8E4m3PerTensor,
};

inline constexpr std::uint16_t kDsv4KvFormatVersion = 1U;
inline constexpr std::uint32_t kDsv4KvBlockRows = 64U;
inline constexpr std::uint16_t kDsv4KvBlockHeaderBytes = 64U;

enum class Dsv4KvCacheMode : std::uint8_t {
    ScalarOracle,
    Block,
    PhysicalDevice,
};

using Dsv4SequenceHandle = std::uint64_t;

struct Dsv4KvCacheConfig {
    std::uint32_t block_rows{kDsv4KvBlockRows};
    std::uint32_t sliding_window_rows{128U};
    std::uint64_t host_capacity_bytes{};
    std::vector<int> devices;
    std::vector<std::uint64_t> device_capacity_bytes;
    bool f32_oracle{};
    bool physical_layout{};
};

struct Dsv4KvBlockInfo {
    std::uint64_t id{};
    Dsv4SequenceHandle owner_sequence{};
    std::uint64_t logical_begin{};
    std::uint32_t used_rows{};
    std::uint32_t capacity_rows{};
    std::uint32_t row_width{};
    std::uint32_t layer{};
    std::uint32_t compression_ratio{};
    std::uint32_t refcount{};
    std::uint32_t in_flight{};
    std::uint64_t payload_bytes{};
    std::uint64_t physical_bytes{};
    Dsv4KvBlockKind kind{Dsv4KvBlockKind::Sliding};
    Dsv4KvFormat format{Dsv4KvFormat::F32};
    std::uint16_t format_version{kDsv4KvFormatVersion};
};

[[nodiscard]] Dsv4KvFormat dsv4_kv_format(
    Dsv4KvBlockKind kind, bool f32_oracle = false,
    bool physical_layout = false) noexcept;
[[nodiscard]] std::uint32_t dsv4_kv_block_rows(
    Dsv4KvBlockKind kind, std::uint32_t compression_ratio,
    bool physical_layout = false) noexcept;
[[nodiscard]] std::uint64_t dsv4_kv_row_bytes(
    Dsv4KvBlockKind kind, Dsv4KvFormat format) noexcept;
[[nodiscard]] std::uint64_t dsv4_kv_block_bytes(
    Dsv4KvBlockKind kind, Dsv4KvFormat format,
    std::uint32_t capacity_rows) noexcept;
[[nodiscard]] ValidationResult dsv4_encode_kv_row(
    Dsv4KvBlockKind kind, Dsv4KvFormat format,
    std::span<const float> values, std::span<std::byte> output);
[[nodiscard]] ValidationResult dsv4_decode_kv_row(
    Dsv4KvBlockKind kind, Dsv4KvFormat format,
    std::span<const std::byte> encoded, std::span<float> output);

struct Dsv4KvCacheStats {
    std::uint64_t host_capacity_bytes{};
    std::uint64_t host_used_bytes{};
    std::uint64_t host_peak_bytes{};
    std::vector<std::uint64_t> device_capacity_bytes;
    std::vector<std::uint64_t> device_used_bytes;
    std::vector<std::uint64_t> device_peak_bytes;
    std::uint64_t allocated_blocks{};
    std::uint64_t used_blocks{};
    std::uint64_t allocation_calls{};
    std::uint64_t allocation_nanoseconds{};
    std::uint64_t hits{};
    std::uint64_t misses{};
    std::uint64_t evictions{};
    std::uint64_t promotions{};
    std::uint64_t promotion_nanoseconds{};
    std::uint64_t host_to_device_bytes{};
    std::uint64_t device_to_host_bytes{};
    std::uint64_t host_write_bytes{};
    std::uint64_t gather_bytes{};
    std::uint64_t copy_on_write_blocks{};
    std::uint64_t sequence_creations{};
    std::uint64_t sequence_resets{};
    std::uint64_t sequence_releases{};
    std::uint64_t sequence_truncations{};
};

struct Dsv4KvCacheState;

class Dsv4KvDeviceLease {
public:
    Dsv4KvDeviceLease();
    ~Dsv4KvDeviceLease();
    Dsv4KvDeviceLease(Dsv4KvDeviceLease&&) noexcept;
    Dsv4KvDeviceLease& operator=(Dsv4KvDeviceLease&&) noexcept;
    Dsv4KvDeviceLease(const Dsv4KvDeviceLease&) = delete;
    Dsv4KvDeviceLease& operator=(const Dsv4KvDeviceLease&) = delete;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] const CudaBuffer* buffer() const noexcept;

private:
    std::shared_ptr<Dsv4KvCacheState> state_;
    std::uint64_t block_id_{};
    std::size_t device_slot_{};

    Dsv4KvDeviceLease(std::shared_ptr<Dsv4KvCacheState> state,
                      std::uint64_t block_id, std::size_t device_slot);
    void release() noexcept;
    friend class Dsv4KvCache;
};

// Owns one contiguous physical append while its exact CPU encoder
// and the following device-page patch execute in stream order.  Reservation
// advances only host block-table metadata; commit publishes the accepted row
// into both the canonical host page and caller-provided pinned patch bytes.
class Dsv4KvPhysicalAppend {
public:
    Dsv4KvPhysicalAppend();
    ~Dsv4KvPhysicalAppend();
    Dsv4KvPhysicalAppend(Dsv4KvPhysicalAppend&&) noexcept;
    Dsv4KvPhysicalAppend& operator=(Dsv4KvPhysicalAppend&&) noexcept;
    Dsv4KvPhysicalAppend(const Dsv4KvPhysicalAppend&) = delete;
    Dsv4KvPhysicalAppend& operator=(const Dsv4KvPhysicalAppend&) = delete;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] const CudaBuffer* buffer() const noexcept;
    [[nodiscard]] std::uint64_t data_offset() const noexcept;
    [[nodiscard]] std::uint64_t scale_offset() const noexcept;
    [[nodiscard]] std::uint32_t data_bytes() const noexcept;
    [[nodiscard]] std::uint32_t scale_bytes() const noexcept;
    [[nodiscard]] std::uint64_t patch_bytes() const noexcept;
    [[nodiscard]] ValidationResult commit(
        std::span<const float> values, std::span<std::byte> patch);
    [[nodiscard]] ValidationResult account();

private:
    std::shared_ptr<Dsv4KvCacheState> state_;
    Dsv4KvDeviceLease lease_;
    std::uint64_t block_id_{};
    std::uint32_t physical_row_{};
    std::uint64_t data_offset_{};
    std::uint64_t scale_offset_{};
    std::uint32_t data_bytes_{};
    std::uint32_t scale_bytes_{};
    std::byte* payload_{};
    std::uint64_t payload_bytes_{};
    std::uint32_t row_width_{};
    Dsv4KvBlockKind kind_{Dsv4KvBlockKind::Sliding};
    Dsv4KvFormat format_{Dsv4KvFormat::F32};
    bool committed_{};
    bool accounted_{};

    Dsv4KvPhysicalAppend(
        std::shared_ptr<Dsv4KvCacheState> state,
        Dsv4KvDeviceLease lease, std::uint64_t block_id,
        std::uint32_t physical_row, std::uint64_t data_offset,
        std::uint64_t scale_offset, std::uint32_t data_bytes,
        std::uint32_t scale_bytes, std::byte* payload,
        std::uint64_t payload_bytes, std::uint32_t row_width,
        Dsv4KvBlockKind kind, Dsv4KvFormat format);
    friend class Dsv4KvCache;
};

class Dsv4KvCache {
public:
    explicit Dsv4KvCache(Dsv4KvCacheConfig config,
                         CudaBackend* cuda = nullptr);

    [[nodiscard]] ValidationResult validate() const;
    [[nodiscard]] ParseResult<Dsv4SequenceHandle> create_sequence();
    [[nodiscard]] ParseResult<Dsv4SequenceHandle> fork_sequence(
        Dsv4SequenceHandle source);
    [[nodiscard]] ValidationResult reset_sequence(Dsv4SequenceHandle sequence);
    [[nodiscard]] ValidationResult truncate_sequence(
        Dsv4SequenceHandle sequence, std::uint64_t tokens);
    [[nodiscard]] ValidationResult release_sequence(Dsv4SequenceHandle sequence);

    [[nodiscard]] ValidationResult append(
        Dsv4SequenceHandle sequence, Dsv4KvBlockKind kind,
        std::uint32_t layer, std::uint32_t compression_ratio,
        std::uint64_t logical_row, std::span<const float> values,
        // A physical prefill page may append several rows before it attends
        // the first one. Keep the page's earliest causal sliding row visible
        // until that page's attend loop completes. The default preserves the
        // ordinary end-of-append sliding eviction rule.
        std::uint64_t sliding_retention_floor =
            std::numeric_limits<std::uint64_t>::max());
    [[nodiscard]] ParseResult<Dsv4KvPhysicalAppend>
    reserve_physical_append(
        Dsv4SequenceHandle sequence, Dsv4KvBlockKind kind,
        std::uint32_t layer, std::uint32_t compression_ratio,
        std::uint64_t logical_row, std::size_t device_slot);
    [[nodiscard]] ParseResult<std::vector<float>> row(
        Dsv4SequenceHandle sequence, Dsv4KvBlockKind kind,
        std::uint32_t layer, std::uint64_t logical_row);
    [[nodiscard]] ParseResult<std::vector<float>> gather(
        Dsv4SequenceHandle sequence, Dsv4KvBlockKind kind,
        std::uint32_t layer, std::span<const std::uint32_t> logical_rows);
    // Returns stable views over compact host blocks in logical-row order.
    // The views remain valid until that sequence/layer is mutated.
    [[nodiscard]] ParseResult<std::vector<CudaLightningIndexSegment>>
    learned_index_segments(Dsv4SequenceHandle sequence,
                           std::uint32_t layer, std::uint64_t rows) const;
    [[nodiscard]] ParseResult<Dsv4KvDeviceLease> acquire_device(
        Dsv4SequenceHandle sequence, Dsv4KvBlockKind kind,
        std::uint32_t layer, std::uint64_t logical_row,
        std::size_t device_slot);

    [[nodiscard]] ParseResult<std::vector<Dsv4KvBlockInfo>> block_table(
        Dsv4SequenceHandle sequence, Dsv4KvBlockKind kind,
        std::uint32_t layer) const;
    // Same table into caller-owned storage. The decode path calls this once
    // per kind per layer per token, and at the declared 1,048,576-token
    // context a compressed table holds 4,096 blocks -- about 176,000 block
    // infos per decoded token. Returning by value there both allocates on the
    // timed path, which the steady-state contract forbids, and costs measured
    // milliseconds. `output` is cleared and refilled; its capacity is reused.
    [[nodiscard]] ValidationResult block_table_into(
        Dsv4SequenceHandle sequence, Dsv4KvBlockKind kind,
        std::uint32_t layer, std::vector<Dsv4KvBlockInfo>& output) const;
    [[nodiscard]] Dsv4KvCacheStats stats() const noexcept;

private:
    std::shared_ptr<Dsv4KvCacheState> state_;
};

}  // namespace strata
