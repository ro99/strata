#pragma once

#include "strata/cuda_backend.hpp"
#include "strata/result.hpp"

#include <cstdint>
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
};

enum class Dsv4KvCacheMode : std::uint8_t {
    ScalarOracle,
    Block,
};

using Dsv4SequenceHandle = std::uint64_t;

struct Dsv4KvCacheConfig {
    std::uint32_t block_rows{64U};
    std::uint32_t sliding_window_rows{128U};
    std::uint64_t host_capacity_bytes{};
    std::vector<int> devices;
    std::vector<std::uint64_t> device_capacity_bytes;
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
    Dsv4KvBlockKind kind{Dsv4KvBlockKind::Sliding};
    Dsv4KvFormat format{Dsv4KvFormat::F32};
    std::vector<bool> device_resident;
};

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
        std::uint64_t logical_row, std::span<const float> values);
    [[nodiscard]] ParseResult<std::span<const float>> row(
        Dsv4SequenceHandle sequence, Dsv4KvBlockKind kind,
        std::uint32_t layer, std::uint64_t logical_row);
    [[nodiscard]] ParseResult<std::vector<float>> gather(
        Dsv4SequenceHandle sequence, Dsv4KvBlockKind kind,
        std::uint32_t layer, std::span<const std::uint32_t> logical_rows);
    [[nodiscard]] ParseResult<Dsv4KvDeviceLease> acquire_device(
        Dsv4SequenceHandle sequence, Dsv4KvBlockKind kind,
        std::uint32_t layer, std::uint64_t logical_row,
        std::size_t device_slot);

    [[nodiscard]] ParseResult<std::vector<Dsv4KvBlockInfo>> block_table(
        Dsv4SequenceHandle sequence, Dsv4KvBlockKind kind,
        std::uint32_t layer) const;
    [[nodiscard]] Dsv4KvCacheStats stats() const noexcept;

private:
    std::shared_ptr<Dsv4KvCacheState> state_;
};

}  // namespace strata
