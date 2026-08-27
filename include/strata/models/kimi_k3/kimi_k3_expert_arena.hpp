#pragma once

#include "strata/models/kimi_k3/kimi_k3_checkpoint.hpp"
#include "strata/platform/result.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace strata {

// Where a run may write. Model bytes travel storage -> host RAM -> VRAM and
// nothing derived from them may reach a device the operator is protecting.
struct KimiWriteGuardConfig {
    // Disks no derived byte may be written to, by whole-disk name (`nvme0n1`).
    std::vector<std::string> forbidden_disks;
    // Paths the runtime may write to. Each is resolved to its backing disk and
    // refused if it lands on a forbidden one.
    std::vector<std::string> write_paths;
    // Lock the whole address space, current and future, with `mlockall`. This
    // is what actually closes the swap path: a locked page cannot be written to
    // swap by definition, so no model byte can reach a protected disk that way,
    // and an allocation that no longer fits fails with ENOMEM instead of paging
    // — which is the memory ceiling the charter asks for. Needs RLIMIT_MEMLOCK
    // to cover the resident set (`ulimit -l unlimited`), not root.
    bool lock_address_space{true};
    // Refuse to run while a swap file or partition sits on a forbidden disk.
    // Only consulted when `lock_address_space` is off or its lock failed: with
    // the address space locked the premise of this check — that a large
    // anonymous allocation can be paged out — is false.
    bool require_no_forbidden_swap{true};
    // Set RLIMIT_CORE to zero. A crash at 200 GiB of RSS otherwise writes a
    // core file to the working directory.
    bool disable_core_dumps{true};
};

[[nodiscard]] ValidationResult kimi_apply_write_guard(
    const KimiWriteGuardConfig& config);

// Cumulative sectors written to one whole disk, from `/sys/block/<disk>/stat`
// field 10. Sampling this before and after a run turns "no model byte reached
// the NVMe" from an assertion into a measurement.
[[nodiscard]] ParseResult<std::uint64_t> kimi_disk_sectors_written(
    const std::string& disk);
[[nodiscard]] ParseResult<std::uint64_t> kimi_disk_sectors_read(
    const std::string& disk);

struct KimiArenaConfig {
    std::uint64_t capacity_bytes{};
    // Lock the arena so the kernel cannot page 185 GiB of expert weights to a
    // swap file. Failure is reported, not absorbed: an unlocked arena silently
    // reintroduces the write path the run exists to avoid.
    bool lock_pages{true};
};

// One admitted expert's slot in the arena.
struct KimiArenaSlot {
    std::uint32_t layer{};
    std::uint32_t expert{};
    std::uint64_t offset{};
    std::uint64_t bytes{};
};

// Fixed-capacity host arena of routed experts with LRU admission. The canonical
// copy stays in the checkpoint's shards; this is a cache over them, so a miss
// is a read and never a write.
class KimiExpertArena {
public:
    KimiExpertArena() = default;
    ~KimiExpertArena();
    KimiExpertArena(KimiExpertArena&&) noexcept;
    KimiExpertArena& operator=(KimiExpertArena&&) noexcept;
    KimiExpertArena(const KimiExpertArena&) = delete;
    KimiExpertArena& operator=(const KimiExpertArena&) = delete;

    [[nodiscard]] ValidationResult reset(const KimiArenaConfig& config);
    void clear() noexcept;

    [[nodiscard]] std::uint64_t capacity_bytes() const noexcept {
        return capacity_bytes_;
    }
    [[nodiscard]] std::uint32_t slot_count() const noexcept {
        return static_cast<std::uint32_t>(slots_.size());
    }
    [[nodiscard]] bool locked() const noexcept { return locked_; }
    // The arena's base address, so whoever owns the CUDA backend can register
    // it for DMA. Pinning is not done here: the arena is a host-memory
    // component and must not depend on the device backend.
    [[nodiscard]] const void* base() const noexcept { return base_; }

    // Returns the resident bytes for one expert, or an empty span on a miss.
    [[nodiscard]] std::span<const std::byte> find(std::uint32_t layer,
                                                  std::uint32_t expert) noexcept;
    // Reserves a slot for one expert, evicting the least recently used entry
    // when the arena is full. The returned span is the destination a read
    // fills; it is not published until `publish` is called, so a failed read
    // cannot leave a half-written expert visible.
    [[nodiscard]] std::span<std::byte> reserve(std::uint32_t layer,
                                               std::uint32_t expert,
                                               std::uint64_t bytes);
    void publish(std::uint32_t layer, std::uint32_t expert) noexcept;

    [[nodiscard]] std::uint64_t hits() const noexcept { return hits_; }
    [[nodiscard]] std::uint64_t misses() const noexcept { return misses_; }
    [[nodiscard]] std::uint64_t evictions() const noexcept { return evictions_; }

private:
    struct Entry {
        std::uint64_t key{};
        std::uint64_t offset{};
        std::uint64_t bytes{};
        std::uint64_t stamp{};
        bool published{};
    };

    void release() noexcept;
    [[nodiscard]] std::int64_t locate(std::uint64_t key) const noexcept;

    std::uint64_t capacity_bytes_{};
    std::uint64_t slot_bytes_{};
    std::uint64_t clock_{};
    std::uint64_t hits_{};
    std::uint64_t misses_{};
    std::uint64_t evictions_{};
    bool locked_{};
    std::byte* base_{};
    std::vector<Entry> slots_;
};

// Where each of an expert's six payloads sits inside its arena slot.
//
// The slot mirrors the shard: `stage` sorts the six by (shard, offset) so a
// contiguous run reads as one submission, and copies them in that order. A
// reader of the slot has to apply the same order, so both sides call this
// rather than each deriving it. On this checkpoint the six are one 16.73 MiB
// extent and the slot is a byte-for-byte image of it.
struct KimiExpertSlotLayout {
    std::uint64_t gate_packed{};
    std::uint64_t gate_scale{};
    std::uint64_t up_packed{};
    std::uint64_t up_scale{};
    std::uint64_t down_packed{};
    std::uint64_t down_scale{};
    std::uint64_t total_bytes{};
};

[[nodiscard]] KimiExpertSlotLayout kimi_expert_slot_layout(
    const KimiExpertModules& modules) noexcept;

struct KimiReadRequest {
    std::uint32_t layer{};
    std::uint32_t expert{};
};

struct KimiReaderConfig {
    // Concurrent in-flight reads. The target's SATA link reaches ~400 MB/s at
    // queue depth four and only 178-238 MB/s at depth one, so a serial reader
    // runs at half speed for structural reasons rather than bandwidth ones.
    std::uint32_t queue_depth{4U};
    // O_DIRECT for bulk expert reads. Throughput is the same as buffered, but
    // a buffered read keeps a second copy of every expert in the page cache,
    // which a 185 GiB arena inside a 245 GiB budget cannot afford. It also
    // cannot dirty a page by construction. Requires 4096-byte alignment on
    // offset, length, and buffer.
    bool direct{true};
};

struct KimiReaderStats {
    std::uint64_t requests{};
    std::uint64_t bytes_read{};
    // Reads actually issued. An expert's six modules are one contiguous extent
    // in the shard, so a coalescing reader issues one per expert where a
    // per-module reader issues six. That ratio is the mechanism's whole effect
    // on per-request latency, so it is reported rather than inferred.
    std::uint64_t submissions{};
    std::uint64_t coalesced_modules{};
    std::uint64_t peak_queue_depth{};
};

// Measured background write rate of a disk, in bytes per second, sampled over
// `seconds`. A zero-write gate needs the machine's actual idle rate: this one
// idles near 19 KB/s of journald traffic, so a threshold derived from a
// remembered 0.1 KiB/s reports a failure where there is none.
[[nodiscard]] ParseResult<double> kimi_measure_idle_write_rate(
    const std::string& disk, double seconds);

// Reads routed-expert modules from the checkpoint's shards into the arena,
// keeping `queue_depth` reads in flight. Writes nothing anywhere.
class KimiExpertReader {
public:
    KimiExpertReader() = default;
    ~KimiExpertReader();
    KimiExpertReader(KimiExpertReader&&) noexcept;
    KimiExpertReader& operator=(KimiExpertReader&&) noexcept;
    KimiExpertReader(const KimiExpertReader&) = delete;
    KimiExpertReader& operator=(const KimiExpertReader&) = delete;

    [[nodiscard]] ValidationResult open(const KimiCheckpointReader& checkpoint,
                                        const KimiReaderConfig& config);
    void close() noexcept;

    // Fills the arena with every requested expert that is not already resident,
    // issuing reads at the configured queue depth.
    [[nodiscard]] ValidationResult stage(
        const KimiCheckpointReader& checkpoint, KimiExpertArena& arena,
        std::span<const KimiReadRequest> requests);

    [[nodiscard]] const KimiReaderStats& stats() const noexcept { return stats_; }
    [[nodiscard]] bool direct() const noexcept { return direct_; }

private:
    struct Shard {
        std::string name;
        int descriptor{-1};
    };

    [[nodiscard]] int descriptor_for(std::string_view shard) const noexcept;

    std::uint32_t queue_depth_{4U};
    bool direct_{};
    KimiReaderStats stats_;
    std::vector<Shard> shards_;
};

}  // namespace strata
